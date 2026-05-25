# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Speculative decoding utilities for RL training.

This module implements:
1. spec_cut / spec_cut_with_knobs: Metropolis-Hastings based token-level truncation
2. rand_reuse_cut / rand_reuse_all_cut: Random baselines for ablation
3. build_ctx: Construct prefix-reuse context for partial regeneration
4. AdaptiveWindowBucket: Per-bucket window size management for adaptive speculation
"""

import math
from collections import defaultdict, deque

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_like_compat(t: torch.Tensor, seed: int | None = None):
    """Generate random tensor like `t`, optionally with a local seed.

    NOTE: Never touches the global RNG state.
    """
    if seed is None:
        return torch.rand_like(t)
    # Always create the generator on CPU (most compatible), then move if needed
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    rnd = torch.rand(t.shape, generator=g, dtype=torch.float32)
    rnd = rnd.to(device=t.device, dtype=t.dtype)
    return rnd


# ---------------------------------------------------------------------------
# Core speculative cut (Metropolis-Hastings)
# ---------------------------------------------------------------------------


@torch.no_grad()
def spec_cut(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    response_mask: torch.Tensor,
    p_abs_thresh: float | None = None,
    seed: int | None = None,
):
    """Token-level speculative truncation via Metropolis-Hastings acceptance.

    For each token, accept if new_logp >= old_logp OR with probability
    exp(new_logp - old_logp).  Find the first rejected token → `cut_idx`;
    tokens *before* `cut_idx` can be safely reused.

    Returns
    -------
    dict with keys: cut_idx, resp_len, idx_reuse, idx_need,
                    per_request_max_new_tokens, metrics
    """
    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim() == 2
    B, R = new_logp.shape
    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    # delta log-prob
    log_ratio = (new_logp - old_logp).masked_fill(~valid, 0.0)

    U = _rand_like_compat(new_logp, seed=seed).clamp_min(1e-12)
    logU = torch.log(U)

    # accept if: not valid, or delta >= 0, or logU <= delta
    accept_mask = (~valid) | (log_ratio >= 0) | (logU <= log_ratio)

    # optional absolute threshold
    if p_abs_thresh is not None and p_abs_thresh > 0.0:
        log_th = math.log(p_abs_thresh)
        accept_mask &= (new_logp >= log_th) | (~valid)

    bad = valid & (~accept_mask)
    has_bad = bad.any(dim=1)
    # argmax returns 0 when all False — handle explicitly
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)
    first_bad = torch.where(has_bad, first_bad, torch.full_like(first_bad, R))
    cut_idx = torch.where(has_bad, first_bad, resp_len)

    reuse_mask = cut_idx == resp_len
    need_mask = ~reuse_mask
    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need = torch.nonzero(need_mask, as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = (resp_len - cut_idx).clamp(min=0)
    metrics = {
        "spec/skip_ratio": reuse_mask.float().mean().item(),
        "spec/cont_ratio": need_mask.float().mean().item(),
        "spec/avg_cut_idx": cut_idx.float().mean().item(),
        "spec/avg_resp_len": resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
    }
    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse,
        "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


@torch.no_grad()
def spec_cut_with_knobs(
    old_logp: torch.Tensor,  # [B,R]
    new_logp: torch.Tensor,  # [B,R]
    response_mask: torch.Tensor,  # [B,R] 1=valid response token
    *,
    bias: float = 0.0,
    scale: float = 1.0,
    p_abs_thresh: float | None = None,
    seed: int | None = None,
):
    """spec_cut with explicit bias/scale controls.

    * bias > 0  → more strict (harder to accept)
    * bias < 0  → more lenient (easier to accept)
    * scale > 1 → amplify differences (more strict)
    * scale < 1 → shrink differences (more lenient)
    """
    assert old_logp.shape == new_logp.shape == response_mask.shape and old_logp.dim() == 2
    B, R = new_logp.shape
    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    delta = (new_logp - old_logp).masked_fill(~valid, 0.0)
    delta2 = scale * (delta - bias)

    logU = torch.log(_rand_like_compat(new_logp, seed=seed).clamp_min(1e-12))
    accept = (~valid) | (delta2 >= 0) | (logU <= delta2)

    if p_abs_thresh is not None and p_abs_thresh > 0.0:
        log_th = math.log(p_abs_thresh)
        accept &= (new_logp >= log_th) | (~valid)

    bad = valid & (~accept)
    has_bad = bad.any(dim=1)
    first_bad = torch.argmax(bad.to(torch.int8), dim=1)
    # Explicitly handle "no bad" case: set to R instead of 0
    first_bad = torch.where(has_bad, first_bad, torch.full_like(first_bad, R))
    cut_idx = torch.where(has_bad, first_bad, resp_len)

    reuse_mask = cut_idx == resp_len
    need_mask = ~reuse_mask
    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need = torch.nonzero(need_mask, as_tuple=False).squeeze(-1)

    saved_tokens = cut_idx.to(torch.float32)
    # Per-sample reuse ratio: cut_idx / resp_len (0~1)
    # Guard against resp_len=0 (empty response) → avoid NaN
    safe_resp_len = resp_len.clone()
    safe_resp_len[safe_resp_len == 0] = 1  # avoid div-by-zero; cut_idx is also 0 here
    reuse_ratio = (cut_idx.to(torch.float32) / safe_resp_len.to(torch.float32)).clamp(0, 1)
    metrics = {
        "spec/skip_ratio": reuse_mask.float().mean().item(),
        "spec/cont_ratio": need_mask.float().mean().item(),
        "spec/avg_cut_idx": cut_idx.float().mean().item(),
        "spec/avg_resp_len": resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/avg_reuse_ratio": reuse_ratio.mean().item(),
        "spec/reuse_ratio_min": reuse_ratio.min().item(),
        "spec/reuse_ratio_max": reuse_ratio.max().item(),
        "spec/bias": float(bias),
        "spec/scale": float(scale),
    }
    return {
        "cut_idx": cut_idx,
        "idx_reuse": idx_reuse,
        "idx_need": idx_need,
        "resp_len": resp_len,
        "per_request_max_new_tokens": (R - cut_idx).to(torch.long),
        "per_sample_reuse_ratio": reuse_ratio,  # [B] tensor, 0~1
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Random baselines
# ---------------------------------------------------------------------------


@torch.no_grad()
def rand_reuse_cut(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    reuse_prob: float,
    seed: int | None = None,
):
    """Row-wise random decision: reuse whole row with probability `reuse_prob`."""
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))

    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    row_rand = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)
    reuse_mask = row_rand < reuse_prob
    need_mask = ~reuse_mask

    zero_like = torch.zeros_like(resp_len)
    cut_idx = torch.where(reuse_mask, resp_len, zero_like)
    cut_idx = cut_idx.clamp(min=0, max=resp_len)

    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need = torch.nonzero(need_mask, as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio": reuse_mask.float().mean().item(),
        "spec/cont_ratio": need_mask.float().mean().item(),
        "spec/avg_cut_idx": cut_idx.float().mean().item(),
        "spec/avg_resp_len": resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_p": float(reuse_prob),
    }
    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse,
        "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


@torch.no_grad()
def rand_reuse_all_cut(
    old_logp: torch.Tensor,
    new_logp: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    reuse_prob: float,
    seed: int | None = None,
):
    """Randomly select a truncation point for every row."""
    assert response_mask.dim() == 2
    B, R = response_mask.shape
    reuse_prob = float(max(0.0, min(1.0, reuse_prob)))

    valid = response_mask.bool()
    resp_len = valid.sum(dim=1).to(torch.long)

    row_rand = _rand_like_compat(new_logp[:, :1], seed=seed).squeeze(-1)
    cut_idx = torch.floor(row_rand * resp_len.to(torch.float32)).to(torch.long)
    cut_idx = cut_idx.clamp(min=0, max=resp_len)

    reuse_mask = cut_idx == resp_len
    need_mask = ~reuse_mask
    idx_reuse = torch.nonzero(reuse_mask, as_tuple=False).squeeze(-1)
    idx_need = torch.nonzero(need_mask, as_tuple=False).squeeze(-1)
    per_request_max_new_tokens = (R - cut_idx).to(torch.long)

    saved_tokens = cut_idx.to(torch.float32)
    metrics = {
        "spec/skip_ratio": reuse_mask.float().mean().item(),
        "spec/cont_ratio": need_mask.float().mean().item(),
        "spec/avg_cut_idx": cut_idx.float().mean().item(),
        "spec/avg_resp_len": resp_len.float().mean().item(),
        "spec/avg_saved_tokens": saved_tokens.float().mean().item(),
        "spec/random_reuse_all_p": float(reuse_prob),
    }
    return {
        "cut_idx": cut_idx,
        "resp_len": resp_len,
        "idx_reuse": idx_reuse,
        "idx_need": idx_need,
        "per_request_max_new_tokens": per_request_max_new_tokens,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Adaptive Window Bucket (new)
# ---------------------------------------------------------------------------


class AdaptiveWindowBucket:
    """Manage per-bucket window sizes for adaptive speculative decoding.

    Bucket layout (response_len = R):
        buckets = [R/16, R/8, R/4, R/2, 3*R/4, R]

    Rules
    -----
    * At least one full window accepted:
        next_bucket = map_to_nearest(floor(cut_idx / W_base) * W_base)
    * No full window accepted (r = cut_idx / W_base):
        r >= 0.75   → next = R/4
        0.4 <= r < 0.75 → next = R/8
        0.15 <= r < 0.4 → next = R/16
        r < 0.15    → disable speculation
    * Consecutive failure → decrease one bucket.
    * Two consecutive full accepts at current bucket → increase one bucket.
    """

    def __init__(self, response_len: int):
        R = response_len
        self.R = R
        self.W_base = R / 4.0
        # Fixed 6 buckets from smallest to largest
        self.buckets = [R // 16, R // 8, R // 4, R // 2, 3 * R // 4, R]
        # Ensure all buckets >= 1
        self.buckets = [max(1, b) for b in self.buckets]
        self.num_buckets = len(self.buckets)
        # Start at R/4 (index 2)
        self.current_idx = 2
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.enabled = True

    @property
    def current_window(self) -> int:
        """Current speculative window size in tokens."""
        if not self.enabled:
            return 0
        return self.buckets[self.current_idx]

    def map_to_nearest_bucket(self, value: int) -> int:
        """Map a token count to the nearest bucket index."""
        value = max(1, min(value, self.R))
        best_idx = 0
        best_diff = abs(value - self.buckets[0])
        for i, b in enumerate(self.buckets[1:], start=1):
            diff = abs(value - b)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def update(self, cut_idx: int, resp_len: int) -> dict:
        """Update bucket based on this round's cut result.

        Parameters
        ----------
        cut_idx : int
            First rejected token position (or resp_len if all accepted).
        resp_len : int
            Actual response length.

        Returns
        -------
        info dict with keys: prev_window, next_window, action, enabled
        """
        R = self.R
        W_base = self.W_base
        prev_window = self.current_window
        action = "keep"

        # Determine if at least one full window was accepted
        full_window_accepted = cut_idx >= int(W_base)

        if full_window_accepted:
            # Rule 1: at least one full window accepted
            raw_bucket = int((cut_idx // W_base) * W_base)
            next_idx = self.map_to_nearest_bucket(raw_bucket)
            next_idx = max(next_idx, self.current_idx)  # don't go below current
            next_idx = min(next_idx, self.num_buckets - 1)

            if next_idx > self.current_idx:
                action = "promote_window"
                self.consecutive_successes = 0
            elif next_idx == self.current_idx:
                self.consecutive_successes += 1
                if self.consecutive_successes >= 2 and self.current_idx < self.num_buckets - 1:
                    # Two consecutive full accepts → increase one bucket
                    next_idx = self.current_idx + 1
                    action = "promote_success"
                    self.consecutive_successes = 0
            else:
                self.consecutive_successes = 0

            self.consecutive_failures = 0
            self.current_idx = next_idx
            self.enabled = True

        else:
            # Rule 2: no full window accepted
            r = cut_idx / W_base if W_base > 0 else 0.0

            if r >= 0.75:
                target_idx = 2  # R/4
            elif r >= 0.4:
                target_idx = 1  # R/8
            elif r >= 0.15:
                target_idx = 0  # R/16
            else:
                target_idx = -1  # disable

            self.consecutive_failures += 1
            # Consecutive failure: decrease one bucket
            if target_idx >= 0:
                target_idx = max(0, target_idx - (self.consecutive_failures - 1))
                target_idx = min(target_idx, self.current_idx)
            else:
                # Below threshold → disable
                self.enabled = False
                action = "disable"

            if self.enabled:
                if target_idx < self.current_idx:
                    action = "demote"
                self.current_idx = max(0, target_idx)
                self.consecutive_successes = 0

        return {
            "prev_window": prev_window,
            "next_window": self.current_window,
            "action": action,
            "enabled": self.enabled,
            "bucket_idx": self.current_idx,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }

    def reset(self):
        """Reset to initial state (e.g., on epoch boundary)."""
        self.current_idx = 2
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.enabled = True

    def state_dict(self):
        return {
            "R": self.R,
            "current_idx": self.current_idx,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "enabled": self.enabled,
        }

    def load_state_dict(self, state):
        self.R = state.get("R", self.R)
        self.current_idx = state.get("current_idx", 2)
        self.consecutive_failures = state.get("consecutive_failures", 0)
        self.consecutive_successes = state.get("consecutive_successes", 0)
        self.enabled = state.get("enabled", True)


# ---------------------------------------------------------------------------
# Context builder (unchanged logic, minor robustness fixes)
# ---------------------------------------------------------------------------


def build_ctx(
    p_ids,
    p_msk,
    p_pos,
    response_ids,  # [N, R]
    cut_idx,  # [N]
    pad_id: int,
):
    """Build input context for partial regeneration.

    Layout: [pad ... | prompt (right-aligned) | reused_response_prefix ]
    """
    N, P = p_ids.shape
    R = response_ids.shape[1]
    k_vec = cut_idx.clamp(min=0, max=R)  # [N]
    max_k = int(k_vec.max().item()) if N > 0 else 0
    ctx_len = P + max_k

    # actual prompt lengths (right-aligned non-pad)
    Lp = p_msk.sum(dim=1)  # [N]
    start = ctx_len - (Lp + k_vec)  # [N] left-pad amount

    col = torch.arange(ctx_len, device=p_ids.device).unsqueeze(0)  # [1, ctx_len]
    start_ = start.unsqueeze(1)  # [N, 1]
    Lp_ = Lp.unsqueeze(1)
    k_ = k_vec.unsqueeze(1)

    # --- prompt segment ---
    prom_mask = (col >= start_) & (col < start_ + Lp_)
    src_prom_col = (P - Lp_) + (col - start_)
    src_prom_col = src_prom_col.clamp(0, P - 1)

    # --- prefix segment ---
    pref_mask = (col >= start_ + Lp_) & (col < start_ + Lp_ + k_)
    resp_cut = response_ids[:, :max_k]  # [N, max_k]
    src_pref_col = (col - (start_ + Lp_)).clamp(0, max_k - 1)

    # assemble ids
    ctx_ids = torch.full((N, ctx_len), pad_id, dtype=p_ids.dtype, device=p_ids.device)
    prom_vals = torch.gather(p_ids, 1, src_prom_col)
    ctx_ids = torch.where(prom_mask, prom_vals, ctx_ids)
    pref_vals = torch.gather(resp_cut, 1, src_pref_col)
    ctx_ids = torch.where(pref_mask, pref_vals, ctx_ids)

    # attention mask
    ctx_msk = (prom_mask | pref_mask).to(p_msk.dtype)

    # position ids
    ctx_pos = torch.zeros((N, ctx_len), dtype=p_pos.dtype, device=p_pos.device)
    prom_pos_vals = torch.gather(p_pos, 1, src_prom_col)
    ctx_pos = torch.where(prom_mask, prom_pos_vals, ctx_pos)

    last_pos = p_pos[:, -1].unsqueeze(1)  # [N, 1]
    if max_k > 0:
        ar = torch.arange(1, max_k + 1, device=p_pos.device).unsqueeze(0)
        pref_pos_table = last_pos + ar
        src_pref_col_clamped = src_pref_col.clamp(0, max_k - 1)
        pref_pos_vals = torch.gather(pref_pos_table, 1, src_pref_col_clamped)
        ctx_pos = torch.where(pref_mask, pref_pos_vals, ctx_pos)

    return ctx_ids, ctx_msk, ctx_pos, max_k


# ---------------------------------------------------------------------------
# Data alignment helper (unchanged)
# ---------------------------------------------------------------------------


def _hashable_key(x):
    """Convert a prompt to a hashable key (handles str / list / dict / numpy)."""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return tuple(_hashable_key(i) for i in x)
    if isinstance(x, dict):
        # 优先取常见文本字段，否则用有序 json 作为兜底
        for k in ("text", "content", "prompt", "input"):
            if k in x:
                return _hashable_key(x[k])
        import json
        return json.dumps(x, sort_keys=True, ensure_ascii=False)
    # numpy scalar / array
    if hasattr(x, "item"):
        return x.item()
    if hasattr(x, "tolist"):
        return _hashable_key(x.tolist())
    return str(x)


def align_prev_to_gen(
    *,
    prev_data: dict,
    gen_batch,
    tokenizer,
    n_repeat: int,
    log_prob_key: str = "log_probs",
):
    """Align tensors from a previous rollout file to the row-order of gen_batch."""
    ref_prompts_raw = gen_batch.non_tensor_batch["prompt"][::n_repeat]

    # Normalise both sides to hashable keys
    bucket = defaultdict(deque)
    for idx, txt in enumerate(prev_data["input"]):
        bucket[_hashable_key(txt)].append(idx)

    perm_rows = []
    match_ok = True
    for txt in ref_prompts_raw:
        key = _hashable_key(txt)
        if len(bucket.get(key, ())) < n_repeat:
            match_ok = False
            break
        for _ in range(n_repeat):
            perm_rows.append(bucket[key].popleft())

    # Fallback: if exact-match fails, use index-aligned order
    if not match_ok:
        import warnings
        n_ref = len(ref_prompts_raw)
        n_prev = len(prev_data["input"])
        # print debug info
        prev_sample = prev_data["input"][0] if n_prev > 0 else "N/A"
        ref_sample = ref_prompts_raw[0] if n_ref > 0 else "N/A"
        warnings.warn(
            f"[align_prev_to_gen] prompt text match failed (n_repeat={n_repeat}, "
            f"prev={n_prev}, ref={n_ref}). "
            f"Falling back to index-aligned order. "
            f"prev_sample={repr(prev_sample)[:200]}, "
            f"ref_sample={repr(ref_sample)[:200]}",
            stacklevel=2,
        )
        # index-aligned: assume same batch ordering, repeat each row n_repeat times
        perm_rows = []
        for i in range(min(n_ref, n_prev)):
            base = min(i, n_prev - 1)
            for _ in range(n_repeat):
                perm_rows.append(base)

    perm = torch.tensor(perm_rows, dtype=torch.long)
    return {
        log_prob_key: prev_data[log_prob_key].index_select(0, perm),
        "perm": perm,
    }