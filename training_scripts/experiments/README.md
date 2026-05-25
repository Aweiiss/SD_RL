# Spec-RL 三配置对比实验框架

## 实验设计

| 配置 | 名称 | 说明 |
|------|------|------|
| Config 1 | **Vanilla** | 标准 GRPO，无投机推理 |
| Config 2 | **Spec-Fixed** | 投机推理，使用固定投机步长（原始 spec_cut） |
| Config 3 | **Spec-Adaptive** | 投机推理，使用自适应窗口桶策略（你的改进） |

## 对比指标

| 指标 | 类型 | 说明 |
|------|------|------|
| `critic score` / val_reward | 质量 | 验证集奖励分数 |
| `gen_time` | 时间 | 单次 generation 耗时 |
| `throughput_tokens` | 吞吐 | 每秒生成的 token 数 |
| `throughput_samples` | 吞吐 | 每秒处理的样本数 |
| `step_time` | 时间 | 单次训练 step 总耗时 |
| `gen_time improvement` | 相对 | gen_time 降低百分比 |
| `throughput improvement` | 相对 | throughput 提升百分比 |
| `step_time improvement` | 相对 | step_time 降低百分比 |

## 目录结构

```
experiments/
├── README.md                      # 本文件
├── 01_run_vanilla.sh              # Config 1: 无投机
├── 02_run_spec_fixed.sh           # Config 2: 固定投机步长
├── 03_run_spec_adaptive.sh        # Config 3: 自适应窗口桶
├── experiment_manager.py          # 实验管理器（一键运行+分析）
└── analyze_results.py             # 结果分析+可视化
```

## 使用方式

### 方式一：一键运行（推荐）

```bash
cd experiments

# 编辑参数后运行
python experiment_manager.py \
    --project_dir /path/to/your/verl/project \
    --model_path /path/to/Qwen3-1.7B-base \
    --data_path /path/to/data \
    --num_gpu 2
```

这会依次运行三个实验，最后自动对比分析。

### 方式二：手动逐个运行

```bash
cd experiments

# Step 1: Vanilla baseline
bash 01_run_vanilla.sh 2>&1 | tee logs/vanilla_manual.log

# Step 2: Spec with fixed window
bash 02_run_spec_fixed.sh 2>&1 | tee logs/spec_fixed_manual.log

# Step 3: Spec with adaptive window
bash 03_run_spec_adaptive.sh 2>&1 | tee logs/spec_adaptive_manual.log

# Step 4: Analyze
python analyze_results.py \
    --project_dir /path/to/your/verl/project \
    --log_vanilla logs/vanilla_manual.log \
    --log_spec_fixed logs/spec_fixed_manual.log \
    --log_spec_adaptive logs/spec_adaptive_manual.log
```

### 方式三：仅运行单个配置

```bash
python experiment_manager.py \
    --project_dir /path/to/your/verl/project \
    --config vanilla  # 只跑 baseline
```

### 方式四：仅分析已有日志

```bash
python experiment_manager.py \
    --project_dir /path/to/your/verl/project \
    --skip_run
```

## 输出文件

运行后会在 `project_dir/analysis/` 目录生成：

```
analysis/
├── report.json              # 结构化对比数据
├── summary.txt              # 文本摘要报告
├── comparison_curves.png    # 指标随 step 变化曲线
├── improvement_bars.png     # 相对提升柱状图
└── adaptive_window.png      # 自适应窗口变化曲线（仅 Config 3）
```

## 关键参数说明

### 训练脚本中的关键差异

三个脚本唯一的不同是最后几行：

```bash
# Config 1 (Vanilla)
trainer.spec_decoding=False \
trainer.spec_bias=0.0 \

# Config 2 (Spec-Fixed)  
trainer.spec_decoding=True \
trainer.spec_bias=0.0 \

# Config 3 (Spec-Adaptive)
trainer.spec_decoding=True \
trainer.spec_bias=0.0 \
```

Config 2 和 Config 3 都使用 `spec_decoding=True`，但 **Spec-Adaptive** 会自动启用 `AdaptiveWindowBucket` 类进行窗口管理。

### 自适应窗口桶的参数（已内置，无需配置）

```python
# 窗口桶配置（R = max_response_length）
buckets = [R/16, R/8, R/4, R/2, 3*R/4, R]
# 例如 R=4096: [256, 512, 1024, 2048, 3072, 4096]
# 起始窗口: R/4 = 1024
```

## 修改摘要

### 对 `speculative_decoding.py` 的改动

1. **Bug 修复**：`math.log(0.0)` 崩溃、全局 RNG 污染、`argmax` 语义陷阱
2. **新增 `AdaptiveWindowBucket` 类**：实现你定义的 6 桶自适应窗口策略

### 对 `ray_trainer.py` 的改动

1. **修复投机解码数据流断裂**：`combined_gen_batch_for_gen` 正确传递给 `generate_sequences`
2. **删除遗留的 `breakpoint()`**
3. **修复 `spec_bias` 符号隐式反转**
4. **添加自适应窗口集成**：`sid -> AdaptiveWindowBucket` 映射
5. **添加 spec 相关监控指标**：`spec/window`, `spec/awb_action` 等

## 注意事项

1. **首次运行的前几个 step** 没有历史 rollout 数据，投机解码会自动退化到标准生成模式
2. **rollout_data_dir 必须配置**，否则无法保存/加载历史数据
3. 三个实验应使用 **相同的随机种子** 和 **相同的数据集** 以保证公平对比
4. 建议每个实验至少运行 **50 个 step** 以获得稳定的统计数据
