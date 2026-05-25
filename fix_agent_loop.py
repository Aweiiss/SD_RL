#!/usr/bin/env python3
"""
Fix agent_loop.py _postprocess to handle speculative decoding's variable-length prompts.

Run this script in your conda environment:
    python3 fix_agent_loop.py
"""

import importlib.util
import sys

def find_agent_loop():
    """Find agent_loop.py in the Python path."""
    spec = importlib.util.find_spec('verl.experimental.agent_loop')
    if spec and spec.origin:
        return spec.origin
    
    # Fallback: search common locations
    import os
    candidates = [
        "/home/lingquh1xx/.local/envs/SD_RL_new/lib/python3.12/site-packages/verl/experimental/agent_loop/agent_loop.py",
        "/home/lingquh1xx/L2598/Temp/verl/verl/experimental/agent_loop/agent_loop.py",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def fix_file(filepath):
    """Apply the fix to agent_loop.py."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the problematic line and fix it
    old_code = 'prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)'
    new_code = '''# FIX for speculative decoding: truncate prompt_ids to max_prompt_length
        # to handle variable-length prompts from build_ctx's response prefix reuse
        _max_pl = 512  # max_prompt_length, should match data.max_prompt_length config
        prompt_ids = torch.cat([
            input.prompt_ids[:_max_pl] if len(input.prompt_ids) > _max_pl else input.prompt_ids
            for input in inputs
        ], dim=0)'''
    
    if old_code not in content:
        print(f"ERROR: Could not find the target line in {filepath}")
        print("The file may have a different version. Manual fix required.")
        return False
    
    if new_code in content:
        print(f"Fix already applied in {filepath}")
        return True
    
    content = content.replace(old_code, new_code)
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"Successfully fixed: {filepath}")
    return True

def main():
    filepath = find_agent_loop()
    if not filepath:
        print("ERROR: Could not find agent_loop.py")
        print("Please run with your conda environment activated:")
        print("    conda activate SD_RL_new")
        print("    python3 fix_agent_loop.py")
        sys.exit(1)
    
    print(f"Found agent_loop.py: {filepath}")
    
    if fix_file(filepath):
        print("\nNext steps:")
        print("1. Reinstall verl: cd /home/lingquh1xx/L2598/Temp/verl && pip install -e . --no-deps")
        print("2. Rerun experiment: bash 03_run_spec_adaptive.sh")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
