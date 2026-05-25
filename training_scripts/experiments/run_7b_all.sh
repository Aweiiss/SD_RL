#!/bin/bash
# =============================================================================
# 7B模型三种配置批量运行脚本
# 用法: bash run_7b_all.sh [开始配置编号]
#   bash run_7b_all.sh 5    # 从vanilla开始（默认）
#   bash run_7b_all.sh 6    # 从spec-fixed开始
# =============================================================================
set -euo pipefail

START_CONFIG="${1:-5}"
PROJECT_DIR="/home/lingquh1xx/L2598/Temp/verl"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="/home/lingquh1xx/L2598/Temp/verl/model/Qwen2.5-7B-Instruct"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  7B模型三种配置对比实验${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查模型是否存在
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo -e "${RED}[ERROR] 7B模型未找到: ${MODEL_PATH}${NC}"
    echo "请先下载模型:"
    echo "  cd ${PROJECT_DIR}"
    echo "  python download_7b_model.py --output_dir ${MODEL_PATH}"
    exit 1
fi

echo -e "${GREEN}[OK] 模型已就绪: ${MODEL_PATH}${NC}"

# 检查代码修改是否应用
echo ""
echo -e "${YELLOW}[INFO] 检查代码修改...${NC}"
if grep -q "AdaptiveWindowBucket" "${PROJECT_DIR}/verl/utils/speculative_decoding.py" 2>/dev/null; then
    echo -e "${GREEN}[OK] speculative_decoding.py 已修改${NC}"
else
    echo -e "${RED}[WARN] speculative_decoding.py 未修改，正在应用...${NC}"
    cp "${SCRIPT_DIR}/../verl/utils/speculative_decoding.py" \
       "${PROJECT_DIR}/verl/utils/speculative_decoding.py"
fi

if grep -q "combined_gen_batch_for_gen" "${PROJECT_DIR}/verl/trainer/ppo/ray_trainer.py" 2>/dev/null; then
    echo -e "${GREEN}[OK] ray_trainer.py 已修改${NC}"
else
    echo -e "${RED}[WARN] ray_trainer.py 未修改，正在应用...${NC}"
    cp "${SCRIPT_DIR}/../verl/trainer/ppo/ray_trainer.py" \
       "${PROJECT_DIR}/verl/trainer/ppo/ray_trainer.py"
fi

# 创建日志目录
mkdir -p "${PROJECT_DIR}/logs"

# 运行配置
CONFIGS=(
    "05:7B_Vanilla (无投机解码):${SCRIPT_DIR}/05_run_7b_vanilla.sh"
    "06:7B_Spec-Fixed (固定窗口投机):${SCRIPT_DIR}/06_run_7b_spec_fixed.sh"
    "07:7B_Spec-Adaptive (自适应窗口投机):${SCRIPT_DIR}/07_run_7b_spec_adaptive.sh"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=':' read -r cfg_id cfg_name cfg_script <<< "$cfg"

    # 如果指定了起始配置，跳过之前的
    if [ "${cfg_id}" -lt "${START_CONFIG}" ]; then
        echo -e "${YELLOW}[SKIP] 跳过 ${cfg_name}${NC}"
        continue
    fi

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  开始运行: ${cfg_name}${NC}"
    echo -e "${GREEN}========================================${NC}"

    # 检查显存
    echo -e "${YELLOW}[INFO] 当前GPU状态:${NC}"
    nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi不可用"

    # 运行实验
    set +e
    bash "${cfg_script}"
    exit_code=$?
    set -e

    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}[ERROR] ${cfg_name} 失败 (exit code: ${exit_code})${NC}"
        echo -e "${YELLOW}是否继续下一个配置? (y/n)${NC}"
        read -r continue_next
        if [ "$continue_next" != "y" ]; then
            echo "已中止"
            exit 1
        fi
    else
        echo -e "${GREEN}[SUCCESS] ${cfg_name} 完成${NC}"
    fi

    # 清理显存缓存
    echo -e "${YELLOW}[INFO] 清理GPU缓存...${NC}"
    python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  所有7B实验完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}分析结果:${NC}"
echo "  python experiments/analyze_results.py --log_dir ${PROJECT_DIR}/logs --output ${PROJECT_DIR}/results_7b_comparison.md"
