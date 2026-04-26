#!/bin/bash
# =============================================================================
# 環境セットアップスクリプト
# =============================================================================
# 2つの環境を管理:
#   .venv-agent              - AnyText用 (Python 3.10, PyTorch 2.0.1)
#   .venv-predict-inpainting - Qwen2-VL用 (Python 3.11, PyTorch 2.6)
# =============================================================================

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "環境セットアップ"
echo "============================================"

# Agent環境 (AnyText用)
if [ ! -d ".venv-agent" ]; then
    echo ""
    echo "=== .venv-agent (AnyText用) を作成 ==="
    uv venv --python 3.10 .venv-agent
    .venv-agent/bin/pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    uv pip install --python .venv-agent/bin/python -r agent/requirements.txt
else
    echo "✅ .venv-agent は既に存在します"
fi

# Predict-Inpainting環境 (Qwen2-VL用)
if [ ! -d ".venv-predict-inpainting" ]; then
    echo ""
    echo "=== .venv-predict-inpainting (Qwen2-VL用) を作成 ==="
    uv venv --python 3.11 .venv-predict-inpainting
    uv pip install --python .venv-predict-inpainting/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu124
    uv pip install --python .venv-predict-inpainting/bin/python "transformers>=4.45.0" peft accelerate bitsandbytes editdistance tensorboard tqdm pydantic datasets huggingface-hub pillow
else
    echo "✅ .venv-predict-inpainting は既に存在します"
fi

echo ""
echo "============================================"
echo "セットアップ完了"
echo "============================================"
echo ""
echo "使い方:"
echo ""
echo "  [AnyText Agent用]"
echo "    source .venv-agent/bin/activate"
echo "    python -m agent.run"
echo ""
echo "  [Qwen2-VL 学習/推論用]"
echo "    source .venv-predict-inpainting/bin/activate"
echo "    python train_qwen_hiertext.py"
echo "    python inference_qwen_hiertext.py"
