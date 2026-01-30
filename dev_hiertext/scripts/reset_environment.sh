#!/bin/bash
# =============================================================================
# Python環境完全リセットスクリプト
#
# 使用方法:
#   chmod +x scripts/reset_environment.sh
#   ./scripts/reset_environment.sh
#
# 前提条件:
#   - uv がインストール済み (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - CUDA 12.4 がインストール済み
# =============================================================================

set -e

# 色付き出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

log_info "プロジェクトディレクトリ: $PROJECT_ROOT"

# =============================================================================
# Step 1: 環境確認
# =============================================================================
log_info "Step 1: 環境確認..."

# uv確認
if ! command -v uv &> /dev/null; then
    log_error "uv がインストールされていません"
    log_info "以下のコマンドでインストールしてください:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
log_success "uv: $(uv --version)"

# Python確認
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
log_success "Python: $PYTHON_VERSION"

# CUDA確認
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    log_success "CUDA: $CUDA_VERSION"
else
    log_warning "nvcc が見つかりません（CUDA Toolkitが必要な場合はインストールしてください）"
fi

# GPU確認
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    log_success "GPU: $GPU_INFO"
else
    log_warning "nvidia-smi が見つかりません"
fi

# =============================================================================
# Step 2: 既存環境のバックアップと削除
# =============================================================================
log_info "Step 2: 既存環境のバックアップと削除..."

# 既存の.venvをバックアップ
if [ -d ".venv" ]; then
    BACKUP_NAME=".venv.backup.$(date +%Y%m%d_%H%M%S)"
    log_warning "既存の .venv を $BACKUP_NAME にバックアップ"
    mv .venv "$BACKUP_NAME"
fi

# ロックファイルをバックアップ
if [ -f "uv.lock" ]; then
    cp uv.lock "uv.lock.backup.$(date +%Y%m%d_%H%M%S)"
    rm uv.lock
    log_info "uv.lock をバックアップして削除"
fi

if [ -f "poetry.lock" ]; then
    cp poetry.lock "poetry.lock.backup.$(date +%Y%m%d_%H%M%S)"
    log_info "poetry.lock をバックアップ"
fi

# =============================================================================
# Step 3: pyproject.toml を更新
# =============================================================================
log_info "Step 3: pyproject.toml を更新..."

if [ -f "pyproject.toml.new" ]; then
    cp pyproject.toml "pyproject.toml.backup.$(date +%Y%m%d_%H%M%S)"
    mv pyproject.toml.new pyproject.toml
    log_success "pyproject.toml を更新しました"
else
    log_warning "pyproject.toml.new が見つかりません。既存のpyproject.tomlを使用します"
fi

# =============================================================================
# Step 4: 新しい仮想環境を作成
# =============================================================================
log_info "Step 4: 新しい仮想環境を作成..."

uv venv --python 3.11 .venv
log_success "仮想環境を作成しました"

# =============================================================================
# Step 5: 依存関係をインストール
# =============================================================================
log_info "Step 5: 依存関係をインストール..."

# PyTorch (CUDA 12.4) を先にインストール
log_info "PyTorch + CUDA 12.4 をインストール..."
uv pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# その他の依存関係をインストール
log_info "その他の依存関係をインストール..."
uv sync

log_success "依存関係のインストールが完了しました"

# =============================================================================
# Step 6: インストール検証
# =============================================================================
log_info "Step 6: インストール検証..."

# PyTorch + CUDA検証
source .venv/bin/activate
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 主要パッケージの検証
python3 -c "
import transformers
import datasets
import diffusers
import modelscope
import langchain
print(f'transformers: {transformers.__version__}')
print(f'datasets: {datasets.__version__}')
print(f'diffusers: {diffusers.__version__}')
print(f'modelscope: {modelscope.__version__}')
print(f'langchain: {langchain.__version__}')
"

# 依存関係の衝突チェック
log_info "依存関係の衝突チェック..."
if pip check; then
    log_success "依存関係に問題はありません"
else
    log_warning "一部の依存関係に問題があります（上記を確認してください）"
fi

# =============================================================================
# Step 7: ロックファイルを生成
# =============================================================================
log_info "Step 7: ロックファイルを生成..."

# 現在のパッケージリストを保存
pip freeze > requirements.lock.txt
log_success "requirements.lock.txt を生成しました"

# =============================================================================
# 完了
# =============================================================================
echo ""
log_success "========================================"
log_success "環境リセットが完了しました！"
log_success "========================================"
echo ""
log_info "使用方法:"
echo "  source .venv/bin/activate"
echo ""
log_info "依存関係の追加:"
echo "  uv add <package-name>"
echo ""
log_info "ロックファイルから環境を再現:"
echo "  uv sync"
echo ""
