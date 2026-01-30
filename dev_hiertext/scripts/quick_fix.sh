#!/bin/bash
# =============================================================================
# クイックフィックススクリプト
# 完全リセットせずに現在の依存関係衝突を修正
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."

log_info "クイックフィックスを開始..."

# 仮想環境をアクティベート
source .venv/bin/activate

# packaging のダウングレード（langchain-core互換）
log_info "packaging をダウングレード..."
pip install "packaging>=23.2.0,<26.0.0"

# accelerate を更新
log_info "accelerate を更新..."
pip install "accelerate>=1.0.0"

# peft を更新
log_info "peft を更新..."
pip install "peft>=0.12.0"

# 依存関係チェック
log_info "依存関係チェック..."
pip check || true

log_success "クイックフィックス完了！"
