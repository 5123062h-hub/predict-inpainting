# 環境管理ガイド

## 推奨環境構成

| コンポーネント | 推奨バージョン | 備考 |
|--------------|--------------|------|
| Python | 3.11.x | 3.12は一部ライブラリ非対応 |
| CUDA Toolkit | 12.4 | PyTorch 2.6対応 |
| PyTorch | 2.6.0+cu124 | CUDA 12.4ビルド |
| transformers | 4.45-4.57 | huggingface-hub <1.0 必須 |
| datasets | 3.0-3.6 | modelscope制約 |
| huggingface-hub | 0.24-0.36 | <1.0 必須 |

## GPU/CUDA/PyTorch 互換性表

| PyTorch | CUDA Toolkit | Driver (最小) | 備考 |
|---------|-------------|---------------|------|
| 2.6.x | 12.4 | 550.54.15+ | 推奨 |
| 2.5.x | 12.4 | 550.54.15+ | 安定 |
| 2.4.x | 12.1 | 530.30.02+ | - |
| 2.3.x | 12.1 | 530.30.02+ | - |

## パッケージ依存関係の制約

### modelscope の制約
```
datasets>=3.0.0,<=3.6.0
huggingface-hub<1.0
```

### langchain の制約
```
packaging>=23.2.0,<26.0.0
```

### diffusers の制約
```
huggingface-hub>=0.34.0,<1.0
```

## 環境セットアップ

### 新規セットアップ（推奨）

```bash
# 1. uv をインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. リポジトリをクローン
git clone <repository-url>
cd dev_hiertext

# 3. 環境をセットアップ
./scripts/reset_environment.sh

# 4. 仮想環境をアクティベート
source .venv/bin/activate
```

### 既存環境の修復

```bash
# クイックフィックス（軽微な問題）
./scripts/quick_fix.sh

# 完全リセット（深刻な問題）
./scripts/reset_environment.sh
```

## パッケージ追加ルール

### ✅ 正しい方法

```bash
# 1. パッケージを追加
uv add <package-name>

# 2. ロックファイルを更新
uv lock

# 3. 同期
uv sync

# 4. 変更をコミット
git add pyproject.toml uv.lock
git commit -m "Add <package-name>"
```

### ❌ 避けるべき方法

```bash
# 直接pipでインストール（ロックファイルと乖離）
pip install <package>

# バージョン指定なしでアップグレード
pip install --upgrade <package>
```

## GPUライブラリ更新ルール

### PyTorch アップグレード手順

1. **互換性確認**: CUDA Toolkitバージョンを確認
2. **クリーンインストール**:
   ```bash
   uv pip uninstall torch torchvision torchaudio
   uv pip install torch==X.X.X+cuXXX --index-url https://download.pytorch.org/whl/cuXXX
   ```
3. **依存パッケージ更新**:
   ```bash
   uv add transformers accelerate --upgrade
   ```
4. **検証**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

### xformers / flash-attn インストール

```bash
# xformers（PyTorchバージョンに合わせる）
uv pip install xformers --index-url https://download.pytorch.org/whl/cu124

# flash-attn（ビルドが必要）
MAX_JOBS=4 uv pip install flash-attn --no-build-isolation
```

## トラブルシューティング

### よくあるエラー

#### 1. `ImportError: cannot import name 'LargeList' from 'datasets'`
**原因**: datasets バージョンが古い
**解決**: `uv add "datasets>=3.0.0"`

#### 2. `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`
**原因**: huggingface_hub が 1.0+ にアップグレードされた
**解決**: `uv add "huggingface-hub>=0.24.0,<1.0.0"`

#### 3. `langchain-core requires packaging<26.0.0`
**原因**: packaging が新しすぎる
**解決**: `uv add "packaging>=23.2.0,<26.0.0"`

#### 4. CUDA out of memory
**解決**: バッチサイズを減らすか、gradient checkpointing を有効化

### 環境の健全性チェック

```bash
# 依存関係の衝突チェック
pip check

# インストール済みパッケージ一覧
pip list --format=freeze

# PyTorch + CUDA 検証
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

## ファイル構成

```
dev_hiertext/
├── pyproject.toml          # 依存関係定義（メイン）
├── uv.lock                  # ロックファイル（自動生成）
├── requirements.lock.txt    # pip freeze 出力（バックアップ）
├── .venv/                   # 仮想環境
├── scripts/
│   ├── reset_environment.sh # 完全リセット
│   └── quick_fix.sh         # クイックフィックス
└── docs/
    └── ENVIRONMENT.md       # このドキュメント
```
