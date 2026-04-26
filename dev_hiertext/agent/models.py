"""
構造化出力用のPydanticモデル定義

VLMからの評価結果とエージェントの状態を管理するためのデータモデル。
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class EvaluationResult(BaseModel):
    """VLMによる評価結果の構造化出力モデル"""

    # 総合スコア (0.0 - 1.0)
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="総合評価スコア（0.0〜1.0）"
    )

    # 文字の正確性スコア (0.0 - 1.0)
    text_accuracy_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="生成されたテキストが正解テキストとどれだけ一致しているか"
    )

    # 背景との馴染み具合スコア (0.0 - 1.0)
    background_harmony_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="テキストが背景画像に自然に溶け込んでいるか"
    )

    # OCRで検出されたテキスト
    detected_text: Optional[str] = Field(
        None,
        description="画像からOCRで検出されたテキスト"
    )

    # 検出された問題点
    detected_issues: list[str] = Field(
        default_factory=list,
        description="検出された具体的な問題点のリスト"
    )

    # 修正提案
    correction_suggestions: list[str] = Field(
        default_factory=list,
        description="改善のための具体的な提案"
    )

    # 推奨する修正テキスト（文字が間違っている場合）
    suggested_text: Optional[str] = Field(
        None,
        description="修正すべきテキストがある場合の推奨テキスト"
    )

    # 構造的パラメータフィードバック
    suggested_mask_scale: Optional[float] = Field(
        None,
        ge=0.5,
        le=2.0,
        description="マスクサイズ倍率。文字詰まり→1.2、文字疎→0.8"
    )

    suggested_crop_margin: Optional[float] = Field(
        None,
        ge=0.2,
        le=3.0,
        description="クロップマージン倍率。背景広すぎ→0.5"
    )

    suggested_cfg_scale: Optional[float] = Field(
        None,
        ge=5.0,
        le=20.0,
        description="CFG scale。文字化け→15.0、不自然→7.5"
    )

    # シーンキャプション（背景・素材・照明のみ、テキスト内容・スタイル指示を含まない）
    # 必須フィールド: 常に画像から背景シーンを記述すること
    suggested_prompt: str = Field(
        ...,
        description=(
            "REQUIRED. Background scene description only — no text content, no style instructions. "
            "Example: 'a weathered wooden sign board mounted on a brick wall'"
        )
    )

    # テキスト描画スタイルの修正指示（評価ごとに更新される）
    # 問題がなければ null を返す
    text_style_instruction: Optional[str] = Field(
        None,
        description=(
            "Text rendering style instruction to fix observed problems. "
            "Example: 'neon-style glowing orange letters matching the city lights'. "
            "Set to null ONLY if text rendering is already good (score >= 0.9)."
        )
    )

    # 詳細な評価理由
    reasoning: str = Field(
        ...,
        description="評価の詳細な理由"
    )


class GenerationRequest(BaseModel):
    """画像生成リクエストのモデル"""

    # 生成するテキスト
    text: str = Field(..., description="画像に埋め込むテキスト")

    # マスク画像のパス
    mask_image_path: str = Field(..., description="マスク画像のファイルパス")

    # 元画像のパス
    original_image_path: str = Field(..., description="元画像のファイルパス")

    # 追加のスタイル指示（オプション）
    style_prompt: Optional[str] = Field(
        None,
        description="テキストのスタイルに関する追加指示"
    )


class GenerationResult(BaseModel):
    """画像生成結果のモデル"""

    # 生成された画像のパス
    generated_image_path: str = Field(..., description="生成された画像のファイルパス")

    # 生成に使用したテキスト
    used_text: str = Field(..., description="生成に使用したテキスト")

    # 生成成功フラグ
    success: bool = Field(True, description="生成が成功したかどうか")

    # エラーメッセージ（失敗時）
    error_message: Optional[str] = Field(None, description="エラーメッセージ")


class AgentState(BaseModel):
    """エージェントの状態を管理するモデル"""

    # 現在の試行回数
    current_iteration: int = Field(0, description="現在の試行回数")

    # 最大試行回数
    max_iterations: int = Field(5, description="最大試行回数")

    # 目標スコア閾値
    target_score: float = Field(0.9, description="目標とするスコア閾値")

    # 正解テキスト
    ground_truth_text: str = Field(..., description="正解テキスト")

    # 現在の生成テキスト
    current_text: str = Field(..., description="現在の生成テキスト")

    # 評価履歴
    evaluation_history: list[EvaluationResult] = Field(
        default_factory=list,
        description="過去の評価結果の履歴"
    )

    # 生成履歴
    generation_history: list[GenerationResult] = Field(
        default_factory=list,
        description="過去の生成結果の履歴"
    )

    # 完了フラグ
    is_completed: bool = Field(False, description="タスクが完了したかどうか")

    # 完了理由
    completion_reason: Optional[str] = Field(None, description="完了した理由")


class LoopStatus(str, Enum):
    """ループの状態を表す列挙型"""
    CONTINUE = "continue"  # ループ継続
    SUCCESS = "success"    # 目標達成で終了
    MAX_ITERATIONS = "max_iterations"  # 最大試行回数到達
    EARLY_STOP = "early_stop"  # 改善なしで早期停止
    ERROR = "error"        # エラーで終了
