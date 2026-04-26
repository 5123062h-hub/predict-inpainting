"""
Text Correction Agent Package

画像内テキスト修正のためのAIエージェントパッケージ。
LangChainを使用した自己修正ループを実装。
"""

from .models import (
    EvaluationResult,
    GenerationRequest,
    GenerationResult,
    AgentState,
    LoopStatus
)
from .generator import (
    BaseGenerator,
    AnyTextGenerator,
    LocalAnyTextGenerator
)
from .evaluator import (
    BaseEvaluator,
    QwenVLEvaluator,
    TransformersQwenEvaluator
)
from .agent import (
    TextCorrectionAgent,
    TextCorrectionAgentWithLLM,
    TextCorrectionConfig,
    CorrectionResult
)

__all__ = [
    # Models
    "EvaluationResult",
    "GenerationRequest",
    "GenerationResult",
    "AgentState",
    "LoopStatus",
    # Generators
    "BaseGenerator",
    "AnyTextGenerator",
    "LocalAnyTextGenerator",
    # Evaluators
    "BaseEvaluator",
    "QwenVLEvaluator",
    "TransformersQwenEvaluator",
    # Agent
    "TextCorrectionAgent",
    "TextCorrectionAgentWithLLM",
    "TextCorrectionConfig",
    "CorrectionResult",
]

__version__ = "0.1.0"
