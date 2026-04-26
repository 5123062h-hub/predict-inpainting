"""
Text Correction Agent

LangChainを使用した自己修正ループエージェント。
画像生成と評価を繰り返し、目標スコアに達するまで改善を続ける。
"""

from typing import Optional, Callable
from dataclasses import dataclass, field
import logging

from .models import (
    AgentState,
    GenerationRequest,
    GenerationResult,
    EvaluationResult,
    LoopStatus
)
from .generator import BaseGenerator
from .evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TextCorrectionConfig:
    """エージェント設定"""
    max_iterations: int = 5
    target_score: float = 0.9
    early_stop_no_improvement: int = 3  # 改善がない場合の早期停止回数（N-1回連続改善なしで停止）
    verbose: bool = True


@dataclass
class CorrectionResult:
    """修正処理の最終結果"""
    final_image_path: str
    final_text: str
    final_score: float
    iterations_used: int
    status: LoopStatus
    evaluation_history: list[EvaluationResult] = field(default_factory=list)
    generation_history: list[GenerationResult] = field(default_factory=list)


class TextCorrectionAgent:
    """
    テキスト修正エージェント

    画像生成（AnyText）と評価（VLM）を組み合わせた自己修正ループを実行する。
    """

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BaseEvaluator,
        config: Optional[TextCorrectionConfig] = None,
        text_modifier: Optional[Callable[[str, EvaluationResult], str]] = None
    ):
        """
        Args:
            generator: 画像生成器
            evaluator: 画像評価器
            config: エージェント設定
            text_modifier: 評価結果に基づいてテキストを修正するコールバック関数
        """
        self.generator = generator
        self.evaluator = evaluator
        self.config = config or TextCorrectionConfig()
        self.text_modifier = text_modifier or self._default_text_modifier

    def _apply_structural_feedback(self, evaluation: EvaluationResult):
        """Evaluatorの構造的FBをgeneratorのパラメータに適用"""
        if evaluation.suggested_cfg_scale is not None:
            self.generator.cfg_scale = evaluation.suggested_cfg_scale
            logger.info(f"Evaluator adjusted cfg_scale to {evaluation.suggested_cfg_scale}")

        if evaluation.suggested_crop_margin is not None:
            self.generator._next_crop_margin = evaluation.suggested_crop_margin
            logger.info(f"Evaluator adjusted crop_margin to {evaluation.suggested_crop_margin}")

    def _default_text_modifier(
        self,
        current_text: str,
        evaluation: EvaluationResult
    ) -> str:
        """
        デフォルトのテキスト修正ロジック

        評価結果のsuggested_textがあればそれを使用、
        なければ元のテキストをそのまま返す。
        """
        if evaluation.suggested_text:
            logger.info(f"Using suggested text: {evaluation.suggested_text}")
            return evaluation.suggested_text
        return current_text

    def _should_continue(
        self,
        state: AgentState,
        current_score: float
    ) -> LoopStatus:
        """
        ループを継続すべきかどうかを判定

        Args:
            state: 現在のエージェント状態
            current_score: 現在の評価スコア

        Returns:
            LoopStatus: ループの状態
        """
        # 目標スコア達成
        if current_score >= self.config.target_score:
            return LoopStatus.SUCCESS

        # 最大試行回数到達
        if state.current_iteration >= self.config.max_iterations:
            return LoopStatus.MAX_ITERATIONS

        # 改善がない場合の早期停止チェック
        if len(state.evaluation_history) >= self.config.early_stop_no_improvement:
            recent_scores = [
                e.overall_score
                for e in state.evaluation_history[-self.config.early_stop_no_improvement:]
            ]
            if all(s <= recent_scores[0] for s in recent_scores[1:]):
                logger.warning("No improvement detected, early stopping")
                return LoopStatus.EARLY_STOP

        return LoopStatus.CONTINUE

    def _log_iteration(
        self,
        iteration: int,
        text: str,
        evaluation: EvaluationResult
    ) -> None:
        """イテレーション情報をログ出力"""
        if self.config.verbose:
            logger.info(f"\n{'='*50}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"Target Text: {text}")
            logger.info(f"Detected Text (OCR): {evaluation.detected_text}")
            logger.info(f"Overall Score: {evaluation.overall_score:.3f}")
            logger.info(f"  - Text Accuracy: {evaluation.text_accuracy_score:.3f}")
            logger.info(f"  - Background Harmony: {evaluation.background_harmony_score:.3f}")
            if evaluation.detected_issues:
                logger.info(f"Issues: {evaluation.detected_issues}")
            if evaluation.correction_suggestions:
                logger.info(f"Suggestions: {evaluation.correction_suggestions}")
            logger.info(f"{'='*50}\n")

    def run(
        self,
        original_image_path: str,
        mask_image_path: str,
        ground_truth_text: str,
        initial_text: Optional[str] = None,
        style_prompt: Optional[str] = None
    ) -> CorrectionResult:
        """
        テキスト修正ループを実行

        Args:
            original_image_path: 元画像のパス
            mask_image_path: マスク画像のパス
            ground_truth_text: 正解テキスト
            initial_text: 初期テキスト（Noneの場合はground_truth_textを使用）
            style_prompt: スタイル指示（オプション）

        Returns:
            CorrectionResult: 修正処理の結果
        """
        # 初期状態を設定
        current_text = initial_text or ground_truth_text
        current_style_prompt = style_prompt
        current_scene: Optional[str] = None   # 初回評価後に設定・固定
        current_style: Optional[str] = None   # 毎評価後に更新

        state = AgentState(
            current_iteration=0,
            max_iterations=self.config.max_iterations,
            target_score=self.config.target_score,
            ground_truth_text=ground_truth_text,
            current_text=current_text
        )

        logger.info("Starting text correction loop")
        logger.info(f"Target score: {self.config.target_score}")
        logger.info(f"Max iterations: {self.config.max_iterations}")
        logger.info(f"Ground truth text: {ground_truth_text}")

        while True:
            state.current_iteration += 1

            # Step 1: 画像を生成
            generation_request = GenerationRequest(
                text=current_text,
                mask_image_path=mask_image_path,
                original_image_path=original_image_path,
                style_prompt=current_style_prompt
            )

            logger.info(f"--- Iteration {state.current_iteration} ---")
            logger.info(f"  text         : {current_text}")
            logger.info(f"  scene        : {current_scene or '(not yet captured)'}")
            logger.info(f"  style        : {current_style or '(none)'}")
            logger.info(f"  style_prompt : {current_style_prompt or '(none)'}")
            generation_result = self.generator.generate(generation_request)

            if not generation_result.success:
                logger.error(f"Generation failed: {generation_result.error_message}")
                return CorrectionResult(
                    final_image_path="",
                    final_text=current_text,
                    final_score=0.0,
                    iterations_used=state.current_iteration,
                    status=LoopStatus.ERROR,
                    evaluation_history=state.evaluation_history,
                    generation_history=state.generation_history
                )

            state.generation_history.append(generation_result)

            # Step 2: 生成画像を評価（マスク領域クロップ+全体画像の2枚評価）
            logger.info("Evaluating generated image")
            evaluation = self.evaluator.evaluate(
                generated_image_path=generation_result.generated_image_path,
                ground_truth_text=ground_truth_text,
                original_image_path=original_image_path,
                mask_image_path=mask_image_path,
            )

            state.evaluation_history.append(evaluation)

            # ログ出力
            self._log_iteration(state.current_iteration, current_text, evaluation)

            # Step 3: ループ継続判定
            status = self._should_continue(state, evaluation.overall_score)

            if status != LoopStatus.CONTINUE:
                logger.info(f"Loop ended with status: {status.value}")

                # 早期停止の場合、ベストスコアの結果を返す
                if status == LoopStatus.EARLY_STOP and state.evaluation_history:
                    best_idx = max(
                        range(len(state.evaluation_history)),
                        key=lambda i: state.evaluation_history[i].overall_score
                    )
                    best_eval = state.evaluation_history[best_idx]
                    best_gen = state.generation_history[best_idx]
                    logger.info(f"Returning best result from iteration {best_idx + 1} (score: {best_eval.overall_score:.3f})")
                    return CorrectionResult(
                        final_image_path=best_gen.generated_image_path,
                        final_text=best_gen.used_text,
                        final_score=best_eval.overall_score,
                        iterations_used=state.current_iteration,
                        status=status,
                        evaluation_history=state.evaluation_history,
                        generation_history=state.generation_history
                    )

                return CorrectionResult(
                    final_image_path=generation_result.generated_image_path,
                    final_text=current_text,
                    final_score=evaluation.overall_score,
                    iterations_used=state.current_iteration,
                    status=status,
                    evaluation_history=state.evaluation_history,
                    generation_history=state.generation_history
                )

            # Step 4: 構造的フィードバックをgeneratorに適用
            self._apply_structural_feedback(evaluation)

            if hasattr(self.generator, 'adjust_parameters'):
                self.generator.adjust_parameters(evaluation.overall_score, state.current_iteration)

            # Step 5: シーン説明とスタイル指示を更新してstyle_promptを組み立て
            # シーン説明は初回のみ設定（背景は変わらないため固定）
            if current_scene is None and evaluation.suggested_prompt:
                current_scene = evaluation.suggested_prompt
                logger.info(f"Scene captured: {current_scene}")

            # スタイル修正は毎回更新（評価ごとに改善指示が変わる）
            if evaluation.text_style_instruction:
                current_style = evaluation.text_style_instruction
                logger.info(f"Style instruction updated: {current_style}")

            # style_prompt = scene + style_correction を組み立て
            parts = [p for p in [current_scene, current_style] if p]
            current_style_prompt = ', '.join(parts) if parts else style_prompt

            # Step 6: テキストを修正して次のイテレーションへ
            current_text = self.text_modifier(current_text, evaluation)
            state.current_text = current_text


class TextCorrectionAgentWithLLM(TextCorrectionAgent):
    """
    LLMを使用したテキスト修正機能を持つエージェント

    評価結果を基にLLMがテキスト修正案を生成する。
    """

    def __init__(
        self,
        generator: BaseGenerator,
        evaluator: BaseEvaluator,
        correction_llm=None,
        config: Optional[TextCorrectionConfig] = None
    ):
        """
        Args:
            generator: 画像生成器
            evaluator: 画像評価器
            correction_llm: テキスト修正用LLM（LangChain ChatModel）
            config: エージェント設定
        """
        super().__init__(generator, evaluator, config)
        self.correction_llm = correction_llm

        if correction_llm:
            self.text_modifier = self._llm_text_modifier

    def _llm_text_modifier(
        self,
        current_text: str,
        evaluation: EvaluationResult
    ) -> str:
        """
        LLMを使用してテキストを修正

        評価結果を基に、LLMがより適切なテキストを提案する。
        """
        if not self.correction_llm:
            return self._default_text_modifier(current_text, evaluation)

        # 評価からsuggested_textがあれば優先使用
        if evaluation.suggested_text:
            return evaluation.suggested_text

        # LLMに修正を依頼
        prompt = f"""以下の評価結果を基に、テキストを修正してください。

現在のテキスト: {current_text}

評価結果:
- 総合スコア: {evaluation.overall_score}
- 検出された問題: {evaluation.detected_issues}
- 修正提案: {evaluation.correction_suggestions}

修正後のテキストのみを出力してください（説明不要）:"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.correction_llm.invoke([HumanMessage(content=prompt)])
            modified_text = response.content.strip()
            logger.info(f"LLM suggested text: {modified_text}")
            return modified_text
        except Exception as e:
            logger.warning(f"LLM text modification failed: {e}")
            return current_text
