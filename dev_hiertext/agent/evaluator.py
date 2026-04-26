"""
Evaluator モジュール

VLMを使用した画像評価のための抽象クラスと実装。
Qwen2.5-VL-7B-Instructや他のVLMに差し替え可能な設計。
"""

import re
import base64
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

from .models import EvaluationResult

logger = logging.getLogger(__name__)


# VLM評価用システムプロンプト
EVALUATION_SYSTEM_PROMPT = """You are a technical evaluator for AI text-in-image generation
using AnyText (an inpainting-based text rendering model).

Your job is to:
1. Score the generated image
2. Diagnose specific rendering problems
3. Write a complete AnyText-compatible prompt for the next generation attempt

---

## Scoring Criteria

### text_accuracy_score (0.0-1.0)
- Are all characters correctly formed and legible?
- Does the text exactly match the target string (case-sensitive)?
- Are characters overlapping, truncated, fused, or mirrored?

### background_harmony_score (0.0-1.0)
- Does the text blend naturally with the surrounding scene?
- Are there white rectangles, hard edges, color mismatches, or lighting inconsistencies?
- Does the text style (font, weight, texture) match the surface material?

---

## Parameter Diagnosis
Set fields to null when no change is needed.

| Observed problem | Suggested fix |
|---|---|
| Characters cramped / overlapping | suggested_mask_scale > 1.0 (e.g. 1.3) |
| Characters too small / sparse for area | suggested_mask_scale < 1.0 (e.g. 0.8) |
| Too much empty background in crop | suggested_crop_margin < 1.0 (e.g. 0.5) |
| Text too long for region | suggested_text = shortened version (preserve meaning) |
| Garbled / random characters | suggested_cfg_scale = 13.0–15.0 |
| Unnatural artifacts / white blobs | suggested_cfg_scale = 7.5–9.0 |

---

## Scene Caption — `suggested_prompt` (ALWAYS fill this)

Describe only the **background scene** where text appears.
Do NOT include the target text content, rendering style, or any instructions.

Rules:
- Name the physical surface (wooden board, metal sign, glass window, brick wall, chalkboard, etc.)
- Include dominant colors, materials, and lighting if visible
- Write as a natural BLIP-2-style image caption (noun phrase, present tense)
- Match exactly what you see — do not invent details

Good examples:
- `"a weathered wooden sign board mounted on a brick wall"`
- `"a vibrant nighttime city street with neon lights"`
- `"a dark chalkboard menu on a restaurant wall"`
- `"a green metal road sign post on a suburban street"`
- `"a glass shop window with warm indoor lighting"`

---

## Text Style Instruction — `text_style_instruction` (fill when text quality can improve)

Based on your evaluation, write a **concise style instruction** describing how the text should
be rendered to blend naturally with the scene and fix observed problems.

Rules:
- Describe letter appearance: color, glow, outline, weight, material finish
- Match the style to what would naturally appear in the scene
- Be specific (not just "better quality") — describe the visual result you want
- Set to `null` if the current rendering is already good (score >= 0.9)

Good examples:
- `"neon-style glowing orange tube letters matching the city lights"`
- `"hand-painted white serif letters with slight weathering"`
- `"bold white sans-serif with dark drop shadow for contrast on bright background"`
- `"chalk-white handwritten letters on dark surface"`
- `"gold vinyl lettering with slight reflection"`

Bad examples (too vague — avoid):
- `"better quality"`, `"clearer text"`, `"more natural"`

---

## Scoring Guide
- 0.95+: Production quality, no visible issues
- 0.8–0.95: Minor imperfections
- 0.6–0.8: Clearly visible problems
- <0.6: Major failure — unusable

---

## Output Format
Output ONLY valid JSON. No other text.

{format_instructions}
"""

EVALUATION_USER_PROMPT = """Evaluate the following images.

**Target text**: "{ground_truth_text}"

**Image 1 (text region crop)**: Where text was generated.
  → Evaluate text_accuracy_score: read each character carefully against the target.
  → Assess text rendering quality for text_style_instruction.

**Image 2 (full image)**: The complete composited result.
  → Evaluate background_harmony_score: check for blending artifacts, color mismatches.
  → Describe the background scene for suggested_prompt (scene caption only, no text content).

Output steps:
1. Read Image 1 character by character → detected_text
2. Score text_accuracy_score vs target
3. Score background_harmony_score from Image 2
4. List detected_issues and correction_suggestions
5. Describe the background scene from Image 2 → suggested_prompt (scene only, no style)
6. Write a style instruction to improve text rendering → text_style_instruction (or null if good)

Output your evaluation as JSON.
"""

EVALUATION_USER_PROMPT_SINGLE = """Evaluate the following image.

**Target text**: "{ground_truth_text}"

Output steps:
1. Read the text character by character → detected_text
2. Score text_accuracy_score vs target
3. Score background_harmony_score
4. List detected_issues and correction_suggestions
5. Describe the background scene → suggested_prompt (scene caption only, no text content)
6. Write a style instruction → text_style_instruction (or null if rendering is good)

Output your evaluation as JSON.
"""


class BaseEvaluator(ABC):
    """画像評価の抽象基底クラス"""

    @abstractmethod
    def evaluate(
        self,
        generated_image_path: str,
        ground_truth_text: str,
        original_image_path: Optional[str] = None,
        mask_image_path: Optional[str] = None,
    ) -> EvaluationResult:
        """
        生成画像を評価する

        Args:
            generated_image_path: 評価する生成画像のパス
            ground_truth_text: 正解テキスト
            original_image_path: 元画像のパス（比較用、オプション）
            mask_image_path: マスク画像のパス（テキスト領域クロップ用、オプション）

        Returns:
            EvaluationResult: 評価結果
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        評価器が利用可能かどうかを確認

        Returns:
            bool: 利用可能な場合True
        """
        pass


class QwenVLEvaluator(BaseEvaluator):
    """
    Qwen2.5-VL-7B-Instruct を使用した画像評価器

    LangChainを使用してVLMと連携し、構造化された評価結果を取得する。
    """

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        api_base: Optional[str] = None,
        device: str = 'cuda',
        use_vllm: bool = True,
        adjust_text_length: bool = False,
    ):
        """
        Args:
            model_name: モデル名
            api_base: vLLM等のAPIエンドポイント（使用する場合）
            device: 使用デバイス
            use_vllm: vLLMサーバーを使用するかどうか
            adjust_text_length: テキスト長調整を有効にするか（qwen_eval_server.py専用）
        """
        self.model_name = model_name
        self.api_base = api_base or 'http://localhost:8001/v1'
        self.device = device
        self.use_vllm = use_vllm
        self.adjust_text_length = adjust_text_length
        self.llm = None
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    def _initialize_llm(self):
        """LLMを初期化"""
        if self.use_vllm:
            # vLLMサーバー経由で使用
            from langchain_openai import ChatOpenAI

            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key='EMPTY',  # vLLMでは不要
                openai_api_base=self.api_base,
                max_tokens=2048,
                temperature=0.1,  # 評価の一貫性のため低めに設定
            )
        else:
            # ローカルで直接ロード（transformersベース）
            # 注: この実装は環境に合わせてカスタマイズが必要
            raise NotImplementedError(
                'Direct model loading is not implemented. Please use vLLM server or customize this implementation.'
            )

    def is_available(self) -> bool:
        """VLMが利用可能か確認"""
        try:
            if self.llm is None:
                self._initialize_llm()
            # 簡単なテストクエリを送信
            return True
        except Exception as e:
            logger.warning(f'VLM is not available: {e}')
            return False

    def _encode_image_to_base64(self, image_path: str) -> str:
        """画像をBase64エンコード"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _get_image_media_type(self, image_path: str) -> str:
        """画像のMIMEタイプを取得"""
        suffix = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
        }
        return media_types.get(suffix, 'image/png')

    def _crop_to_mask(self, image_path: str, mask_path: str, upscale_size: int = 768) -> Optional[str]:
        """
        マスク領域を generator と同じ正方形クロップ＋アップスケールで切り出し、
        Base64を返す。

        generator.py の square-crop ロジックと同一:
          margin = max(int(max(tw, th) * 1.0), 50)
          crop_side = max(tw, th) + margin * 2
          中心を基準に正方形クロップ → upscale_size にリサイズ

        Args:
            image_path: 生成画像パス（貼り戻し済み全体画像）
            mask_path: マスク画像パス
            upscale_size: アップスケール後のサイズ（デフォルト768）

        Returns:
            クロップ＋アップスケール画像のBase64文字列。失敗時はNone。
        """
        try:
            img = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                return None

            # マスクを画像サイズに合わせる
            h, w = img.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            all_pts = np.vstack(contours)
            tx, ty, tw, th = cv2.boundingRect(all_pts)

            # generator と同一の正方形クロップロジック
            margin = max(int(max(tw, th) * 1.0), 50)
            crop_side = max(tw, th) + margin * 2

            cx = tx + tw // 2
            cy = ty + th // 2
            crop_x1 = cx - crop_side // 2
            crop_y1 = cy - crop_side // 2
            crop_x2 = crop_x1 + crop_side
            crop_y2 = crop_y1 + crop_side

            # 画像範囲内に収まるようシフト
            if crop_x1 < 0:
                crop_x2 -= crop_x1
                crop_x1 = 0
            if crop_y1 < 0:
                crop_y2 -= crop_y1
                crop_y1 = 0
            if crop_x2 > w:
                crop_x1 -= crop_x2 - w
                crop_x2 = w
            if crop_y2 > h:
                crop_y1 -= crop_y2 - h
                crop_y2 = h

            # クランプ
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(w, crop_x2)
            crop_y2 = min(h, crop_y2)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

            # アップスケール（AnyText 入力と同じ解像度に揃える）
            crop_up = cv2.resize(crop, (upscale_size, upscale_size), interpolation=cv2.INTER_LANCZOS4)

            _, buf = cv2.imencode('.png', crop_up)
            return base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            logger.warning(f'Failed to crop image to mask: {e}')
            return None

    def _parse_response(self, response_text: str) -> EvaluationResult:
        """レスポンスをパースしてEvaluationResultに変換"""
        # JSONブロックを抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSONブロックがない場合、全体をJSONとして試行
            json_str = response_text.strip()

        result = self.parser.parse(json_str)
        logger.info(f'Evaluation - Score: {result.overall_score:.3f}, Detected: "{result.detected_text}"')
        logger.info(f'  suggested_prompt      : {result.suggested_prompt}')
        logger.info(f'  text_style_instruction: {result.text_style_instruction}')
        logger.debug(f'  raw VLM response:\n{response_text}')
        return result

    def evaluate(
        self,
        generated_image_path: str,
        ground_truth_text: str,
        original_image_path: Optional[str] = None,
        mask_image_path: Optional[str] = None,
    ) -> EvaluationResult:
        """
        VLMを使用して画像を評価

        Args:
            generated_image_path: 評価する生成画像のパス
            ground_truth_text: 正解テキスト
            original_image_path: 元画像のパス（オプション）
            mask_image_path: マスク画像のパス（指定時はクロップ+全体の2枚評価）

        Returns:
            EvaluationResult: 評価結果
        """
        del original_image_path  # 未使用（インターフェース互換性のため保持）

        if self.llm is None:
            self._initialize_llm()

        # システムプロンプトを構築
        system_prompt = EVALUATION_SYSTEM_PROMPT.format(format_instructions=self.parser.get_format_instructions())

        # 全体画像をBase64エンコード
        full_base64 = self._encode_image_to_base64(generated_image_path)
        media_type = self._get_image_media_type(generated_image_path)

        # マスクがあればクロップ画像も作成し2枚評価
        crop_base64 = None
        if mask_image_path:
            crop_base64 = self._crop_to_mask(generated_image_path, mask_image_path)

        if crop_base64:
            user_prompt = EVALUATION_USER_PROMPT.format(ground_truth_text=ground_truth_text)
            user_prompt += f'\nAdjust text length: {str(self.adjust_text_length).lower()}'
            content = [
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{crop_base64}'}},
                {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{full_base64}'}},
                {'type': 'text', 'text': user_prompt},
            ]
        else:
            user_prompt = EVALUATION_USER_PROMPT_SINGLE.format(ground_truth_text=ground_truth_text)
            user_prompt += f'\nAdjust text length: {str(self.adjust_text_length).lower()}'
            content = [
                {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{full_base64}'}},
                {'type': 'text', 'text': user_prompt},
            ]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ]

        response = self.llm.invoke(messages)
        response_text = response.content

        logger.debug(f'VLM Response: {response_text}')

        return self._parse_response(response_text)


class TransformersQwenEvaluator(BaseEvaluator):
    """
    Transformersを直接使用するQwen2.5-VL評価器

    vLLMサーバーなしで直接モデルをロードして使用する。
    """

    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2.5-VL-7B-Instruct',
        device: str = 'cuda',
        torch_dtype: str = 'bfloat16',
        max_new_tokens: int = 2048,
    ):
        """
        Args:
            model_name: Qwen2.5-VLモデル名
            device: 使用デバイス
            torch_dtype: PyTorchのデータ型
            max_new_tokens: 生成する最大トークン数
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self.parser = PydanticOutputParser(pydantic_object=EvaluationResult)

    def _load_model(self):
        """モデルをロード"""
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        dtype = getattr(torch, self.torch_dtype)

        logger.info(f'Loading Qwen2.5-VL model: {self.model_name}')

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map='auto',
            attn_implementation='flash_attention_2',  # 高速化（オプション）
        )

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        logger.info('Qwen2.5-VL model loaded successfully')

    def is_available(self) -> bool:
        """モデルが利用可能か確認"""
        try:
            if self.model is None:
                self._load_model()
            return self.model is not None
        except Exception as e:
            logger.warning(f'Qwen2.5-VL is not available: {e}')
            return False

    def _build_messages(self, image_path: str, ground_truth_text: str) -> list:
        """Qwen2.5-VL用のメッセージを構築"""
        system_prompt = EVALUATION_SYSTEM_PROMPT.format(format_instructions=self.parser.get_format_instructions())

        user_prompt = EVALUATION_USER_PROMPT.format(ground_truth_text=ground_truth_text)

        messages = [
            {'role': 'system', 'content': system_prompt},
            {
                'role': 'user',
                'content': [{'type': 'image', 'image': f'file://{image_path}'}, {'type': 'text', 'text': user_prompt}],
            },
        ]

        return messages

    def _parse_response(self, response_text: str) -> EvaluationResult:
        """レスポンスをパースしてEvaluationResultに変換"""
        # JSONブロックを抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSONブロックがない場合、JSON部分を探す
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text.strip()

        return self.parser.parse(json_str)

    def evaluate(
        self,
        generated_image_path: str,
        ground_truth_text: str,
        original_image_path: Optional[str] = None,
        mask_image_path: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Qwen2.5-VLを使用して画像を評価

        Args:
            generated_image_path: 評価する生成画像のパス
            ground_truth_text: 正解テキスト
            original_image_path: 元画像のパス（オプション）
            mask_image_path: マスク画像のパス（未使用、互換性のため）

        Returns:
            EvaluationResult: 評価結果
        """
        del original_image_path, mask_image_path  # 未使用（インターフェース互換性のため保持）

        from PIL import Image

        try:
            # モデルが未ロードならロード
            if self.model is None:
                self._load_model()

            # メッセージを構築
            messages = self._build_messages(generated_image_path, ground_truth_text)

            # テキスト入力を準備
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 画像を読み込み
            image = Image.open(generated_image_path)

            # 入力を準備
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors='pt')
            inputs = inputs.to(self.model.device)

            # 生成
            logger.info('Running Qwen2.5-VL evaluation...')
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # 評価の一貫性のため
            )

            # 入力部分を除いて出力をデコード
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

            response_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            logger.debug(f'VLM Response: {response_text}')

            # レスポンスをパース
            return self._parse_response(response_text)

        except Exception as e:
            logger.error(f'Qwen2.5-VL evaluation failed: {e}')
            import traceback

            traceback.print_exc()
            return EvaluationResult(
                overall_score=0.0,
                text_accuracy_score=0.0,
                background_harmony_score=0.0,
                suggested_prompt='unknown scene',
                detected_issues=[f'VLM evaluation error: {str(e)}'],
                correction_suggestions=[],
                reasoning=f'Error during evaluation: {str(e)}',
            )
