#!/usr/bin/env python3
"""
Qwen2.5-VL 評価サーバー (FastAPI)

.venv-predict-inpainting環境で起動し、OpenAI互換APIを提供する。
.venv-agent環境のエージェントからHTTPでアクセス可能。

使用方法:
    source .venv-predict-inpainting/bin/activate
    pip install fastapi uvicorn
    python qwen_eval_server.py --port 8000
"""

import io
import re
import json
import time
import uuid
import base64
import logging
import argparse
from typing import Any
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# グローバル変数でモデルを保持
model = None
processor = None
model_name_global = None


def load_model(model_name: str, device: str = 'cuda', torch_dtype: str = 'bfloat16'):
    """Qwen2.5-VLモデルをロード"""
    global model, processor, model_name_global

    from transformers import AutoProcessor, AutoModelForVision2Seq

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info(f'Loading model: {model_name}')
    processor = AutoProcessor.from_pretrained(model_name)
    # AutoModelForVision2Seq を使用して正しいモデルクラスを自動選択
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map='auto',
    )
    model_name_global = model_name
    logger.info('Model loaded successfully')


def calculate_edit_distance_score(detected: str, ground_truth: str) -> float:
    """編集距離に基づくスコアを計算 (0.0-1.0)"""
    import editdistance

    # 正規化: 大文字小文字を無視、空白を正規化
    detected_norm = ' '.join(detected.upper().split())
    gt_norm = ' '.join(ground_truth.upper().split())

    if not gt_norm:
        return 0.0 if detected_norm else 1.0

    distance = editdistance.eval(detected_norm, gt_norm)
    max_len = max(len(detected_norm), len(gt_norm))

    if max_len == 0:
        return 1.0

    score = 1.0 - (distance / max_len)
    return max(0.0, score)


def read_text_from_image(image) -> str:
    """VLMを使って画像からテキストを読み取る (OCR)"""
    global model, processor

    ocr_prompt = (
        'Read the text that is physically rendered and visible in this image.\n'
        'Transcribe ONLY the exact characters you can actually see — do NOT guess, infer, '
        'auto-correct, or complete words based on context or prior knowledge.\n'
        'If a character looks ambiguous or partially rendered, write what you literally see, '
        'even if it appears to be a typo or incomplete word.\n'
        'Do NOT output what the text "should" say — output only what is visually present.\n'
        'Output the transcribed text only, no explanation.\n'
        'If no text is visible or legible, output: [UNREADABLE]'
    )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': ocr_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    detected_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return detected_text.strip()


def evaluate_background_harmony(image, detected_text: str, ground_truth_text: str) -> tuple[float, str]:
    """背景との調和を評価"""
    global model, processor

    harmony_prompt = (
        'Evaluate the quality of the text rendering in this image.\n\n'
        'Criteria:\n'
        '1. Color harmony between the text and background\n'
        '2. Legibility of the text\n'
        '3. Naturalness of text edges and boundaries\n'
        '4. Overall visual coherence (no obvious artifacts or mismatches)\n\n'
        'Rate on a scale of 0.0 to 1.0 and output in this exact format:\n'
        'Score: [number]\n'
        'Reason: [brief explanation]'
    )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': harmony_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
        )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Extract score
    score_match = re.search(r'Score[：:]\s*([\d.]+)', output, re.IGNORECASE)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.5
    else:
        # Fallback: find any float/int in the output
        num_match = re.search(r'(0\.\d+|1\.0|0|1)', output)
        score = float(num_match.group(1)) if num_match else 0.5

    return score, output


def describe_background_scene(image) -> str:
    """VLMを使って背景シーンを記述する（suggested_prompt生成用）"""
    global model, processor

    scene_prompt = (
        'Describe the background scene where the text appears in this image.\n'
        'Focus only on the physical surface, materials, colors, and lighting.\n'
        'Do NOT mention the text content itself.\n'
        'Write a brief noun phrase such as:\n'
        "  'a weathered wooden sign board mounted on a brick wall'\n"
        "  'a green metal road sign post on a suburban street'\n"
        "  'a glass shop window with warm indoor lighting'\n"
        'Output only the scene description, nothing else.'
    )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': scene_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    scene_description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return scene_description.strip() or 'an outdoor scene'


def generate_text_style_instruction(image) -> str:
    """VLMを使ってテキストスタイル指示を生成する（text_style_instruction生成用）"""
    global model, processor

    style_prompt = (
        'Look at the text rendered in this image.\n'
        'Based on what you actually see — the background material, lighting, and current text appearance —\n'
        'describe how the text should be styled to blend more naturally with the scene.\n'
        'Be specific about: letter color, outline, glow, weight, material finish, shadow.\n'
        # "\n"
        # "The following are format examples only. Do NOT copy them — write your own based on this image:\n"
        # "  (e.g. 'bold engraved gold letters with a dark shadow on bronze metal')\n"
        # "  (e.g. 'hand-painted white serif letters with slight weathering on wood')\n"
        # "  (e.g. 'neon-style glowing orange tube letters matching the city lights')\n"
        # "\n"
        'Output only your style instruction for this image as a single sentence, nothing else.'
    )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': style_prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    style_instruction = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return style_instruction.strip() or None


def generate_suggested_text(image, ground_truth_text: str, detected_text: str) -> str | None:
    """
    テキストが長すぎて崩れている場合に、短縮版を提案する。
    問題なければ None を返す。
    """
    global model, processor

    prompt = (
        f'The target text was: "{ground_truth_text}"\n'
        f'The text actually visible in the image appears to be: "{detected_text}"\n\n'
        'Look carefully at the text rendered in the image.\n'
        'If the text looks cramped, overflowing, cut off, garbled, or illegible '
        'because there is too much text for the available space, '
        'suggest a shorter version of the target text that preserves the core meaning, '
        'is shorter than the original target text, '
        'and would realistically fit in the same space.\n'
        'If the text looks acceptable (not cramped, not cut off), output exactly: NONE\n'
        'Output only the suggested shorter text or NONE, nothing else.'
    )

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors='pt',
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.1,
            do_sample=False,
        )

    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    result = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if not result or result.upper() == 'NONE':
        return None
    return result


def evaluate_image(
    image_base64: str,
    ground_truth_text: str,
    adjust_text_length: bool = False,
    crop_image_base64: str | None = None,
) -> dict:
    """画像を評価してJSON結果を返す（2段階評価）"""
    global model, processor

    from PIL import Image

    # Base64デコード - full image (背景調和・シーン記述用)
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # クロップ画像があればOCRに使用、なければfull imageで代用
    if crop_image_base64:
        crop_data = base64.b64decode(crop_image_base64)
        ocr_image = Image.open(io.BytesIO(crop_data)).convert('RGB')
        logger.info('Step 1: Reading text from mask crop image...')
    else:
        ocr_image = image
        logger.info('Step 1: Reading text from full image...')

    detected_text = read_text_from_image(ocr_image)
    logger.info(f'Detected text: "{detected_text}"')
    logger.info(f'Ground truth: "{ground_truth_text}"')

    # Step 2: 編集距離でテキスト精度を計算
    text_accuracy_score = calculate_edit_distance_score(detected_text, ground_truth_text)
    logger.info(f'Text accuracy score (edit distance): {text_accuracy_score:.3f}')

    # Step 3: 背景との調和を評価
    logger.info('Step 3: Evaluating background harmony...')
    background_score, harmony_reasoning = evaluate_background_harmony(image, detected_text, ground_truth_text)
    logger.info(f'Background harmony score: {background_score:.3f}')

    # Step 4: 背景シーンを記述（suggested_prompt）
    logger.info('Step 4: Describing background scene...')
    suggested_prompt = describe_background_scene(image)
    logger.info(f'Background scene: "{suggested_prompt}"')

    # Step 5: テキストスタイル指示を生成（text_style_instruction）
    logger.info('Step 5: Generating text style instruction...')
    text_style_instruction = generate_text_style_instruction(image)
    logger.info(f'Text style instruction: "{text_style_instruction}"')

    # Step 6: テキスト長調整（adjust_text_length が有効な場合のみ）
    suggested_text = None
    if adjust_text_length:
        logger.info('Step 6: Checking if text length adjustment is needed...')
        suggested_text = generate_suggested_text(image, ground_truth_text, detected_text)
        logger.info(f'Suggested text: "{suggested_text}"')

    # 総合スコア
    overall_score = 0.5 * text_accuracy_score + 0.5 * background_score

    # 問題点の検出
    detected_issues = []
    if text_accuracy_score < 0.5:
        detected_issues.append(
            f'Text significantly differs: detected="{detected_text}", ground_truth="{ground_truth_text}"'
        )
    elif text_accuracy_score < 0.9:
        detected_issues.append(f'Minor text discrepancy: detected="{detected_text}"')

    if background_score < 0.5:
        detected_issues.append('Poor background harmony')

    if '[UNREADABLE]' in detected_text:
        detected_issues.append('Text is not legible')
        text_accuracy_score = 0.0
        overall_score = 0.0

    result = {
        'overall_score': round(overall_score, 3),
        'text_accuracy_score': round(text_accuracy_score, 3),
        'background_harmony_score': round(background_score, 3),
        'detected_text': detected_text,
        'detected_issues': detected_issues,
        'correction_suggestions': [f'Render the target text "{ground_truth_text}" more clearly']
        if text_accuracy_score < 0.9
        else [],
        'suggested_prompt': suggested_prompt,
        'text_style_instruction': text_style_instruction,
        'suggested_text': suggested_text,
        'reasoning': (
            f'Detected: "{detected_text}" / edit-distance score: {text_accuracy_score:.3f}'
            f' / background harmony: {harmony_reasoning[:200]}'
        ),
    }

    logger.info(f'Final evaluation: {result}')
    return result


# Pydanticモデル
class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str = 'qwen2.5-vl-7b-instruct'
    messages: list[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.1


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = 'chat.completion'
    created: int
    model: str
    choices: list[ChatCompletionChoice]


# FastAPIアプリ
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown


app = FastAPI(title='Qwen2.5-VL Evaluation Server', lifespan=lifespan)


@app.get('/health')
async def health():
    """ヘルスチェック"""
    return {'status': 'ok', 'model_loaded': model is not None}


class EvaluateRequest(BaseModel):
    image_base64: str
    ground_truth_text: str


@app.post('/evaluate')
async def evaluate_endpoint(request: EvaluateRequest):
    """直接評価エンドポイント（OCR + 編集距離ベース）"""
    global model, processor

    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    try:
        result = evaluate_image(request.image_base64, request.ground_truth_text)
        return result
    except Exception as e:
        logger.error(f'Evaluation error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/v1/chat/completions')
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """OpenAI互換のchat completions API"""
    global model, processor

    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')

    try:
        messages = request.messages

        # 画像とテキストを抽出
        images: list[str] = []  # base64文字列のリスト（送信順: crop→full）
        ground_truth_text = ''
        adjust_text_length = False

        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'image_url':
                            url = item.get('image_url', {}).get('url', '')
                            if url.startswith('data:'):
                                # data:image/png;base64,xxxxx の形式
                                images.append(url.split(',', 1)[1] if ',' in url else url)
                        elif item.get('type') == 'text':
                            text = item.get('text', '')
                            # 正解テキストを抽出（英語 "Target text" / 日本語 "正解テキスト" の両方に対応）
                            match = re.search(
                                r'\*?\*?(?:Target text|正解テキスト)\*?\*?[：:]\s*"?(.+?)"?(?:\n|$)',
                                text,
                                re.IGNORECASE,
                            )
                            if match:
                                ground_truth_text = match.group(1).strip()
                            # テキスト長調整フラグを抽出
                            flag_match = re.search(r'Adjust text length:\s*(true|false)', text, re.IGNORECASE)
                            if flag_match:
                                adjust_text_length = flag_match.group(1).lower() == 'true'

        if not images:
            raise HTTPException(status_code=400, detail='No image provided')

        # 2枚送られた場合: 1枚目=マスククロップ(OCR用), 2枚目=full image(調和評価用)
        # 1枚の場合: full imageのみ
        full_image_base64 = images[-1]
        crop_image_base64 = images[0] if len(images) >= 2 else None
        if crop_image_base64:
            logger.info(f'Received {len(images)} images: using crop for OCR, full for harmony')

        # 評価実行
        result = evaluate_image(
            full_image_base64,
            ground_truth_text,
            adjust_text_length=adjust_text_length,
            crop_image_base64=crop_image_base64,
        )

        # OpenAI形式でレスポンス
        return ChatCompletionResponse(
            id=f'chatcmpl-{uuid.uuid4().hex[:8]}',
            created=int(time.time()),
            model=model_name_global or 'qwen2.5-vl-7b-instruct',
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role='assistant',
                        content=json.dumps(result, ensure_ascii=False),
                    ),
                    finish_reason='stop',
                )
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error: {e}')
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL評価サーバー')
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct', help='モデル名')
    parser.add_argument('--port', type=int, default=8000, help='ポート番号')
    parser.add_argument('--host', default='0.0.0.0', help='ホスト')
    parser.add_argument('--device', default='cuda', help='デバイス')
    parser.add_argument('--dtype', default='bfloat16', help='データ型')
    args = parser.parse_args()

    # モデルロード
    load_model(args.model, args.device, args.dtype)

    # サーバー起動
    logger.info(f'Starting server on {args.host}:{args.port}')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
