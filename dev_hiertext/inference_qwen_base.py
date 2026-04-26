#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習前のベースQwen2.5-VL-7B-Instructモデルを使用したOCR推論スクリプト
LoRAなしの元のモデルで推論を実行
"""

import os
import re
import gzip
import json
import argparse
from typing import Dict, Optional

import torch
import editdistance
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# =====================
# アノテーション読み込み
# =====================


def load_hiertext_annotations(annotation_file: str) -> Dict:
    """
    HierTextアノテーションファイルを読み込み、image_id + para_idx でアクセスできる辞書を返す

    Args:
        annotation_file: アノテーションファイルパス (.jsonl.gz または .json)

    Returns:
        {(image_id, para_idx): {"text": "...", "vertices_list": [...], ...}, ...}
    """
    print(f'Loading HierText annotations from {annotation_file}...')

    # アノテーションファイルを読み込み
    if annotation_file.endswith('.gz'):
        with gzip.open(annotation_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    annotations_dict = {}
    annotations = data['annotations']

    for ann in annotations:
        image_id = ann['image_id']

        for para_idx, para in enumerate(ann['paragraphs']):
            if not para.get('legible', True):
                continue

            # 段落内のテキストを収集
            para_texts = []
            vertices_list = []

            for line in para['lines']:
                if not line.get('legible', True):
                    continue

                words = line.get('words', [])
                for word in words:
                    if not word.get('legible', True):
                        continue
                    text = word.get('text', '').strip()
                    if text:
                        para_texts.append(text)
                        vertices_list.append(word['vertices'])

            para_text = ' '.join(para_texts)

            # 辞書に格納
            annotations_dict[(image_id, para_idx)] = {
                'text': para_text,
                'vertices_list': vertices_list,
                'word_count': len(para_texts),
            }

    print(f'Loaded {len(annotations_dict)} paragraph annotations')
    return annotations_dict


def extract_image_info_from_path(image_path: str) -> Optional[tuple]:
    """
    画像パスから image_id と para_idx を抽出

    Args:
        image_path: 画像のパス (例: mask_002a72617c3b1335_para0.png or masked_002a72617c3b1335_para0.jpg)

    Returns:
        (image_id, para_idx) のタプル、抽出失敗時は None
    """
    # ファイル名を取得
    filename = os.path.basename(image_path)

    # mask_{image_id}_para{para_idx}.png の形式
    match = re.match(r'mask_([0-9a-f]+)_para(\d+)\.png', filename)
    if match:
        image_id = match.group(1)
        para_idx = int(match.group(2))
        return (image_id, para_idx)

    # masked_{image_id}_para{para_idx}.jpg の形式
    match = re.match(r'masked_([0-9a-f]+)_para(\d+)\.(jpg|png)', filename)
    if match:
        image_id = match.group(1)
        para_idx = int(match.group(2))
        return (image_id, para_idx)

    return None


def infer_mask_path_from_image(image_path: str) -> Optional[str]:
    """
    マスク画像のパスから対応するモノクロマスク画像のパスを推測

    Args:
        image_path: マスク画像のパス (例: masked_002a72617c3b1335_para0.jpg)

    Returns:
        モノクロマスク画像のパス、推測失敗時は None
    """
    image_info = extract_image_info_from_path(image_path)
    if not image_info:
        return None

    image_id, para_idx = image_info

    # 画像ディレクトリから対応するマスクディレクトリを推測
    image_dir = os.path.dirname(image_path)

    # Masked_Images_* -> Mask_Monochro_* のパターンマッチング
    # 例: Masked_Images_val -> Mask_Monochro_val
    #     Masked_Images_train -> Mask_Monochro_train
    #     Masked_Images_test -> Mask_Monochro_test
    mask_dir = re.sub(r'Masked_Images_(\w+)', r'Mask_Monochro_\1', image_dir)

    # マスクファイル名を生成
    mask_filename = f'mask_{image_id}_para{para_idx}.png'
    mask_path = os.path.join(mask_dir, mask_filename)

    if os.path.exists(mask_path):
        return mask_path

    return None


# =====================
# OCR評価指標
# =====================


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate (CER) を計算

    CER = (挿入 + 削除 + 置換) / 正解文字数
    低いほど良い（0が完璧）

    Args:
        reference: 正解テキスト
        hypothesis: 予測テキスト

    Returns:
        CER値 (0.0 ~ 1.0以上)
    """
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    distance = editdistance.eval(reference, hypothesis)
    return distance / len(reference)


def calculate_ned(reference: str, hypothesis: str) -> float:
    """
    Normalized Edit Distance (NED) を計算

    NED = Levenshtein距離 / max(len(ref), len(hyp))
    0-1の範囲、0が完璧

    Args:
        reference: 正解テキスト
        hypothesis: 予測テキスト

    Returns:
        NED値 (0.0 ~ 1.0)
    """
    max_len = max(len(reference), len(hypothesis), 1)
    distance = editdistance.eval(reference, hypothesis)
    return distance / max_len


def load_and_preprocess_image(image_path):
    """画像をロードし、アスペクト比を保持しつつ正方形にパディング"""
    try:
        image = Image.open(image_path).convert('RGB')
        target_size = 448
        w, h = image.size

        if w <= 0 or h <= 0:
            print(f'Warning: Invalid image dimensions ({w}, {h}) for {image_path}')
            raise ValueError(f'Invalid image dimensions: {w}x{h}')

        # アスペクト比を保持しつつリサイズ
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)

        if new_w <= 0 or new_h <= 0:
            print(f'Warning: Calculated size would be ({new_w}, {new_h}) for {image_path}, using fallback')
            new_w = new_h = target_size

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 正方形にパディング（白背景）
        padded_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded_image.paste(image, (paste_x, paste_y))

        return padded_image
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        raise


def load_and_preprocess_mask_image(mask_path):
    """マスク画像をロードし、前処理を適用"""
    try:
        mask_image = Image.open(mask_path).convert('L')
        target_size = 448
        w, h = mask_image.size

        if w <= 0 or h <= 0:
            print(f'Warning: Invalid mask dimensions ({w}, {h}) for {mask_path}')
            raise ValueError(f'Invalid mask dimensions: {w}x{h}')

        # アスペクト比を保持しつつリサイズ
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)

        if new_w <= 0 or new_h <= 0:
            print(f'Warning: Calculated mask size would be ({new_w}, {new_h}) for {mask_path}, using fallback')
            new_w = new_h = target_size

        mask_image = mask_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_mask = Image.new('L', (target_size, target_size), 255)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        padded_mask.paste(mask_image, (paste_x, paste_y))

        return padded_mask
    except Exception as e:
        print(f'Error loading mask {mask_path}: {e}')
        raise


def load_base_model_and_processor():
    """ベースQwen2.5-VL-7B-Instructモデルとプロセッサーを読み込み（LoRAなし）"""

    print('Loading base Qwen2.5-VL-7B-Instruct model (no LoRA)...')

    base_model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'

    # プロセッサーをロード
    processor = AutoProcessor.from_pretrained(base_model_id, use_fast=False, padding_side='left')
    print('Processor loaded successfully')

    # ベースモデルをロード（LoRAなし）
    print('Loading base model...')
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.eval()
    print('Base Qwen model loaded successfully (no fine-tuning)')

    return model, processor


def predict_ocr(model, processor, masked_image_path, mask_image_path):
    """OCR推論を実行"""

    # 画像を読み込み（学習時と同じ前処理）
    masked_image = load_and_preprocess_image(masked_image_path)
    mask_image = load_and_preprocess_mask_image(mask_image_path)

    # ベースモデル用プロンプト（より明示的な指示）
    prompt_text = (
        'You are given two images:\n'
        '1. A masked image where text regions are hidden\n'
        '2. A binary mask where white region is text and black region is background.\n'
        'Task: Output ONLY the exact text content in the white masked region. '
        'Do not explain, describe, or add any other information. '
        'Just output the text itself.\n'
    )

    # メッセージ構成
    content_items = [
        {'type': 'text', 'text': prompt_text},
        {'type': 'image', 'image': masked_image},
        {'type': 'image', 'image': mask_image},
    ]

    messages = [
        {
            'role': 'user',
            'content': content_items,
        }
    ]

    # チャットテンプレートを適用
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        print(f'Chat template error: {e}')
        prompt = '<|im_start|>user\nOCR task with masked image and mask.<|im_end|>\n<|im_start|>assistant\n'

    # 画像を抽出
    processor_images = []
    for item in content_items:
        if item['type'] == 'image':
            processor_images.append(item['image'])

    # トークン化
    inputs = processor(
        text=prompt,
        images=processor_images,
        return_tensors='pt',
        padding=False,
        truncation=False,
        do_resize=False,
    )

    # GPU利用可能なら移動
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # 推論実行
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    # 結果をデコード
    generated_text = processor.batch_decode(
        outputs[:, inputs['input_ids'].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    return generated_text.strip()


def main():
    parser = argparse.ArgumentParser(description='ベースQwenモデル（学習前）を使用したOCR推論')
    parser.add_argument('--image', '-i', required=True, help='マスク済み画像のパス')
    parser.add_argument('--mask', '-k', help='白黒マスク画像のパス（省略時は自動推測）')

    args = parser.parse_args()

    # 入力検証
    if not os.path.exists(args.image):
        print(f'Error: Image file not found: {args.image}')
        return

    # マスクパスの処理
    mask_path = args.mask
    if not mask_path:
        # マスクパスが指定されていない場合は自動推測
        mask_path = infer_mask_path_from_image(args.image)
        if not mask_path:
            print(f'Error: Could not infer mask path from image: {args.image}')
            print('Please specify mask path explicitly with --mask')
            return
        print(f'Inferred mask path: {mask_path}')

    if not os.path.exists(mask_path):
        print(f'Error: Mask file not found: {mask_path}')
        return

    # ベースモデルとプロセッサーを読み込み
    model, processor = load_base_model_and_processor()

    # マスクパスからアノテーションファイルを自動推測
    ground_truth = None
    image_info = extract_image_info_from_path(mask_path)

    if image_info:
        # マスクのディレクトリから推測（Mask_Monochro_val or Mask_Monochro_train or Mask_Monochro_test）
        mask_dir = os.path.dirname(mask_path)
        if 'val' in mask_dir.lower():
            annotation_file = '/home/user/dev/dev_hiertext/hiertext/gt/validation.jsonl.gz'
        elif 'train' in mask_dir.lower():
            annotation_file = '/home/user/dev/dev_hiertext/hiertext/gt/train.jsonl.gz'
        elif 'test' in mask_dir.lower():
            annotation_file = '/home/user/dev/dev_hiertext/hiertext/gt/test.jsonl'
        else:
            annotation_file = None

        if annotation_file and os.path.exists(annotation_file):
            try:
                annotations = load_hiertext_annotations(annotation_file)
                ground_truth_data = annotations.get(image_info)
                if ground_truth_data:
                    ground_truth = ground_truth_data['text']
            except Exception as e:
                print(f'Warning: Failed to load annotations: {e}')

    try:
        # OCR推論を実行
        result = predict_ocr(model, processor, args.image, mask_path)
        print('\n' + '=' * 50)
        print('OCR Inference Result (Base Model - No Fine-tuning)')
        print('=' * 50)
        print(f'Predicted Text: {result}')

        if ground_truth:
            print(f'Ground Truth:   {ground_truth}')
            print(f'Exact Match:    {"✓" if result == ground_truth else "✗"}')

            # 評価指標を計算
            cer = calculate_cer(ground_truth, result)
            ned = calculate_ned(ground_truth, result)
            print('\nMetrics:')
            print(f'  CER: {cer:.4f}')
            print(f'  NED: {ned:.4f}')
            print(f'  1-NED (Similarity): {1.0 - ned:.4f}')

        print('=' * 50 + '\n')

    except Exception as e:
        print(f'Error during inference: {e}')
        import traceback

        traceback.print_exc()


if __name__ == '__main__':
    main()
