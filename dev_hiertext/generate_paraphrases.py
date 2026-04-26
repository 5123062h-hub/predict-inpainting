#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierTextデータセットのパラフレーズキャッシュ事前生成スクリプト
学習前に一度だけ実行する前処理ステップ。

Usage:
    python generate_paraphrases.py --output ./paraphrase_cache_train.json
    python generate_paraphrases.py --max_samples 100  # デバッグ用
"""

import os
import json
import argparse

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText

from hiertext_dataset import Dataset


def generate_paraphrase_cache(samples, cache_file, n_paraphrases=3):
    """Qwen基底モデルを使ってパラフレーズを一括生成し、JSONキャッシュに保存する。
    .doneファイルが存在する場合は完了済みとしてスキップ。
    途中のキャッシュファイルが存在する場合は続きから再開する。"""
    done_file = cache_file + '.done'
    if os.path.exists(done_file):
        print(f'Paraphrase cache already complete ({cache_file}), skipping generation.')
        return

    # 途中から再開：既存のキャッシュをロード
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, encoding='utf-8') as f:
            cache = json.load(f)
        print(f'Resuming from existing cache: {len(cache)} entries already processed.')
    else:
        print(f'Generating paraphrase cache for {len(samples)} samples...')

    model_id = 'Qwen/Qwen2.5-VL-7B-Instruct'
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    gen_processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    gen_model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map='auto', low_cpu_mem_usage=True
    )
    gen_model.eval()

    new_count = 0
    for i, s in enumerate(samples):
        key = f'{s["image_id"]}_para{s["para_idx"]}'

        # 処理済みキーはスキップ
        if key in cache:
            continue

        ctx = ' | '.join(s['surrounding_texts'][:3]) if s['surrounding_texts'] else '(none)'
        word_count = len(s['text'].split())
        prompt = (
            f'Generate {n_paraphrases} short alternative phrasings of the text "{s["text"]}"'
            f' that could plausibly appear in a scene where the surrounding text is: {ctx}.'
            f' Keep each under {word_count + 2} words.'
            f' Output only the variations, one per line, with no numbering or extra text.'
        )
        messages = [{'role': 'user', 'content': prompt}]
        try:
            text = gen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inp = gen_processor(text=text, return_tensors='pt').to(gen_model.device)
            with torch.inference_mode():
                out = gen_model.generate(
                    **inp,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=gen_processor.tokenizer.eos_token_id,
                )
            generated = gen_processor.tokenizer.decode(
                out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True
            )
            lines = [line.strip().lstrip('0123456789.-) ') for line in generated.strip().split('\n') if line.strip()]
            cache[key] = [line for line in lines if line][:n_paraphrases]
        except Exception as e:
            print(f'  Warning: paraphrase generation failed for {key}: {e}')
            cache[key] = []

        new_count += 1

        # 100件ごとに中間保存（中断しても続きから再開できる）
        if new_count % 100 == 0:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            print(f'  Progress: {i + 1}/{len(samples)} samples ({len(cache)} total entries saved).')

    del gen_model, gen_processor
    torch.cuda.empty_cache()

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    with open(done_file, 'w') as f:
        f.write('done')
    print(f'Paraphrase cache saved to {cache_file} ({len(cache)} entries).')


def main():
    parser = argparse.ArgumentParser(description='HierText パラフレーズキャッシュ生成スクリプト')
    parser.add_argument(
        '--annotation_file',
        type=str,
        default='./hiertext/gt/train.jsonl.gz',
        help='HierText アノテーションファイルのパス',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./paraphrase_cache_train.json',
        help='出力キャッシュJSONファイルのパス',
    )
    parser.add_argument(
        '--n_paraphrases',
        type=int,
        default=3,
        help='1サンプルあたりのパラフレーズ生成数',
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='処理するサンプル数の上限（デバッグ用）',
    )
    args = parser.parse_args()

    # アノテーションのみロード（画像不要）
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct', use_fast=False)
    print('Loading HierText dataset...')
    dataset = Dataset(
        annotation_file=args.annotation_file,
        processor=processor,
        max_samples=args.max_samples,
        mask_dir=None,
        masked_image_dir=None,
        augment=False,
    )

    generate_paraphrase_cache(dataset.samples, args.output, n_paraphrases=args.n_paraphrases)


if __name__ == '__main__':
    main()
