#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierTextデータセット用 - マスク予測 + 座標補助
単語数9以下の行のみを使用
"""

import os
import gzip
import json
import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.spatial import ConvexHull
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

MIN_MASK_RATIO = 0.0005  # 画像面積の0.05%未満はスキップ


class Dataset(Dataset):
    """HierTextデータセット: マスク領域予測 + 座標情報補助"""

    def __init__(
        self,
        annotation_file,
        processor,
        max_samples=None,
        max_samples_per_image=None,
        mask_dir=None,
        masked_image_dir=None,
        augment=False,
    ):
        self.samples = []
        self.processor = processor
        self._image_cache = {}
        self.mask_dir = mask_dir
        self.masked_image_dir = masked_image_dir
        self.augment = augment
        if self.augment:
            self.color_jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)

        self._max_samples_per_image = max_samples_per_image

        print(f'Loading HierText dataset from {annotation_file}...')

        # アノテーションファイルを読み込み (gzip圧縮にも対応)
        if annotation_file.endswith('.gz'):
            with gzip.open(annotation_file, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

        annotations = data['annotations']

        # 各画像のアノテーションを処理
        for ann in tqdm(annotations, desc='Processing annotations'):
            image_id = ann['image_id']

            # まず、画像内の全段落のテキストを収集
            all_paragraphs = []
            for para_idx, para in enumerate(ann['paragraphs']):
                if not para.get('legible', True):
                    continue

                para_texts = []
                total_word_count = 0
                para_vertices_list = []

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
                            para_vertices_list.append(word['vertices'])
                            total_word_count += 1

                para_text = ' '.join(para_texts)
                all_paragraphs.append(
                    {
                        'para_idx': para_idx,
                        'text': para_text,
                        'word_count': total_word_count,
                        'vertices_list': para_vertices_list,
                    }
                )

            # 各段落を処理（単語数9以下のもののみ）
            for para_data in all_paragraphs:
                if para_data['word_count'] > 9 or para_data['word_count'] == 0:
                    continue

                # この段落以外のテキストを収集（周辺文字）
                surrounding_texts = [
                    p['text'] for p in all_paragraphs if p['para_idx'] != para_data['para_idx'] and p['text']
                ]

                # mask_ratio を凸包面積から近似計算
                mask_ratio = 0.0
                all_points = np.array(
                    [(v[0], v[1]) for verts in para_data['vertices_list'] for v in verts],
                    dtype=np.float32,
                )
                if len(all_points) >= 3:
                    try:
                        hull = ConvexHull(all_points)
                        mask_ratio = hull.volume / (ann.get('image_width', 1) * ann.get('image_height', 1))
                    except Exception:
                        pass

                if mask_ratio < MIN_MASK_RATIO:
                    continue

                sample_data = {
                    'image_id': image_id,
                    'para_idx': para_data['para_idx'],
                    'text': para_data['text'],
                    'word_count': para_data['word_count'],
                    'mask_ratio': mask_ratio,
                    'vertices_list': para_data['vertices_list'],
                    'surrounding_texts': surrounding_texts,
                }
                self.samples.append(sample_data)

        # 画像ごとにランダムで N サンプルを選択
        if self._max_samples_per_image:
            from collections import defaultdict
            by_image = defaultdict(list)
            for s in self.samples:
                by_image[s['image_id']].append(s)

            selected = []
            for samples in by_image.values():
                random.shuffle(samples)
                selected.extend(samples[:self._max_samples_per_image])
            self.samples = selected
            print(f'Filtered to top-{self._max_samples_per_image} per image: {len(self.samples)} samples')

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f'Loaded {len(self.samples)} HierText samples (paragraphs with ≤9 words)')

    def _load_and_preprocess_image(self, image_path):
        """画像をロード（リサイズはprocessorに委ねる）"""
        if image_path not in self._image_cache:
            image = Image.open(image_path).convert('RGB')
            self._image_cache[image_path] = image
        return self._image_cache[image_path]

    def save_augmented_samples(self, n=10, output_dir='./aug_samples'):
        """拡張前後の比較画像を保存（3パネル: 元画像 | 拡張後masked | 拡張後mask）"""
        if not self.augment:
            print('Warning: augment=False のため拡張は適用されません')
        os.makedirs(output_dir, exist_ok=True)
        jitter = self.color_jitter if self.augment else ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

        for i in range(min(n, len(self.samples))):
            sample = self.samples[i]
            image_id, para_idx = sample['image_id'], sample['para_idx']

            masked_path = os.path.join(self.masked_image_dir, f'masked_{image_id}_para{para_idx}.jpg')
            mask_path = os.path.join(self.mask_dir, f'mask_{image_id}_para{para_idx}.png')

            orig_masked = self._load_and_preprocess_image(masked_path)
            orig_mask = Image.open(mask_path).convert('RGB')

            # 拡張を手動適用
            angle = random.uniform(-5, 5)
            aug_masked = orig_masked.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
            aug_mask = orig_mask.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            aug_masked = jitter(aug_masked)
            if random.random() < 0.5:
                aug_masked = aug_masked.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

            # 可視化用に448pxにリサイズしてパネル作成
            viz_size = 448
            orig_masked_viz = orig_masked.resize((viz_size, viz_size), Image.Resampling.LANCZOS)
            aug_masked_viz = aug_masked.resize((viz_size, viz_size), Image.Resampling.LANCZOS)
            aug_mask_viz = aug_mask.resize((viz_size, viz_size), Image.Resampling.NEAREST)
            panel = Image.new('RGB', (viz_size * 3, viz_size), (200, 200, 200))
            panel.paste(orig_masked_viz, (0, 0))
            panel.paste(aug_masked_viz, (viz_size, 0))
            panel.paste(aug_mask_viz, (viz_size * 2, 0))

            label_safe = sample['text'][:20].replace('/', '_').replace(' ', '_')
            panel.save(os.path.join(output_dir, f'aug_sample_{i:04d}_{label_safe}.jpg'))

        print(f'Saved {min(n, len(self.samples))} augmented samples to {output_dir}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_data = self.samples[idx]

        prompt_text = (
            'You are given two images:\n'
            '1. A masked image where text regions are hidden\n'
            '2. A binary mask where white region is text and black region is background.\n'
            'Task: Predict the text in the white masked region.\n'
        )

        # マスクされた画像を使用
        image_id = sample_data['image_id']
        para_idx = sample_data['para_idx']

        # マスクされた画像のパスを構築（段落ごと）
        masked_image_path = os.path.join(self.masked_image_dir, f'masked_{image_id}_para{para_idx}.jpg')
        masked_image = self._load_and_preprocess_image(masked_image_path)

        # マスク画像のパスを構築（段落ごと）
        mask_path = os.path.join(self.mask_dir, f'mask_{image_id}_para{para_idx}.png')
        mask_image = Image.open(mask_path).convert('L')

        mask_image = mask_image.convert('RGB')

        # データ拡張（キャッシュ後・processor前に適用）
        if self.augment:
            # 小角度ランダム回転（±5°）: 両画像に同じ角度を適用
            angle = random.uniform(-5, 5)
            masked_image = masked_image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
            mask_image = mask_image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

            # ColorJitter: masked_image のみ（マスク画像の白黒は変えない）
            masked_image = self.color_jitter(masked_image)

            # GaussianBlur: 50%の確率でかける
            if random.random() < 0.5:
                radius = random.uniform(0.5, 1.5)
                masked_image = masked_image.filter(ImageFilter.GaussianBlur(radius=radius))

        processor_images = [masked_image, mask_image]

        content_items = [
            {'type': 'text', 'text': str(prompt_text)},
            {'type': 'image'},
            {'type': 'image'},
        ]

        target_text = sample_data['text']

        # messagesにassistant roleを含めて、processorに完全なフォーマットを任せる
        messages = [
            {
                'role': 'user',
                'content': content_items,
            },
            {'role': 'assistant', 'content': target_text},
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # processor_imagesは既に準備済み（PILイメージのリスト）
        enc = self.processor(
            text=prompt,
            images=processor_images,
            return_tensors='pt',
            padding=False,
        )

        # labelsの作成: assistant応答部分のみを学習対象とする
        input_ids = enc['input_ids'][0]

        # トークンIDを取得
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')

        # assistant応答開始位置をトークン空間で直接探す（テキストラウンドトリップによるずれを回避）
        marker_ids = self.processor.tokenizer.encode(
            '<|im_start|>assistant\n', add_special_tokens=False
        )
        input_ids_list = input_ids.tolist()
        assistant_start_pos = None
        for i in range(len(input_ids_list) - len(marker_ids), -1, -1):
            if input_ids_list[i:i + len(marker_ids)] == marker_ids:
                assistant_start_pos = i + len(marker_ids)
                break
        assert assistant_start_pos is not None, 'Error: <|im_start|>assistant not found in input_ids'

        # 最後の<|im_end|>の位置を探す
        end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        assistant_end_pos = end_positions[-1].item()

        # labelsを作成（assistant応答部分のみ学習）
        labels = torch.full_like(input_ids, -100)
        labels[assistant_start_pos : assistant_end_pos + 1] = input_ids[assistant_start_pos : assistant_end_pos + 1]

        enc['labels'] = labels.unsqueeze(0)

        # import pdb; pdb.set_trace()

        # batch dimを削除
        enc = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in enc.items()}

        # 周辺テキストと画像パスを追加
        enc['surrounding_texts'] = sample_data.get('surrounding_texts', [])
        enc['masked_image_path'] = masked_image_path

        return enc
