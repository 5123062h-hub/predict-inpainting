#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextOCRデータセット用 - マスク予測
空間クラスタリングで作成した疑似段落ベース
hiertext_dataset.py と同一インターフェース
"""

import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.spatial import ConvexHull
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

MIN_MASK_RATIO = 0.0005


def _split_band_by_x_gap(band_indices, word_boxes, heights):
    if not band_indices:
        return []
    band_sorted = sorted(band_indices, key=lambda i: (word_boxes[i][0] + word_boxes[i][2]) / 2)
    lines = [[band_sorted[0]]]
    for k in range(1, len(band_sorted)):
        prev_i = band_sorted[k - 1]
        curr_i = band_sorted[k]
        x_gap = max(0.0, word_boxes[curr_i][0] - word_boxes[prev_i][2])
        if x_gap < 3 * max(heights[prev_i], heights[curr_i]):
            lines[-1].append(curr_i)
        else:
            lines.append([curr_i])
    return lines


def _cluster_words_into_paragraphs(word_boxes):
    """
    word_boxes: list of (x1, y1, x2, y2)
    Returns: list of list of indices
    """
    n = len(word_boxes)
    if n == 0:
        return []

    heights = [max(b[3] - b[1], 1) for b in word_boxes]
    y_centers = [(b[1] + b[3]) / 2 for b in word_boxes]

    order = sorted(range(n), key=lambda i: y_centers[i])
    lines = []
    current_band = [order[0]]

    for k in range(1, len(order)):
        prev_i = order[k - 1]
        curr_i = order[k]
        y_dist = abs(y_centers[curr_i] - y_centers[prev_i])
        if y_dist <= 0.5 * min(heights[prev_i], heights[curr_i]):
            current_band.append(curr_i)
        else:
            lines.extend(_split_band_by_x_gap(current_band, word_boxes, heights))
            current_band = [curr_i]
    lines.extend(_split_band_by_x_gap(current_band, word_boxes, heights))

    line_data = []
    for line in lines:
        line.sort(key=lambda i: word_boxes[i][0])
        x1 = min(word_boxes[i][0] for i in line)
        y1 = min(word_boxes[i][1] for i in line)
        x2 = max(word_boxes[i][2] for i in line)
        y2 = max(word_boxes[i][3] for i in line)
        line_data.append({'indices': line, 'bbox': (x1, y1, x2, y2), 'height': max(y2 - y1, 1)})

    line_data.sort(key=lambda l: (l['bbox'][1] + l['bbox'][3]) / 2)

    paragraphs = [[line_data[0]]]
    for i in range(1, len(line_data)):
        prev = paragraphs[-1][-1]
        curr = line_data[i]
        local_h = min(prev['height'], curr['height'])
        v_gap = curr['bbox'][1] - prev['bbox'][3]
        h_overlap = min(prev['bbox'][2], curr['bbox'][2]) - max(prev['bbox'][0], curr['bbox'][0])
        min_w = min(prev['bbox'][2] - prev['bbox'][0], curr['bbox'][2] - curr['bbox'][0], 1)

        if v_gap <= 1.5 * local_h and h_overlap / min_w > 0.1:
            paragraphs[-1].append(curr)
        else:
            paragraphs.append([curr])

    return [[i for line in para for i in line['indices']] for para in paragraphs]


class TextOCRDataset(Dataset):
    """TextOCRデータセット: 疑似段落ベースのマスク領域予測"""

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

        print(f'Loading TextOCR dataset from {annotation_file}...')
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        imgs = data['imgs']
        anns = data['anns']
        img2anns = data['imgToAnns']

        for img_id, img_info in tqdm(imgs.items(), desc='Processing annotations'):
            ann_ids = img2anns.get(img_id, [])
            if not ann_ids:
                continue

            img_width = img_info.get('width', 1)
            img_height = img_info.get('height', 1)

            words = []
            for aid in ann_ids:
                ann = anns.get(aid)
                if ann is None:
                    continue
                text = ann.get('utf8_string', '.').strip()
                if not text or text == '.':
                    continue
                bx, by, bw, bh = ann['bbox']
                words.append({
                    'text': text,
                    'bbox': (bx, by, bx + bw, by + bh),
                    'points': ann.get('points', []),
                })

            if not words:
                continue

            word_boxes = [w['bbox'] for w in words]
            para_groups = _cluster_words_into_paragraphs(word_boxes)

            # 画像内の全疑似段落テキストを収集（surrounding_texts 用）
            all_para_texts = []
            for indices in para_groups:
                t = ' '.join(words[i]['text'] for i in indices)
                if t:
                    all_para_texts.append(t)

            for para_idx, word_indices in enumerate(para_groups):
                word_count = len(word_indices)
                if word_count == 0 or word_count > 9:
                    continue

                para_words = [words[i] for i in word_indices]
                para_text = ' '.join(w['text'] for w in para_words)

                # mask_ratio を凸包面積から近似
                all_points = []
                for w in para_words:
                    pts = w['points']
                    for k in range(0, len(pts) - 1, 2):
                        all_points.append((pts[k], pts[k + 1]))
                    if not pts:
                        bx1, by1, bx2, by2 = w['bbox']
                        all_points.extend([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])

                mask_ratio = 0.0
                if len(all_points) >= 3:
                    try:
                        hull = ConvexHull(np.array(all_points, dtype=np.float32))
                        mask_ratio = hull.volume / (img_width * img_height)
                    except Exception:
                        pass

                if mask_ratio < MIN_MASK_RATIO:
                    continue

                surrounding_texts = [t for t in all_para_texts if t != para_text]

                self.samples.append({
                    'image_id': img_id,
                    'para_idx': para_idx,
                    'text': para_text,
                    'word_count': word_count,
                    'mask_ratio': mask_ratio,
                    'surrounding_texts': surrounding_texts,
                })

        if max_samples_per_image:
            by_image = defaultdict(list)
            for s in self.samples:
                by_image[s['image_id']].append(s)
            selected = []
            for samples in by_image.values():
                random.shuffle(samples)
                selected.extend(samples[:max_samples_per_image])
            self.samples = selected
            print(f'Filtered to top-{max_samples_per_image} per image: {len(self.samples)} samples')

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f'Loaded {len(self.samples)} TextOCR samples (pseudo-paragraphs with ≤9 words)')

    def _load_image(self, path):
        if path not in self._image_cache:
            self._image_cache[path] = Image.open(path).convert('RGB')
        return self._image_cache[path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_id = sample['image_id']
        para_idx = sample['para_idx']

        masked_image_path = os.path.join(self.masked_image_dir, f'masked_{image_id}_para{para_idx}.jpg')
        mask_path = os.path.join(self.mask_dir, f'mask_{image_id}_para{para_idx}.png')

        masked_image = self._load_image(masked_image_path)
        mask_image = Image.open(mask_path).convert('RGB')

        if self.augment:
            angle = random.uniform(-5, 5)
            masked_image = masked_image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
            mask_image = mask_image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
            masked_image = self.color_jitter(masked_image)
            if random.random() < 0.5:
                masked_image = masked_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        prompt_text = (
            'You are given two images:\n'
            '1. A masked image where text regions are hidden\n'
            '2. A binary mask where white region is text and black region is background.\n'
            'Task: Predict the text in the white masked region.\n'
        )

        messages = [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': str(prompt_text)},
                {'type': 'image'},
                {'type': 'image'},
            ]},
            {'role': 'assistant', 'content': sample['text']},
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.processor(
            text=prompt,
            images=[masked_image, mask_image],
            return_tensors='pt',
            padding=False,
        )

        input_ids = enc['input_ids'][0]
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        marker_ids = self.processor.tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
        input_ids_list = input_ids.tolist()

        assistant_start_pos = None
        for i in range(len(input_ids_list) - len(marker_ids), -1, -1):
            if input_ids_list[i:i + len(marker_ids)] == marker_ids:
                assistant_start_pos = i + len(marker_ids)
                break
        assert assistant_start_pos is not None, '<|im_start|>assistant not found'

        end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        assistant_end_pos = end_positions[-1].item()

        labels = torch.full_like(input_ids, -100)
        labels[assistant_start_pos:assistant_end_pos + 1] = input_ids[assistant_start_pos:assistant_end_pos + 1]
        enc['labels'] = labels.unsqueeze(0)

        enc = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v for k, v in enc.items()}
        enc['surrounding_texts'] = sample.get('surrounding_texts', [])
        enc['masked_image_path'] = masked_image_path

        return enc
