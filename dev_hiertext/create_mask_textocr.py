#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextOCRアノテーションから白黒マスク画像とマスクされた画像を生成
空間クラスタリング（行検出→段落検出）で疑似段落を作成し、
単語数9以下の疑似段落のみをマスク対象とする
"""

import os
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

MIN_MASK_RATIO = 0.0005  # 画像面積の0.05%未満はスキップ


def _split_band_by_x_gap(band_indices, word_boxes, heights):
    """Y-band内の単語をX方向の隙間で行に分割"""
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


def cluster_words_into_paragraphs(word_boxes):
    """
    word_boxes: list of (x1, y1, x2, y2)
    2ステップで疑似段落を生成:
      1. Y-bandソート → X-gap分割でword→line  (O(n log n)、チェーン問題なし)
      2. 縦の近さ＋横重複でline→paragraph
    Returns: list of list of indices
    """
    n = len(word_boxes)
    if n == 0:
        return []

    heights = [max(b[3] - b[1], 1) for b in word_boxes]
    y_centers = [(b[1] + b[3]) / 2 for b in word_boxes]

    # --- Step1: Y中心でソート → Y-bandに分割 → バンド内をX-gapで行に分割 ---
    order = sorted(range(n), key=lambda i: y_centers[i])
    lines = []
    current_band = [order[0]]

    for k in range(1, len(order)):
        prev_i = order[k - 1]
        curr_i = order[k]
        y_dist = abs(y_centers[curr_i] - y_centers[prev_i])
        # Y中心の差が両単語の高さの小さい方の50%以内なら同じバンド
        if y_dist <= 0.5 * min(heights[prev_i], heights[curr_i]):
            current_band.append(curr_i)
        else:
            lines.extend(_split_band_by_x_gap(current_band, word_boxes, heights))
            current_band = [curr_i]
    lines.extend(_split_band_by_x_gap(current_band, word_boxes, heights))

    # 行内をX座標でソートしてbboxを計算
    line_data = []
    for line in lines:
        line.sort(key=lambda i: word_boxes[i][0])
        x1 = min(word_boxes[i][0] for i in line)
        y1 = min(word_boxes[i][1] for i in line)
        x2 = max(word_boxes[i][2] for i in line)
        y2 = max(word_boxes[i][3] for i in line)
        line_data.append({'indices': line, 'bbox': (x1, y1, x2, y2), 'height': max(y2 - y1, 1)})

    line_data.sort(key=lambda l: (l['bbox'][1] + l['bbox'][3]) / 2)

    # --- Step2: 縦の近さ＋横重複でline→paragraph ---
    # min(h_prev, h_curr) で保守的に判定、横重複10%以上を要求
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


def create_mask_images():
    base_dir = '/home/user/dev/dev_hiertext/textocr'
    image_dir = f'{base_dir}/train_images/train_images'
    annotation_file = f'{base_dir}/TextOCR_0.1_train.json'
    mask_dir = f'{base_dir}/Mask_Monochro_train'
    masked_image_dir = f'{base_dir}/Masked_Images_train'

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_image_dir, exist_ok=True)

    print('Loading TextOCR annotations...')
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    imgs = data['imgs']
    anns = data['anns']
    img2anns = data['imgToAnns']

    print(f'Images: {len(imgs)}, Annotations: {len(anns)}')

    total_paragraphs = 0
    skipped_small = 0
    skipped_words = 0
    total_samples = 0
    metadata = {}

    for img_idx, (img_id, img_info) in enumerate(imgs.items(), 1):
        img_name = os.path.basename(img_info['file_name'])
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            continue

        ann_ids = img2anns.get(img_id, [])
        if not ann_ids:
            continue

        try:
            original_image = Image.open(img_path).convert('RGB')
            img_width, img_height = original_image.size

            # 判読可能な単語のみ収集
            words = []
            for aid in ann_ids:
                ann = anns.get(aid)
                if ann is None:
                    continue
                text = ann.get('utf8_string', '.')
                if text == '.':  # illegible
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
            para_groups = cluster_words_into_paragraphs(word_boxes)
            total_paragraphs += len(para_groups)

            for para_idx, word_indices in enumerate(para_groups):
                word_count = len(word_indices)
                if word_count == 0 or word_count > 9:
                    skipped_words += 1
                    continue

                para_words = [words[i] for i in word_indices]

                # 全頂点を収集して凸包
                all_points = []
                for w in para_words:
                    pts = w['points']  # [x1,y1,x2,y2,...] flat
                    for k in range(0, len(pts) - 1, 2):
                        all_points.append((pts[k], pts[k + 1]))
                    # points が空の場合は bbox の4角を使用
                    if not pts:
                        bx1, by1, bx2, by2 = w['bbox']
                        all_points.extend([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])

                all_points_arr = np.array(all_points, dtype=np.float32)

                mask_image = Image.new('L', (img_width, img_height), 0)
                draw_mask = ImageDraw.Draw(mask_image)

                if len(all_points_arr) >= 3:
                    try:
                        hull = ConvexHull(all_points_arr)
                        hull_points = [tuple(all_points_arr[i].astype(int)) for i in hull.vertices]
                        draw_mask.polygon(hull_points, fill=255)
                    except Exception:
                        for w in para_words:
                            pts = w['points']
                            if len(pts) >= 6:
                                poly = [(pts[k], pts[k + 1]) for k in range(0, len(pts) - 1, 2)]
                                draw_mask.polygon(poly, fill=255)
                else:
                    for w in para_words:
                        bx1, by1, bx2, by2 = w['bbox']
                        draw_mask.rectangle([bx1, by1, bx2, by2], fill=255)

                mask_image = mask_image.point(lambda x: 255 if x > 127 else 0)

                mask_arr = np.array(mask_image)
                mask_ratio = float((mask_arr > 0).sum()) / (img_width * img_height)

                if mask_ratio < MIN_MASK_RATIO:
                    skipped_small += 1
                    continue

                # マスク領域の背景色を計算して塗りつぶし
                img_arr = np.array(original_image)
                ys, xs = np.where(mask_arr > 0)
                if len(xs) > 0:
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    bbox_img = img_arr[y1:y2 + 1, x1:x2 + 1]
                    bbox_mask = mask_arr[y1:y2 + 1, x1:x2 + 1]
                    bg_pixels = bbox_img[bbox_mask == 0]
                    avg_color = tuple(int(c) for c in bg_pixels.mean(axis=0)) if len(bg_pixels) > 0 else (128, 128, 128)
                else:
                    avg_color = (128, 128, 128)

                masked_image = original_image.copy()
                draw_masked = ImageDraw.Draw(masked_image)
                if len(all_points_arr) >= 3:
                    try:
                        draw_masked.polygon(hull_points, fill=avg_color)
                    except Exception:
                        pass
                else:
                    for w in para_words:
                        bx1, by1, bx2, by2 = w['bbox']
                        draw_masked.rectangle([bx1, by1, bx2, by2], fill=avg_color)

                para_text = ' '.join(w['text'] for w in para_words)
                metadata[f'{img_id}_para{para_idx}'] = {
                    'mask_ratio': mask_ratio,
                    'word_count': word_count,
                    'text': para_text,
                }

                mask_image.save(os.path.join(mask_dir, f'mask_{img_id}_para{para_idx}.png'))
                masked_image.save(os.path.join(masked_image_dir, f'masked_{img_id}_para{para_idx}.jpg'))
                total_samples += 1

        except Exception as e:
            print(f'Error processing {img_id}: {e}')
            continue

        if img_idx % 500 == 0 or img_idx == len(imgs):
            print(f'[{img_idx}/{len(imgs)}] samples={total_samples}')

    print(f'\n{"=" * 60}')
    print(f'Total paragraphs found:    {total_paragraphs}')
    print(f'Skipped (>9 or 0 words):   {skipped_words}')
    print(f'Skipped (mask too small):  {skipped_small}')
    print(f'Total samples created:     {total_samples}')

    metadata_path = os.path.join(base_dir, 'mask_metadata_train.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f'Metadata saved to: {metadata_path}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    create_mask_images()
