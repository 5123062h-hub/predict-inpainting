#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HierText val画像データを使って以下を生成:
1. 白黒マスク画像（どこを予測するか）
2. マスクされた画像（元画像のマスク領域を白で塗りつぶし）
単語数が9以下の段落（paragraph）のみをマスク対象とする
段落ごとに個別のマスク画像を生成
"""

import os
import gzip
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull


def create_mask_images():
    """HierTextアノテーションから白黒マスク画像とマスクされた画像を生成（段落ごとに個別ファイル、単語数9以下の段落のみ）"""

    # ディレクトリパス
    base_dir = '/home/user/dev/dev_hiertext/hiertext'
    image_dir = f'{base_dir}/train'
    annotation_file = f'{base_dir}/gt/train.jsonl.gz'
    mask_dir = f'{base_dir}/Mask_Monochro_train'
    masked_image_dir = f'{base_dir}/Masked_Images_train'

    # マスク画像保存ディレクトリを作成
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(masked_image_dir, exist_ok=True)

    # アノテーションファイルを読み込み
    print('Loading annotations...')

    if annotation_file.endswith('.gz'):
        with gzip.open(annotation_file, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    annotations = data['annotations']
    print(f'Processing {len(annotations)} images...')

    total_paragraphs = 0
    filtered_paragraphs = 0
    total_samples = 0
    metadata = {}  # {"{image_id}_para{idx}": {"mask_ratio": float, "word_count": int}}

    for i, ann in enumerate(annotations, 1):
        image_id = ann['image_id']
        img_name = f'{image_id}.jpg'
        img_path = os.path.join(image_dir, img_name)

        if not os.path.exists(img_path):
            print(f'Image not found: {img_path}')
            continue

        try:
            # 元画像を読み込んでサイズを取得
            original_image = Image.open(img_path).convert('RGB')
            img_width, img_height = original_image.size

            current_image_paragraphs = 0
            current_image_filtered_paragraphs = 0

            # 各段落を処理
            for para_idx, paragraph in enumerate(ann['paragraphs']):
                current_image_paragraphs += 1

                if not paragraph.get('legible', True):
                    continue

                # 段落内の全単語を収集
                para_word_count = 0
                para_vertices = []

                for line in paragraph['lines']:
                    if not line.get('legible', True):
                        continue

                    words = line.get('words', [])
                    for word in words:
                        if not word.get('legible', True):
                            continue

                        vertices = word['vertices']
                        if len(vertices) >= 3:  # 最低3点
                            para_word_count += 1
                            para_vertices.append(vertices)

                # 単語数が9以下の段落のみマスクを作成
                if para_word_count <= 9 and para_word_count > 0:
                    current_image_filtered_paragraphs += 1
                    total_samples += 1

                    # 段落専用のマスク画像を作成 (黒背景)
                    mask_image = Image.new('L', (img_width, img_height), 0)
                    draw_mask = ImageDraw.Draw(mask_image)

                    # 段落専用のマスクされた画像を作成（元画像のコピー）
                    masked_image = original_image.copy()
                    draw_masked = ImageDraw.Draw(masked_image)

                    # 段落内の全単語の頂点を集めて凸包を計算
                    all_points = np.array(
                        [(v[0], v[1]) for vertices in para_vertices for v in vertices],
                        dtype=np.float32,
                    )
                    if len(all_points) >= 3:
                        hull = ConvexHull(all_points)
                        hull_points = [tuple(all_points[i].astype(int)) for i in hull.vertices]
                        draw_mask.polygon(hull_points, fill=255)
                    else:
                        # 点が少なすぎる場合は従来通り個別ポリゴン
                        for vertices in para_vertices:
                            points = [(v[0], v[1]) for v in vertices]
                            draw_mask.polygon(points, fill=255)

                    # 2値化を確実にする（中間値が混入しないようにする）
                    mask_image = mask_image.point(lambda x: 255 if x > 127 else 0)

                    # マスクのバウンディングボックス内の背景ピクセルの平均色を計算
                    img_arr = np.array(original_image)
                    mask_arr = np.array(mask_image)
                    ys, xs = np.where(mask_arr > 0)
                    if len(xs) > 0:
                        x1, x2 = xs.min(), xs.max()
                        y1, y2 = ys.min(), ys.max()
                        bbox_img = img_arr[y1:y2+1, x1:x2+1]
                        bbox_mask = mask_arr[y1:y2+1, x1:x2+1]
                        bg_pixels = bbox_img[bbox_mask == 0]
                        if len(bg_pixels) > 0:
                            avg_color = tuple(int(c) for c in bg_pixels.mean(axis=0))
                        else:
                            avg_color = (128, 128, 128)
                    else:
                        avg_color = (128, 128, 128)

                    if len(all_points) >= 3:
                        draw_masked.polygon(hull_points, fill=avg_color)
                    else:
                        for vertices in para_vertices:
                            points = [(v[0], v[1]) for v in vertices]
                            draw_masked.polygon(points, fill=avg_color)

                    # mask_ratio を計算してメタデータに記録
                    mask_ratio = float((mask_arr > 0).sum()) / (img_width * img_height)
                    metadata[f'{image_id}_para{para_idx}'] = {
                        'mask_ratio': mask_ratio,
                        'word_count': para_word_count,
                    }

                    # 段落ごとにマスク画像を保存
                    mask_filename = f'mask_{image_id}_para{para_idx}.png'
                    mask_path = os.path.join(mask_dir, mask_filename)
                    mask_image.save(mask_path)

                    # 段落ごとにマスクされた画像を保存
                    masked_filename = f'masked_{image_id}_para{para_idx}.jpg'
                    masked_path = os.path.join(masked_image_dir, masked_filename)
                    masked_image.save(masked_path)

            total_paragraphs += current_image_paragraphs
            filtered_paragraphs += current_image_filtered_paragraphs

            if i % 100 == 0 or i == len(annotations):
                print(f'[{i}/{len(annotations)}] Processed {image_id}')
                print(
                    f'  - Paragraphs in image: {current_image_paragraphs}, Filtered (≤9 words): {current_image_filtered_paragraphs}'
                )

        except Exception as e:
            print(f'Error processing {image_id}: {e}')
            continue

    print(f'\n{"=" * 60}')
    print('Mask image generation completed!')
    print(f'{"=" * 60}')
    print(f'Total images processed: {len(annotations)}')
    print(f'Total paragraphs found: {total_paragraphs}')
    print(f'Paragraphs with ≤9 words (used for masking): {filtered_paragraphs}')
    print(f'Total samples created: {total_samples}')
    metadata_path = os.path.join(base_dir, 'mask_metadata_train.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f'Mask images saved to: {mask_dir}')
    print(f'Masked images saved to: {masked_image_dir}')
    print(f'Metadata saved to: {metadata_path}')
    print(f'{"=" * 60}')


def main():
    create_mask_images()


if __name__ == '__main__':
    main()
