#!/usr/bin/env python3
"""
エージェント実行スクリプト

使用方法:
    # マスク付き画像を直接指定（既存の動作）
    python run.py --image Masked_Images_test/masked_xxx_para0.jpg

    # 元画像パスから自動解決（{id}_para{n} 形式）
    python run.py --image hiertext/test/xxx_para0.jpg

    # 元画像のみ指定（最初のparaを自動選択）
    python run.py --image hiertext/test/xxx.jpg

    # テキストを明示的に指定
    python run.py --image masked_xxx_para0.jpg --text "Target Text"

    # モノクロマスクパスも明示的に指定
    python run.py --image image.png --mask mask.png --text "Target Text"
"""

import os
import re
import sys
import json
import logging
import argparse
from typing import Dict, Optional
from pathlib import Path

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import (
    LoopStatus,
    QwenVLEvaluator,
    TextCorrectionAgent,
    TextCorrectionConfig,
    LocalAnyTextGenerator,
    TransformersQwenEvaluator,
)


def setup_logging(verbose: bool = False):
    """ロギング設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def extract_image_info_from_path(image_path: str) -> Optional[tuple]:
    """
    画像パスから image_id と para_idx を抽出

    Args:
        image_path: 画像のパス (例: mask_002a72617c3b1335_para0.png or masked_002a72617c3b1335_para0.jpg)

    Returns:
        (image_id, para_idx) のタプル、抽出失敗時は None
    """
    filename = os.path.basename(image_path)

    # mask_{image_id}_para{para_idx}.png の形式
    match = re.match(r'mask_([0-9a-f]+)_para(\d+)\.png', filename)
    if match:
        return (match.group(1), int(match.group(2)))

    # masked_{image_id}_para{para_idx}.jpg の形式
    match = re.match(r'masked_([0-9a-f]+)_para(\d+)\.(jpg|png)', filename)
    if match:
        return (match.group(1), int(match.group(2)))

    # {image_id}_para{para_idx}.{ext} の形式（プレフィックスなし）
    match = re.match(r'([0-9a-f]+)_para(\d+)\.(jpg|png)', filename)
    if match:
        return (match.group(1), int(match.group(2)))

    return None


def extract_image_id_from_path(image_path: str) -> Optional[str]:
    """
    画像パスから image_id のみ抽出（para_idx なし）

    Args:
        image_path: 元画像のパス (例: 0ecebae5c2677549.jpg)

    Returns:
        image_id 文字列、抽出失敗時は None
    """
    filename = os.path.basename(image_path)

    # {image_id}.{ext} の形式（para指定なし）
    match = re.match(r'([0-9a-f]{16})\.(jpg|png)$', filename)
    if match:
        return match.group(1)

    return None


def infer_mask_path_from_image(image_path: str) -> Optional[str]:
    """
    画像パスから対応するモノクロマスク画像のパスを推測

    Args:
        image_path: マスク画像のパス (例: masked_002a72617c3b1335_para0.jpg)

    Returns:
        モノクロマスク画像のパス、推測失敗時は None

    パス変換例:
        Masked_Images_val/masked_xxx_para0.jpg -> Mask_Monochro_val/mask_xxx_para0.png
        Masked_Images_train/masked_xxx_para0.jpg -> Mask_Monochro_train/mask_xxx_para0.png
    """
    image_info = extract_image_info_from_path(image_path)
    if image_info is None:
        return None

    image_id, para_idx = image_info
    p_image = Path(image_path)
    image_dir_name = p_image.parent.name

    # Masked_Images_* -> Mask_Monochro_* のディレクトリ名変換
    mask_dir_name = image_dir_name.replace('Masked_Images_', 'Mask_Monochro_')
    mask_dir = p_image.parent.parent / mask_dir_name

    mask_path = mask_dir / f'mask_{image_id}_para{para_idx}.png'

    if mask_path.exists():
        return mask_path

    return None


# HierTextベースディレクトリ（デフォルト）
DEFAULT_HIERTEXT_BASE = '/home/user/dev/dev_hiertext/hiertext'

# スプリットリスト
SPLITS = ['test', 'train', 'val']


def resolve_image_paths(image_path: str, hiertext_base: str = None) -> Optional[Dict]:
    """
    任意の画像パスから、元画像とモノクロマスクのパスを解決する

    AnyTextはインペインティングモデルなので、元画像 + モノクロマスクがあれば動作する。
    masked画像（テキスト領域を白塗りしたもの）は不要。

    対応パターン:
        1. masked_{id}_para{n}.jpg - マスク付き画像 → 元画像+マスクに解決
        2. {id}_para{n}.jpg - para指定付きパス → 元画像+マスクに解決
        3. {id}.jpg - 元画像のみ → 最初のpara_idxのマスクを自動選択
        4. mask_{id}_para{n}.png - モノクロマスク → 元画像+マスクに解決

    Args:
        image_path: 入力画像パス
        hiertext_base: HierTextデータセットのベースディレクトリ

    Returns:
        {'image': 元画像パス, 'mask': モノクロマスクパス,
         'image_id': str, 'para_idx': int} または None
    """
    base = hiertext_base or DEFAULT_HIERTEXT_BASE

    # image_id + para_idx を抽出
    info = extract_image_info_from_path(image_path)
    if info:
        image_id, para_idx = info
        return _resolve_by_id_and_para(image_id, para_idx, base)

    # image_id のみ（para指定なし）
    image_id = extract_image_id_from_path(image_path)
    if image_id:
        return _resolve_by_id_auto_para(image_id, base)

    return None


def _find_original_image(image_id: str, base: str) -> Optional[str]:
    """image_id から元画像を検索"""
    base_path = Path(base)
    for split in SPLITS:
        original = base_path / split / f'{image_id}.jpg'
        if original.exists():
            return str(original)
    return None


def _find_mask(image_id: str, para_idx: int, base: str) -> Optional[str]:
    """image_id + para_idx からモノクロマスクを検索"""
    base_path = Path(base)
    for split in SPLITS:
        mask_file = base_path / f'Mask_Monochro_{split}' / f'mask_{image_id}_para{para_idx}.png'
        if mask_file.exists():
            return str(mask_file)
    return None


def _resolve_by_id_and_para(image_id: str, para_idx: int, base: str) -> Optional[Dict]:
    """image_id と para_idx から元画像とマスクを解決"""
    original = _find_original_image(image_id, base)
    mask = _find_mask(image_id, para_idx, base)

    if original and mask:
        logging.info(f'Resolved: image_id={image_id}, para_idx={para_idx}')
        logging.info(f'  Original image: {original}')
        logging.info(f'  Mask: {mask}')
        return {
            'image': original,
            'mask': mask,
            'image_id': image_id,
            'para_idx': para_idx,
        }

    return None


def _resolve_by_id_auto_para(image_id: str, base: str) -> Optional[Dict]:
    """image_id のみから最初に見つかった para_idx で解決"""
    base_path = Path(base)

    original = _find_original_image(image_id, base)
    if not original:
        return None

    for split in SPLITS:
        mask_dir = base_path / f'Mask_Monochro_{split}'
        if not mask_dir.exists():
            continue

        pattern = f'mask_{image_id}_para*.png'
        matches = sorted(mask_dir.glob(pattern))

        for mask_file in matches:
            info = extract_image_info_from_path(str(mask_file))
            if info:
                _, para_idx = info
                logging.info(f'Auto-selected: image_id={image_id}, para_idx={para_idx}')
                logging.info(f'  Original image: {original}')
                logging.info(f'  Mask: {mask_file}')
                return {
                    'image': original,
                    'mask': str(mask_file),
                    'image_id': image_id,
                    'para_idx': para_idx,
                }

    return None


# Qwen推論結果キャッシュ
_qwen_predictions_cache: Dict[str, Dict] = {}

# デフォルトのQwen推論結果ファイルパス
DEFAULT_QWEN_PREDICTIONS_FILE = '/home/user/dev/dev_hiertext/qwen_predictions.json'


def load_qwen_predictions(predictions_file: str) -> Dict:
    """
    Qwen推論結果JSONファイルを読み込む

    Args:
        predictions_file: 推論結果JSONファイルパス

    Returns:
        {(image_id, para_idx): {"predicted_text": "...", ...}, ...}
    """
    global _qwen_predictions_cache

    if predictions_file in _qwen_predictions_cache:
        return _qwen_predictions_cache[predictions_file]

    if not os.path.exists(predictions_file):
        return {}

    logging.info(f'Loading Qwen predictions from {predictions_file}...')

    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions_dict = {}
    for _, value in data.get('predictions', {}).items():
        image_id = value.get('image_id')
        para_idx = value.get('para_idx')
        if image_id is not None and para_idx is not None:
            predictions_dict[(image_id, para_idx)] = value

    logging.info(f'Loaded {len(predictions_dict)} Qwen predictions')
    _qwen_predictions_cache[predictions_file] = predictions_dict
    return predictions_dict


def infer_predicted_text(image_path: str, predictions_file: str = None) -> Optional[str]:
    """
    画像パスから対応するQwen推論テキストを取得

    Args:
        image_path: 画像パス
        predictions_file: Qwen推論結果JSONファイルパス（Noneの場合はデフォルトパスを使用）

    Returns:
        推論テキスト、取得失敗時は None
    """
    # 画像情報を抽出
    image_info = extract_image_info_from_path(image_path)
    if not image_info:
        return None

    # Qwen推論結果から取得
    pred_file = predictions_file or DEFAULT_QWEN_PREDICTIONS_FILE
    if not os.path.exists(pred_file):
        logging.warning(f'Predictions file not found: {pred_file}')
        return None

    try:
        predictions = load_qwen_predictions(pred_file)
        pred_data = predictions.get(image_info)
        if pred_data and pred_data.get('predicted_text'):
            logging.info(f'Using Qwen prediction for {image_info}')
            return pred_data['predicted_text']
    except Exception as e:
        logging.warning(f'Failed to load Qwen predictions: {e}')

    return None


def _infer_text_by_id(image_id: str, para_idx: int, predictions_file: str = None) -> Optional[str]:
    """image_id と para_idx から直接Qwen推論テキストを取得"""
    pred_file = predictions_file or DEFAULT_QWEN_PREDICTIONS_FILE
    if not os.path.exists(pred_file):
        return None
    try:
        predictions = load_qwen_predictions(pred_file)
        pred_data = predictions.get((image_id, para_idx))
        if pred_data and pred_data.get('predicted_text'):
            logging.info(f'Using Qwen prediction for ({image_id}, {para_idx})')
            return pred_data['predicted_text']
    except Exception as e:
        logging.warning(f'Failed to load Qwen predictions: {e}')
    return None


def create_generator(args):
    """Generatorを作成"""
    logging.info(f'Using LocalAnyTextGenerator (API: {args.api_endpoint})')
    return LocalAnyTextGenerator(
        api_endpoint=args.api_endpoint,
        font_path=args.font_path,
        output_dir=args.output_dir,
        ddim_steps=args.ddim_steps,
        strength=args.strength,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        sam_model_path=args.sam_model_path,
        use_sam=args.use_sam,
        save_glyphs=args.external_glyphs,
        use_crop=not args.no_crop,
    )


def create_evaluator(args):
    """Evaluatorを作成"""
    if args.use_vllm:
        logging.info(f'Using QwenVLEvaluator (vLLM: {args.vllm_api_base})')
        return QwenVLEvaluator(
            model_name=args.vlm_model,
            api_base=args.vllm_api_base,
            use_vllm=True,
            adjust_text_length=args.adjust_text_length,
        )

    logging.info('Using TransformersQwenEvaluator')
    return TransformersQwenEvaluator(model_name=args.vlm_model, device=args.device, torch_dtype=args.torch_dtype)


def run_correction(args):
    """テキスト修正を実行"""
    # コンポーネント作成
    generator = create_generator(args)
    evaluator = create_evaluator(args)

    # 設定
    config = TextCorrectionConfig(
        max_iterations=args.max_iterations,
        target_score=args.target_score,
        early_stop_no_improvement=args.early_stop,
        verbose=args.verbose,
    )

    # エージェント作成
    agent = TextCorrectionAgent(generator=generator, evaluator=evaluator, config=config)

    # 実行
    logging.info('=' * 60)
    logging.info('Starting Text Correction Agent')
    logging.info('=' * 60)
    logging.info(f'Image: {args.image}')
    logging.info(f'Mask: {args.mask}')
    logging.info(f'Target Text: {args.text}')
    logging.info(f'Target Score: {args.target_score}')
    logging.info(f'Max Iterations: {args.max_iterations}')
    logging.info(
        f'Initial Params: strength={generator.strength}, cfg_scale={generator.cfg_scale}, ddim_steps={generator.ddim_steps}'
    )
    logging.info('=' * 60)

    result = agent.run(
        original_image_path=args.image,
        mask_image_path=str(args.mask),  # 文字列に変換
        ground_truth_text=args.text,
        initial_text=args.initial_text or args.text,
        style_prompt=args.style_prompt,
    )

    # 結果表示
    print('\n' + '=' * 60)
    print('CORRECTION RESULTS')
    print('=' * 60)
    print(f'Status: {result.status.value}')
    print(f'Final Score: {result.final_score:.3f}')
    print(f'Iterations Used: {result.iterations_used}')
    print(f'Target Text:   {args.text}')
    print(f'Final Text:    {result.final_text}')

    # 同率1位の画像を全て列挙
    best_score = result.final_score
    tied = [
        (i + 1, gen.generated_image_path)
        for i, (ev, gen) in enumerate(zip(result.evaluation_history, result.generation_history))
        if ev.overall_score == best_score
    ]
    if len(tied) > 1:
        print(f'Final Images (tied at {best_score:.3f}):')
        for iter_num, path in tied:
            print(f'  Iteration {iter_num}: {path}')
    else:
        print(f'Final Image: {result.final_image_path}')

    if result.status == LoopStatus.SUCCESS:
        print('\n✓ Target score achieved!')
    elif result.status == LoopStatus.MAX_ITERATIONS:
        print('\n✗ Max iterations reached')
    elif result.status == LoopStatus.EARLY_STOP:
        print('\n⚠ Early stopped (no improvement) - returning best result')
    elif result.status == LoopStatus.ERROR:
        print('\n✗ Error occurred')

    # 評価履歴
    if args.show_history:
        print('\n' + '-' * 40)
        print('Evaluation History:')
        print('-' * 40)
        for i, eval_result in enumerate(result.evaluation_history, 1):
            print(f'\nIteration {i}:')
            print(f'  Score: {eval_result.overall_score:.3f}')
            print(f'  Text Accuracy: {eval_result.text_accuracy_score:.3f}')
            print(f'  Background Harmony: {eval_result.background_harmony_score:.3f}')

            if eval_result.detected_issues:
                print(f'  Issues: {eval_result.detected_issues}')

    return result


def main():
    parser = argparse.ArgumentParser(
        description='テキスト修正エージェント', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )

    # 必須引数
    parser.add_argument(
        '--image', '-i', required=True,
        help='画像パス（masked画像, 元画像, {id}_para{n}.jpg 形式のいずれも可。自動でmasked画像とマスクを解決）'
    )
    parser.add_argument('--mask', '-m', default=None, help='マスク画像のパス（省略時は画像パスから自動推測）')
    parser.add_argument(
        '--text', '-t', default=None, help='正解テキスト（省略時はQwen推論結果またはHierText GTから自動取得）'
    )

    # オプション引数
    parser.add_argument('--initial-text', help='初期テキスト（デフォルト: 正解テキスト）')
    parser.add_argument('--style-prompt', help='スタイル指示')
    parser.add_argument('--output-dir', default='./output', help='出力ディレクトリ')
    parser.add_argument(
        '--predictions',
        '-p',
        default=DEFAULT_QWEN_PREDICTIONS_FILE,
        help=f'Qwen推論結果JSONファイル（デフォルト: {DEFAULT_QWEN_PREDICTIONS_FILE}）',
    )

    # エージェント設定
    parser.add_argument('--max-iterations', type=int, default=10, help='最大試行回数')
    parser.add_argument('--target-score', type=float, default=0.98, help='目標スコア')
    parser.add_argument('--early-stop', type=int, default=3, help='改善なし早期停止回数（N-1回連続改善なしで停止）')
    parser.add_argument('--use-sam', action='store_true', help='SAMを使用してセグメンテーションマスクを生成')
    parser.add_argument(
        '--sam-model-path',
        type=str,
        default='/home/user/dev/VACE/models/VACE-Annotators/sam/sam_vit_b_01ec64.pth',
        help='SAM ViT-Bモデルのパス（セグメント合成用）',
    )

    # AnyText設定
    parser.add_argument('--api-endpoint', default='http://127.0.0.1:5000', help='AnyText APIサーバーのURL')
    parser.add_argument('--font-path', default='/home/user/dev/AnyText/font/Arial_Unicode.ttf', help='フォントパス')
    parser.add_argument('--ddim-steps', type=int, default=20, help='DDIMステップ数')
    parser.add_argument('--strength', type=float, default=1.0, help='編集強度')
    parser.add_argument('--cfg-scale', type=float, default=9.0, help='CFGスケール')
    parser.add_argument('--seed', type=int, default=None, help='ランダムシード')
    parser.add_argument('--external-glyphs', action='store_true', help='外部グリフ画像を生成してAnyTextに渡す')
    parser.add_argument('--no-crop', action='store_true', help='テキスト領域のクロップを行わず画像全体を使用する')

    # VLM設定
    parser.add_argument('--vlm-model', default='Qwen/Qwen2.5-VL-7B-Instruct', help='VLMモデル名')
    parser.add_argument('--use-vllm', action='store_true', help='vLLMサーバーを使用')
    parser.add_argument('--vllm-api-base', default='http://localhost:8001/v1', help='vLLM APIエンドポイント')
    parser.add_argument('--adjust-text-length', action='store_true',
                        help='テキストが長すぎる場合に評価器が短縮テキストを提案する（要: qwen_eval_server.py）')
    parser.add_argument('--device', default='cuda', help='デバイス')
    parser.add_argument('--torch-dtype', default='bfloat16', help='PyTorchデータ型')

    # その他
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細ログ')
    parser.add_argument('--show-history', action='store_true', help='評価履歴を表示')

    args = parser.parse_args()

    # ロギング設定
    setup_logging(args.verbose)

    # 入力ファイルのスマート解決
    # HierTextの命名パターンから元画像とモノクロマスクを自動解決
    resolved = resolve_image_paths(args.image)
    if resolved:
        args.resolved_image_id = resolved['image_id']
        args.resolved_para_idx = resolved['para_idx']
        args.image = resolved['image']
        if args.mask is None:
            args.mask = resolved['mask']
    else:
        args.resolved_image_id = None
        args.resolved_para_idx = None
        # HierTextパターンに一致しない場合はそのまま使用
        if not Path(args.image).exists():
            print(f'Error: Image file not found: {args.image}')
            print('Supported patterns:')
            print('  --image {id}_para{n}.jpg       (auto-resolve original image + mask)')
            print('  --image {id}.jpg               (auto-resolve, first para)')
            print('  --image masked_{id}_para{n}.jpg (auto-resolve original image + mask)')
            print('  --image any_image.jpg --mask mask.png  (explicit)')
            sys.exit(1)
        if args.mask is None:
            print(f'Error: Could not infer mask path from image: {args.image}')
            print('Please specify --mask explicitly')
            sys.exit(1)

    # テキストの自動取得（Qwen推論結果から）
    if args.text is None:
        # resolve済みのimage_id/para_idxがあればそれを使ってテキスト推論
        inferred_text = None
        if args.resolved_image_id and args.resolved_para_idx is not None:
            inferred_text = _infer_text_by_id(
                args.resolved_image_id, args.resolved_para_idx, args.predictions
            )
        if not inferred_text:
            inferred_text = infer_predicted_text(args.image, args.predictions)
        if inferred_text:
            args.text = inferred_text
            logging.info(f'Auto-inferred text: {args.text}')
        else:
            print(f'Error: Could not infer text from image: {args.image}')
            print(f'Please run inference first: python inference_qwen_hiertext.py -i {args.image}')
            print('Or specify --text explicitly')
            sys.exit(1)

    if not Path(args.mask).exists():
        print(f'Error: Mask file not found: {args.mask}')
        sys.exit(1)

    # 出力ディレクトリ作成
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 実行
    try:
        result = run_correction(args)
        sys.exit(0 if result.status == LoopStatus.SUCCESS else 1)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
        sys.exit(130)
    except Exception as e:
        logging.error(f'Error: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
