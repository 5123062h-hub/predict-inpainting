"""
Generator モジュール

テキスト画像生成のための抽象クラスと実装。
AnyTextや他の生成モデルに差し替え可能な設計
"""

import base64
import logging
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import requests

from .models import GenerationResult, GenerationRequest

logger = logging.getLogger(__name__)


def _encode_image_to_base64(img: np.ndarray) -> str:
    """numpy画像(BGR)をBase64 PNG文字列にエンコード"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def _decode_image_from_base64(b64_str: str) -> np.ndarray:
    """Base64 PNG文字列からnumpy画像(BGR)をデコード"""
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _encode_glyph_to_base64(glyph: np.ndarray) -> str:
    """glyph (float64, 0.0/1.0, H x W x 1) をBase64 PNG文字列にエンコード"""
    img_uint8 = (glyph[..., 0] * 255).clip(0, 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', img_uint8)
    return base64.b64encode(buffer).decode('utf-8')


class BaseGenerator(ABC):
    """画像生成の抽象基底クラス"""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        テキストを含む画像を生成する

        Args:
            request: 生成リクエスト

        Returns:
            GenerationResult: 生成結果
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        ジェネレーターが利用可能かどうかを確認

        Returns:
            bool: 利用可能な場合True
        """
        pass


class AnyTextGenerator(BaseGenerator):
    """
    AnyText APIを使用した画像生成器

    AnyTextモデルをAPI経由で呼び出し、マスク領域にテキストを合成する。
    """

    def __init__(
        self,
        api_endpoint: str = 'http://localhost:7860/api/generate',
        output_dir: str = './generated_images',
        timeout: int = 120,
    ):
        """
        Args:
            api_endpoint: AnyText APIのエンドポイント
            output_dir: 生成画像の保存ディレクトリ
            timeout: APIリクエストのタイムアウト（秒）
        """
        self.api_endpoint = api_endpoint
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def is_available(self) -> bool:
        """AnyText APIが利用可能か確認"""
        try:
            # ヘルスチェックエンドポイントを確認
            health_url = self.api_endpoint.rsplit('/', 1)[0] + '/health'
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f'AnyText API is not available: {e}')
            return False

    def _encode_image_to_base64(self, image_path: str) -> str:
        """画像をBase64エンコード"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _save_base64_image(self, base64_data: str, output_path: Path) -> None:
        """Base64画像をファイルに保存"""
        image_data = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_data)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        AnyText APIを使用して画像を生成

        Args:
            request: 生成リクエスト

        Returns:
            GenerationResult: 生成結果
        """
        try:
            # リクエストペイロードを構築
            payload = {
                'text': request.text,
                'original_image': self._encode_image_to_base64(request.original_image_path),
                'mask_image': self._encode_image_to_base64(request.mask_image_path),
            }

            if request.style_prompt:
                payload['style_prompt'] = request.style_prompt

            # API呼び出し
            response = requests.post(self.api_endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()

            # 生成画像を保存
            import uuid

            output_filename = f'generated_{uuid.uuid4().hex[:8]}.png'
            output_path = self.output_dir / output_filename

            self._save_base64_image(result['image'], output_path)

            return GenerationResult(generated_image_path=str(output_path), used_text=request.text, success=True)

        except requests.exceptions.RequestException as e:
            logger.error(f'AnyText API request failed: {e}')
            return GenerationResult(
                generated_image_path='', used_text=request.text, success=False, error_message=str(e)
            )
        except Exception as e:
            logger.error(f'Unexpected error during generation: {e}')
            return GenerationResult(
                generated_image_path='', used_text=request.text, success=False, error_message=str(e)
            )


class LocalAnyTextGenerator(BaseGenerator):
    """
    AnyText API経由の画像生成器

    前処理（SAMセグメント、glyph生成等）はローカルで実行し、
    AnyTextの推論のみHTTP API経由で別プロセスのサーバーに委譲する。
    これにより依存パッケージのバージョン衝突を回避する。

    サーバー起動方法:
        cd /home/user/dev/AnyText
        CUDA_VISIBLE_DEVICES=0 python3 api_server.py --port 5000
    """

    def __init__(
        self,
        api_endpoint: str = 'http://127.0.0.1:5000',
        font_path: str = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        scene_font_path: Optional[str] = None,
        output_dir: str = './generated_images',
        ddim_steps: int = 50,
        strength: float = 1.0,
        cfg_scale: float = 12.0,
        seed: Optional[int] = None,
        a_prompt: str = 'best quality, extremely detailed',
        n_prompt: str = 'low quality, watermark, bad anatomy',
        sam_model_path: str = '/home/user/dev/VACE/models/VACE-Annotators/sam/sam_vit_b_01ec64.pth',
        api_timeout: int = 300,
        use_sam: bool = False,
        save_glyphs: bool = True,
        use_crop: bool = True,
    ):
        """
        Args:
            api_endpoint: AnyText APIサーバーのベースURL
            font_path: フォントファイルのパス（glyph描画用）
            scene_font_path: glyphレンダリング用フォント（Noneの場合はfont_pathと同じ）
            output_dir: 生成画像の保存ディレクトリ
            ddim_steps: 推論ステップ数
            strength: 編集強度
            cfg_scale: CFGスケール
            seed: ランダムシード（Noneの場合はランダム）
            a_prompt: 追加のポジティブプロンプト
            n_prompt: ネガティブプロンプト
            sam_model_path: SAM ViT-Bモデルのパス（セグメント合成用）
            api_timeout: APIリクエストのタイムアウト（秒）
            use_sam: SAMを使用してセグメントマスクを生成するかどうか
            save_glyphs: 生成されたグリフ画像を保存するかどうか
            use_crop: テキスト領域をクロップして生成するかどうか
        """
        self.api_endpoint = api_endpoint.rstrip('/')
        self.font_path = font_path
        self.scene_font_path = scene_font_path or font_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 推論パラメータ
        self.ddim_steps = ddim_steps
        self.strength = strength
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.a_prompt = a_prompt
        self.n_prompt = n_prompt
        self.sam_model_path = sam_model_path
        self.api_timeout = api_timeout
        self.use_sam = use_sam
        self.save_glyphs = save_glyphs
        self.use_crop = use_crop

        self._sam_predictor = None
        self._call_count = 0

        # パラメータプリセット（スコアに応じて切り替え）
        self.param_presets = [
            # プリセット0: デフォルト
            {'ddim_steps': 50, 'cfg_scale': 12.0, 'strength': 1.0},
            # プリセット1: cfg下げてバランス調整
            {'ddim_steps': 50, 'cfg_scale': 9.0, 'strength': 1.0},
            # プリセット2: strength下げて元画像寄り
            {'ddim_steps': 50, 'cfg_scale': 12.0, 'strength': 0.8},
            # プリセット3: 低cfg + 低strength
            {'ddim_steps': 50, 'cfg_scale': 7.5, 'strength': 0.8},
        ]
        self.current_preset = 0

    def adjust_parameters(self, score: float, iteration: int) -> dict:
        """
        スコアとイテレーションに基づいてパラメータを調整

        Args:
            score: 前回の評価スコア
            iteration: 現在のイテレーション番号

        Returns:
            調整されたパラメータの辞書
        """
        # スコアが低い場合、次のプリセットに切り替え
        if score < 0.5:
            self.current_preset = min(self.current_preset + 1, len(self.param_presets) - 1)

        preset = self.param_presets[self.current_preset]
        self.ddim_steps = preset['ddim_steps']
        self.cfg_scale = preset['cfg_scale']
        self.strength = preset['strength']

        # シードをランダム化（同じ結果を避ける）
        self.seed = None

        logger.info(
            f'Adjusted parameters: preset={self.current_preset}, ddim_steps={self.ddim_steps}, cfg_scale={self.cfg_scale}, strength={self.strength}'
        )
        return preset

    def reset_parameters(self):
        """パラメータをデフォルトにリセット"""
        self.current_preset = 0
        preset = self.param_presets[0]
        self.ddim_steps = preset['ddim_steps']
        self.cfg_scale = preset['cfg_scale']
        self.strength = preset['strength']

    def is_available(self) -> bool:
        """AnyText APIサーバーが利用可能か確認"""
        try:
            response = requests.get(f'{self.api_endpoint}/health', timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f'AnyText API is not available: {e}')
            return False

    def _call_anytext_api(self, input_data: dict, params: dict) -> tuple:
        """
        AnyText APIサーバーに推論リクエストを送信

        Args:
            input_data: prompt, seed, draw_pos(numpy), ori_image(numpy)
            params: mode, ddim_steps, etc.

        Returns:
            (results, rtn_code, rtn_warning) - resultsはnumpy画像のリスト
        """
        # リクエストペイロードを構築
        payload = {
            'prompt': input_data['prompt'],
            'seed': int(input_data['seed']),
            'draw_pos': _encode_image_to_base64(input_data['draw_pos']),
            'ori_image': _encode_image_to_base64(input_data['ori_image']),
        }

        # external_glyphsをエンコード
        external_glyphs = params.pop('external_glyphs', None)
        if external_glyphs is not None:
            encoded_glyphs = []
            for g in external_glyphs:
                if g is not None:
                    encoded_glyphs.append(_encode_glyph_to_base64(g))
                else:
                    encoded_glyphs.append(None)
            payload['external_glyphs'] = encoded_glyphs

        # その他のパラメータをペイロードに追加
        payload.update(params)

        # API呼び出し
        response = requests.post(
            f'{self.api_endpoint}/generate',
            json=payload,
            timeout=self.api_timeout,
        )
        if response.status_code != 200:
            # Extract server error details before raising
            try:
                err_body = response.json()
                server_msg = err_body.get('rtn_warning', response.text)
            except Exception:
                server_msg = response.text
            logger.error(f'AnyText API server error ({response.status_code}): {server_msg}')
            response.raise_for_status()
        result = response.json()

        # 結果画像をデコード
        decoded_images = []
        for b64_img in result.get('images', []):
            decoded_images.append(_decode_image_from_base64(b64_img))

        return decoded_images, result.get('rtn_code', -1), result.get('rtn_warning', '')

    def _load_image(self, image_path: str):
        """画像を読み込む"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'Failed to load image: {image_path}')
        return img

    def _resize_image(self, img, max_length: int = 768):
        """
        AnyText 推論用に画像をリサイズ・正方形パディングする。
        - 長辺を max_length にリサイズ（拡大も許容）
        - 短辺を黒パディングで正方形化
        - 結果は max_length の正方形、64の倍数に調整
        """
        h, w = img.shape[:2]

        # 1. 長辺を max_length に合わせてリサイズ（拡大・縮小両対応）
        if h >= w:
            new_h = max_length
            new_w = int(round(w * max_length / h))
        else:
            new_w = max_length
            new_h = int(round(h * max_length / w))

        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # 2. 正方形になるよう黒パディング
        if new_h != max_length or new_w != max_length:
            top = bottom = left = right = 0
            if new_h < max_length:
                pad = max_length - new_h
                top = pad // 2
                bottom = pad - top
            if new_w < max_length:
                pad = max_length - new_w
                left = pad // 2
                right = pad - left
            img = cv2.copyMakeBorder(
                img,
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),  # 黒パディング
            )

        # 3. 64の倍数に調整（念のため）
        h, w = img.shape[:2]
        new_w = w - (w % 64)
        new_h = h - (h % 64)
        if new_w != w or new_h != h:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        return img

    def _load_sam(self):
        """SAM ViT-Bモデルをロード（CPU上で実行してGPUメモリを節約）"""
        from segment_anything import SamPredictor, sam_model_registry

        logger.info(f'Loading SAM model from {self.sam_model_path}')
        sam = sam_model_registry['vit_b'](checkpoint=self.sam_model_path)
        sam.to('cpu')
        self._sam_predictor = SamPredictor(sam)
        logger.info('SAM model loaded (CPU)')

    def _get_segment_mask(self, image, mask_gray):
        """
        SAMを使ってマスク位置のオブジェクトセグメントを取得。

        各マスク領域の重心をポイントプロンプトとしてSAMに渡し、
        対応するオブジェクトセグメントを取得して統合する。

        Args:
            image: 元画像 (BGR, numpy array)
            mask_gray: テキストマスク (grayscale, uint8)

        Returns:
            segment_mask: セグメントマスク (binary uint8, 0 or 255)
                          SAM失敗時は元マスクをフォールバックとして返す
        """
        # SAMモデルをロード（初回のみ）
        if self._sam_predictor is None:
            self._load_sam()

        # マスク領域をinpaintして白塗り部分を除去（SAMが実際のオブジェクトを認識できるように）
        _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        image_for_sam = cv2.inpaint(image, mask_binary, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # inpaint済み画像をSAMにセット（BGRをRGBに変換）
        image_rgb = cv2.cvtColor(image_for_sam, cv2.COLOR_BGR2RGB)
        self._sam_predictor.set_image(image_rgb)

        # マスクの各連結領域の重心を取得
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning('No contours found in mask, using original mask as fallback')
            return mask_binary

        # 全領域の重心をポイントプロンプトとして収集
        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                points.append([cx, cy])

        if not points:
            logger.warning('No valid centroids found, using original mask as fallback')
            return mask_binary

        # 全ポイントを一括でSAMに渡す
        point_coords = np.array(points)
        point_labels = np.ones(len(points), dtype=np.int32)  # 全て前景

        masks, scores, _ = self._sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # 最も高スコアのマスクを選択
        best_idx = np.argmax(scores)
        segment = (masks[best_idx].astype(np.uint8)) * 255

        # セグメントが元マスクを完全にカバーしているか確認
        coverage = cv2.bitwise_and(mask_binary, segment)
        coverage_ratio = cv2.countNonZero(coverage) / max(cv2.countNonZero(mask_binary), 1)

        if coverage_ratio < 0.8:
            logger.warning(
                f'SAM segment covers only {coverage_ratio:.0%} of text mask. Merging segment with original mask.'
            )
            segment = cv2.bitwise_or(segment, mask_binary)

        logger.info(
            f'SAM segmentation: {len(points)} point(s), '
            f'best score={scores[best_idx]:.3f}, '
            f'segment pixels={cv2.countNonZero(segment)}, '
            f'mask coverage={coverage_ratio:.0%}'
        )

        # デバッグ: セグメント結果を保存
        debug_dir = self.output_dir / 'debug_segment'
        debug_dir.mkdir(parents=True, exist_ok=True)
        tag = f'{self._call_count}'
        cv2.imwrite(str(debug_dir / f'{tag}_text_mask.png'), mask_binary)
        cv2.imwrite(str(debug_dir / f'{tag}_inpainted_for_sam.png'), image_for_sam)
        cv2.imwrite(str(debug_dir / f'{tag}_segment_mask.png'), segment)
        # セグメント境界を元画像に描画
        segment_contours, _ = cv2.findContours(segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        debug_overlay = image_for_sam.copy()
        cv2.drawContours(debug_overlay, segment_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f'{tag}_segment_overlay.png'), debug_overlay)
        logger.info(f'Debug segment images saved to {debug_dir}/')

        return segment

    def _count_mask_regions(self, mask_gray, gap: int = 102) -> tuple[int, list]:
        """
        マスク内の連結領域数を検出し、位置情報を返す

        AnyText の separate_pos_imgs() と同一のソートロジック:
        重心Y座標を gap(=102) ピクセル単位でバケット化し、同一バケット内はX順。

        Args:
            mask_gray: グレースケールマスク画像
            gap: ソート時のバケットサイズ（AnyText default=102）

        Returns:
            (領域数, 領域情報リスト)
            領域情報: [(centroid_x, centroid_y, area, width, height, label_id), ...]
        """
        # 二値化
        _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # 連結成分分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
        self._last_labels = labels  # 個別領域マスク取得用に保持

        # 背景(label=0)を除外し、面積でフィルタリング
        regions = []
        min_area = 100  # 最小面積閾値
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                cx, cy = centroids[i]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                regions.append((cx, cy, area, w, h, i))

        # AnyText separate_pos_imgs() と同一: gap単位でバケット化ソート
        regions.sort(key=lambda r: (int(r[1]) // gap, int(r[0]) // gap))

        return len(regions), regions

    def _extract_region_polygons(self, mask_bin: np.ndarray, gap: int = 102) -> list[np.ndarray]:
        """
        マスクの各連結領域からポリゴン（輪郭）を抽出する。

        Args:
            mask_bin: 二値化済みマスク (uint8, 0 or 255)

        Returns:
            ポリゴン（輪郭）リスト。各要素は (N,1,2) の np.ndarray（Y→X順ソート済み）
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin)
        regions = []
        min_area = 100
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            component = ((labels == i) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            max_contour = max(contours, key=cv2.contourArea)
            poly = max_contour  # 矩形ではなく輪郭そのものをポリゴンとして使用
            cy = centroids[i][1]
            cx = centroids[i][0]
            regions.append((cy, cx, poly))

        # Y→X順でソート（_count_mask_regionsと同じ優先度）
        regions.sort(key=lambda r: (int(r[0]) // gap, int(r[1]) // gap))
        return [r[2] for r in regions]

    def _render_glyph_image(
        self,
        text: str,
        polygon: np.ndarray,
        canvas_w: int,
        canvas_h: int,
        scale: int = 2,
        vert_ang: int = 10,
        add_space: bool = False,
        save_glyph: bool = True,
        glyph_tag: str = '',
    ) -> np.ndarray:
        """
        AnyTextのdraw_glyph2()を外部で再現し、高品質glyphを生成する。

        t3_dataset.py draw_glyph2() と同一ロジックで、フォントのみ差し替え可能。

        Args:
            text: レンダリングするテキスト
            polygon: テキスト領域のポリゴン (N,2) — scale前の座標系
            canvas_w: キャンバス幅（scale前）
            canvas_h: キャンバス高さ（scale前）
            scale: 描画スケール倍率（デフォルト2、AnyTextと同じ）
            vert_ang: 縦書き判定の角度閾値
            add_space: テキストにスペースを挿入して幅を調整するか

        Returns:
            binary glyph image (0.0/1.0), shape (canvas_h*scale, canvas_w*scale, 1)
        """
        from PIL import Image, ImageDraw, ImageFont

        try:
            font = ImageFont.truetype(self.scene_font_path, size=60)
        except Exception:
            logger.warning(f'Failed to load scene font: {self.scene_font_path}, falling back to {self.font_path}')
            font = ImageFont.truetype(self.font_path, size=60)

        # AnyText draw_glyph2() と同一のロジック
        enlarge_polygon = polygon * scale
        rect = cv2.minAreaRect(enlarge_polygon)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        w, h = rect[1]
        angle = rect[2]
        if angle < -45:
            angle += 90
        angle = -angle
        if w < h:
            angle += 90

        vert = False
        if abs(angle) % 90 < vert_ang or abs(90 - abs(angle) % 90) % 90 < vert_ang:
            _w = max(box[:, 0]) - min(box[:, 0])
            _h = max(box[:, 1]) - min(box[:, 1])
            if _h >= _w:
                vert = True
                angle = 0

        img = np.zeros((canvas_h * scale, canvas_w * scale, 3), np.uint8)
        img = Image.fromarray(img)

        # フォントサイズ推論（draw_glyph2と同一）
        image4ratio = Image.new('RGB', img.size, 'white')
        draw = ImageDraw.Draw(image4ratio)
        _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
        if _th == 0:
            return np.zeros((canvas_h * scale, canvas_w * scale, 1), dtype=np.float64)
        text_w = min(w, h) * (_tw / _th)
        if text_w <= max(w, h):
            # add space（draw_glyph2と同一）
            if len(text) > 1 and not vert and add_space:
                for i in range(1, 100):
                    text_space = self._insert_spaces(text, i)
                    _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                    if _th2 > 0 and min(w, h) * (_tw2 / _th2) > max(w, h):
                        break
                text = self._insert_spaces(text, i - 1)
            font_size = min(w, h) * 0.80
        else:
            shrink = 0.85 if vert else 0.85
            font_size = min(w, h) / (text_w / max(w, h)) * shrink
        font_size = max(int(font_size), 1)
        new_font = font.font_variant(size=font_size)

        # フォントサイズ調整：領域からはみ出す場合は縮小する
        for _ in range(10):
            if not vert:
                left, top, right, bottom = new_font.getbbox(text)
                text_width = right - left
                text_height = bottom - top
                limit_w = max(w, h) * 0.7
                limit_h = min(w, h) * 0.7
                if text_width > limit_w or text_height > limit_h:
                    logger.debug(
                        f'Glyph too large for 0.7x target: {text_width}x{text_height} > {limit_w:.0f}x{limit_h:.0f}, shrinking font {font_size}'
                    )
                    font_size = int(font_size * 0.9)
                    new_font = font.font_variant(size=font_size)
                else:
                    break
            else:
                total_h = 0
                max_char_w = 0
                for c in text:
                    _, _, _, _b = new_font.getbbox(c)
                    total_h += _b
                    l, _, r, _ = new_font.getbbox(c)
                    max_char_w = max(max_char_w, r - l)
                if total_h > _h * 0.7 or max_char_w > _w * 0.7:
                    logger.debug(
                        f'Vertical glyph too large for 0.7x target: {max_char_w}x{total_h} > {_w * 0.7:.0f}x{_h * 0.7:.0f}, shrinking font {font_size}'
                    )
                    font_size = int(font_size * 0.9)
                    new_font = font.font_variant(size=font_size)
                else:
                    break
            if font_size < 5:
                break

        left, top, right, bottom = new_font.getbbox(text)
        text_width = right - left
        text_height = bottom - top

        layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        if not vert:
            draw.text(
                (rect[0][0] - text_width // 2, rect[0][1] - text_height // 2 - top),
                text,
                font=new_font,
                fill=(255, 255, 255, 255),
            )
        else:
            x_s = min(box[:, 0]) + _w // 2 - text_height // 2
            y_s = min(box[:, 1])
            for c in text:
                draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
                _, _t, _, _b = new_font.getbbox(c)
                y_s += _b

        rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))
        x_offset = int((img.width - rotated_layer.width) / 2)
        y_offset = int((img.height - rotated_layer.height) / 2)
        img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
        img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)

        logger.debug(
            f"Rendered glyph: text='{text}', font_size={font_size}, "
            f'rect=({w:.0f}x{h:.0f}), angle={angle:.1f}, vert={vert}'
        )

        if save_glyph:
            debug_dir = self.output_dir / 'debug_glyphs'
            debug_dir.mkdir(parents=True, exist_ok=True)

            # グリフ画像にポリゴン枠を描画して保存
            save_img = (img[..., 0] * 255).astype(np.uint8)
            save_img_color = cv2.cvtColor(save_img, cv2.COLOR_GRAY2BGR)
            poly_scaled = (polygon * scale).astype(np.int32)
            cv2.polylines(save_img_color, [poly_scaled], True, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f'glyph_{glyph_tag}.png'), save_img_color)

        return img

    @staticmethod
    def _insert_spaces(string: str, n_space: int) -> str:
        """文字間にスペースを挿入（AnyTextのinsert_spaces()と同一）"""
        if n_space == 0:
            return string
        new_string = ''
        for char in string:
            new_string += char + ' ' * n_space
        return new_string[:-n_space]

    def _glyph_to_position_mask(
        self, glyph_img: np.ndarray, canvas_w: int, canvas_h: int, scale: int = 2
    ) -> np.ndarray:
        """
        glyph画像からAnyText用のposition maskを生成する。

        AnyTextのrevise_pos=True相当の処理:
        glyphを膨張 → morphological close → 外接矩形で塗り潰し

        Args:
            glyph_img: _render_glyph_image()の出力 (float64, 0.0/1.0, h*scale x w*scale x 1)
            canvas_w: position maskの幅（scale前）
            canvas_h: position maskの高さ（scale前）
            scale: glyph_imgのスケール倍率

        Returns:
            position mask (uint8, 0 or 255), shape (canvas_h, canvas_w, 1)
        """
        # glyphをscale前の解像度にリサイズ
        glyph_resized = cv2.resize(glyph_img, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
        if len(glyph_resized.shape) == 3:
            glyph_resized = glyph_resized[..., 0]
        glyph_uint8 = (glyph_resized * 255).astype(np.uint8)

        # morphological close（文字間ギャップを埋める）— revise_posと同様
        kernel_h = max(canvas_h // 10, 1)
        kernel_w = max(canvas_w // 10, 1)
        kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)
        closed = cv2.morphologyEx(glyph_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 外接矩形で塗り潰し
        if len(closed.shape) == 2:
            closed = closed[..., np.newaxis]
        contours, _ = cv2.findContours(
            closed if len(closed.shape) == 2 else closed[..., 0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        pos_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if len(contours) == 1:
            hull = cv2.convexHull(contours[0])
            cv2.drawContours(pos_mask, [hull], -1, 255, -1)
        elif len(contours) > 1:
            # 複数輪郭の場合は全体の凸包
            all_points = np.vstack(contours)
            hull = cv2.convexHull(all_points)
            cv2.drawContours(pos_mask, [hull], -1, 255, -1)
        else:
            # フォールバック: glyph自体をマスクとして使用
            pos_mask = glyph_uint8 if len(glyph_uint8.shape) == 2 else glyph_uint8[..., 0]

        return pos_mask[..., np.newaxis]

    def _split_text_for_regions(self, text: str, num_regions: int, regions: list = None) -> list[str]:
        """
        テキストを領域数・サイズに応じて分割する。
        各パートは必ず1文字以上を持つ。テキストが短い場合は
        num_regions を文字数に合わせて切り詰めて返す。

        Args:
            text: 分割するテキスト
            num_regions: マスク領域の数
            regions: 領域情報リスト [(cx, cy, area, w, h, label_id), ...]

        Returns:
            分割されたテキストのリスト（各要素は必ず非空）
        """
        if num_regions <= 1:
            return [text]

        # テキストの実文字数（スペース除く）以上には分割できない
        num_regions = min(num_regions, len(text.strip()))
        if num_regions <= 1:
            return [text]
        if regions and len(regions) > num_regions:
            regions = regions[:num_regions]

        # まず単語で分割を試みる
        words = text.split()

        if len(words) >= num_regions:
            # 領域サイズに応じて単語を分配
            if regions and len(regions) == num_regions:
                total_area = sum(r[2] for r in regions)
                area_ratios = [r[2] / total_area for r in regions]

                result = []
                word_idx = 0
                for i, ratio in enumerate(area_ratios):
                    if i == num_regions - 1:
                        chunk = ' '.join(words[word_idx:])
                    else:
                        remaining_regions = num_regions - i
                        remaining_words = len(words) - word_idx
                        # 後続の各領域に最低1単語を残す
                        num_words = max(1, min(
                            round(len(words) * ratio),
                            remaining_words - (remaining_regions - 1),
                        ))
                        chunk = ' '.join(words[word_idx : word_idx + num_words])
                        word_idx += num_words
                    result.append(chunk)
                return result
            else:
                # 均等分配（フォールバック）
                result = []
                words_per_region = len(words) / num_regions
                start = 0
                for i in range(num_regions):
                    end = int((i + 1) * words_per_region) if i < num_regions - 1 else len(words)
                    end = max(end, start + 1)  # 最低1単語
                    chunk = ' '.join(words[start:end])
                    result.append(chunk)
                    start = end
                return result
        else:
            # 単語数が足りない場合は文字で分割（スペースを除いた文字列を使用）
            chars = text.strip()
            if regions and len(regions) == num_regions:
                total_area = sum(r[2] for r in regions)
                area_ratios = [r[2] / total_area for r in regions]

                result = []
                char_idx = 0
                for i, ratio in enumerate(area_ratios):
                    if i == num_regions - 1:
                        chunk = chars[char_idx:]
                    else:
                        remaining_regions = num_regions - i
                        remaining_chars = len(chars) - char_idx
                        # 後続の各領域に最低1文字を残す
                        num_chars = max(1, min(
                            round(len(chars) * ratio),
                            remaining_chars - (remaining_regions - 1),
                        ))
                        chunk = chars[char_idx : char_idx + num_chars]
                        char_idx += num_chars
                    result.append(chunk)
                return result
            else:
                # 均等分配（フォールバック）
                chars_per_region = len(chars) / num_regions
                result = []
                for i in range(num_regions):
                    start = int(i * chars_per_region)
                    end = int((i + 1) * chars_per_region) if i < num_regions - 1 else len(chars)
                    end = max(end, start + 1)  # 最低1文字
                    result.append(chars[start:end])
                return result

    # AnyText学習データで使用されている接続フレーズ（phrase_list）
    _CONNECTOR_PHRASES = [
        'these texts are written on it:',
        'with the words of:',
        'the written materials on the picture:',
        'textual material depicted in the image:',
        'content and position of the texts are:',
    ]

    def _create_prompt_with_texts(self, texts: list[str], style_prompt: Optional[str] = None) -> str:
        """
        複数テキストを含むプロンプトを作成

        AnyTextは "text" の形式でテキストを指定する必要がある
        複数テキストの場合: "text1" and "text2" and "text3"

        modify_prompt()でテキスト部分が*に置換された後、残りがCLIPの
        シーン記述になるため、AnyText学習データのphrase_listに合致する
        自然な表現を使う。
        """
        if len(texts) == 1:
            quoted = f'"{texts[0]}"'
        else:
            quoted = ' and '.join(f'"{t}"' for t in texts)

        # AnyText論文準拠: シーンキャプション + 接続フレーズ + テキスト指定
        connector = self._CONNECTOR_PHRASES[self._call_count % len(self._CONNECTOR_PHRASES)]

        if style_prompt:
            return f'{style_prompt}, {connector} {quoted}'
        return f'a sign, {connector} {quoted}'

    def _create_prompt_with_text(self, text: str, style_prompt: Optional[str] = None) -> str:
        """
        テキストを含むプロンプトを作成（後方互換性のため維持）
        """
        return self._create_prompt_with_texts([text], style_prompt)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        AnyText APIを使用して画像を生成

        Agent の役割:
        - テキスト領域のクロップ＋アップスケール（小テキストでも十分な解像度を確保）
        - テキスト分割・プロンプト構築
        - 結果を元画像に貼り戻す

        AnyText に委ねる処理:
        - ポリゴン抽出・glyph レンダリング・position mask 作成
        - latent-space inpainting + ピクセル空間ブレンド

        Args:
            request: 生成リクエスト

        Returns:
            GenerationResult: 生成結果
        """
        import uuid

        try:
            self._call_count += 1

            # デバッグ用ディレクトリ作成
            debug_dir = self.output_dir / 'debug_intermediate'
            debug_dir.mkdir(parents=True, exist_ok=True)
            tag = f'{self._call_count}'

            # 画像を読み込む
            ori_image = self._load_image(request.original_image_path)
            mask_image = self._load_image(request.mask_image_path)

            # AnyText util.py resize_image() と同一ロジックでリサイズ
            ori_image = self._resize_image(ori_image, max_length=768)
            mask_image = self._resize_image(mask_image, max_length=768)

            # マスク画像を処理（白い部分がテキスト領域）
            if len(mask_image.shape) == 3:
                mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_image

            h, w = ori_image.shape[:2]
            mask_resized = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)

            # テキストマスクのバウンディングボックスからクロップ範囲を決定
            _, mask_bin_for_crop = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)
            contours_for_crop, _ = cv2.findContours(mask_bin_for_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours_for_crop:
                return GenerationResult(
                    generated_image_path='',
                    used_text=request.text,
                    success=False,
                    error_message='No text regions found in mask',
                )
            all_mask_points = np.vstack(contours_for_crop)
            tx, ty, tw, th = cv2.boundingRect(all_mask_points)

            # マスクbboxを基準に正方形クロップ → 768x768にスケール
            if self.use_crop:
                crop_margin_factor = getattr(self, '_next_crop_margin', 1.0) or 1.0
                self._next_crop_margin = None  # リセット

                # マスクbboxの長辺を基準に正方形サイズを決定
                margin = max(int(max(tw, th) * crop_margin_factor), 50)
                crop_side = max(tw, th) + margin * 2

                # マスク中心を基準に正方形クロップ範囲を決定
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

                # 画像より大きい場合はクランプ
                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(w, crop_x2)
                crop_y2 = min(h, crop_y2)
            else:
                crop_x1 = 0
                crop_y1 = 0
                crop_x2 = w
                crop_y2 = h

            # クロップ
            crop_ori = ori_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            crop_mask = mask_resized[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            crop_h, crop_w = crop_ori.shape[:2]

            # 中間保存: クロップ画像
            cv2.imwrite(str(debug_dir / f'{tag}_1_crop_ori.png'), crop_ori)
            cv2.imwrite(str(debug_dir / f'{tag}_1_crop_mask.png'), crop_mask)

            logger.info(
                f'Square crop: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2}) '
                f'= {crop_w}x{crop_h}, text bbox=({tx},{ty},{tw},{th})'
            )

            # 正方形クロップを768x768にスケール
            up_w = 768
            up_h = 768
            crop_scale = up_w / crop_w

            logger.info(f'Crop upscale: {crop_w}x{crop_h} -> {up_w}x{up_h} ({crop_scale:.2f}x)')

            # アップスケール
            crop_ori_up = cv2.resize(crop_ori, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
            crop_mask_up = cv2.resize(crop_mask, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
            crop_mask_bin = cv2.threshold(crop_mask_up, 127, 255, cv2.THRESH_BINARY)[1]

            # 中間保存: アップスケール画像
            cv2.imwrite(str(debug_dir / f'{tag}_2_crop_ori_up.png'), crop_ori_up)
            cv2.imwrite(str(debug_dir / f'{tag}_2_crop_mask_up.png'), crop_mask_up)
            cv2.imwrite(str(debug_dir / f'{tag}_2_crop_mask_bin.png'), crop_mask_bin)

            # テキスト分割・プロンプト生成
            # AnyText max_chars=20 per text region (ms_wrapper.py)
            # アップスケール後のマスクで領域検出（AnyTextと同じ解像度で整合させる）
            MAX_CHARS = 20
            num_regions, regions = self._count_mask_regions(crop_mask_bin)
            logger.info(f'Detected {num_regions} mask region(s)')
            text_parts = self._split_text_for_regions(request.text, num_regions, regions)
            for i, part in enumerate(text_parts):
                if len(part) > MAX_CHARS:
                    logger.warning(f'Text part {i} "{part}" exceeds {MAX_CHARS} chars, truncating')
                    text_parts[i] = part[:MAX_CHARS]
            logger.info(f'Split text into {len(text_parts)} parts: {text_parts}')

            # グリフ画像の生成と保存（save_glyphs=Trueの場合）
            external_glyphs = []
            # if self.save_glyphs:
            if True:  # 常にグリフを生成・保存する
                polygons = self._extract_region_polygons(crop_mask_bin)
                if len(polygons) == len(text_parts):
                    logger.info(f'Generating local glyphs for {len(text_parts)} regions')
                    for i, (part, poly) in enumerate(zip(text_parts, polygons)):
                        glyph = self._render_glyph_image(
                            part, poly, up_w, up_h, scale=2, save_glyph=True, glyph_tag=f'{self._call_count}_{i}'
                        )
                        external_glyphs.append(glyph)
                else:
                    logger.warning(f'Polygon count mismatch: {len(polygons)} vs {len(text_parts)}')

            prompt = self._create_prompt_with_texts(text_parts, request.style_prompt)

            # draw_pos を準備
            if self.use_sam:
                logger.info('Using SAM to generate segment mask for draw_pos...')
                # SAMはアップスケール前の元画像クロップに対して実行する方がセマンティックに正しい
                segment_mask = self._get_segment_mask(crop_ori, crop_mask)
                # SAMの結果をAnyTextへの入力サイズにリサイズ
                crop_mask_bin = cv2.resize(segment_mask, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
            else:
                logger.info('Using binary mask for draw_pos...')
            # BGR形式に変換してbase64エンコードに備える
            draw_pos = cv2.cvtColor(crop_mask_bin, cv2.COLOR_GRAY2BGR)

            # 中間保存: draw_pos
            cv2.imwrite(str(debug_dir / f'{tag}_3_draw_pos.png'), draw_pos)

            # シード設定
            seed = self.seed if self.seed is not None else np.random.randint(0, 2**31)

            # 入力データを構築
            input_data = {
                'prompt': prompt,
                'seed': seed,
                'draw_pos': draw_pos,
                'ori_image': crop_ori_up,
            }

            # 推論パラメータ
            # - external_glyphs なし: AnyText内部のdraw_glyph2()で整合したglyph/positionを生成
            # - skip_blending=False: AnyText内蔵のソフトマスクブレンドを使用
            #   (latent-space inpainting + pixel-space Gaussian blending)
            params = {
                'mode': 'edit',
                'skip_blending': False,
                'sort_priority': 'y',
                'show_debug': False,
                'revise_pos': False,
                'image_count': 1,
                'ddim_steps': self.ddim_steps,
                'strength': self.strength,
                'cfg_scale': self.cfg_scale,
                'eta': 0.0,
                'a_prompt': self.a_prompt,
                'n_prompt': self.n_prompt,
            }

            if external_glyphs:
                params['external_glyphs'] = external_glyphs

            logger.info(f'Generating image with text: {request.text}')
            logger.info(f'  style_prompt : {request.style_prompt or "(none)"}')
            logger.info(f'  final prompt : {prompt}')

            # API経由で推論実行
            results, rtn_code, rtn_warning = self._call_anytext_api(input_data, params)

            if rtn_code < 0:
                logger.error(f'AnyText generation failed: {rtn_warning}')
                return GenerationResult(
                    generated_image_path='',
                    used_text=request.text,
                    success=False,
                    error_message=rtn_warning or 'Generation failed',
                )

            if rtn_warning:
                logger.warning(f'AnyText warning: {rtn_warning}')

            # 結果画像を保存
            output_filename = f'anytext_generated_{self._call_count}_{uuid.uuid4().hex[:8]}.png'
            output_path = self.output_dir / output_filename

            if results and len(results) > 0:
                result_image = results[0]

                # API結果はBGR numpy array
                if not isinstance(result_image, np.ndarray):
                    result_image = np.array(result_image)

                # AnyText結果をクロップサイズに戻して元画像に貼り戻す
                # AnyText内蔵ブレンドにより非テキスト領域は元画像が保持されているため、
                # 単純な貼り戻しで境界は目立たない
                result_crop = cv2.resize(result_image, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                result_full = ori_image.copy()
                result_full[crop_y1:crop_y2, crop_x1:crop_x2] = result_crop

                cv2.imwrite(str(output_path), result_full)
                logger.info(f'Generated image saved: {output_path}')

                return GenerationResult(generated_image_path=str(output_path), used_text=request.text, success=True)
            else:
                return GenerationResult(
                    generated_image_path='',
                    used_text=request.text,
                    success=False,
                    error_message='No result image returned',
                )

        except Exception as e:
            logger.error(f'Error during AnyText generation: {e}')
            import traceback

            traceback.print_exc()
            return GenerationResult(
                generated_image_path='', used_text=request.text, success=False, error_message=str(e)
            )
