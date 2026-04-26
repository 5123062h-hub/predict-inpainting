"""AnyText簡易テスト — modelscope pipelineをバイパスして直接ロード"""
import sys
import os

# ローカルAnyTextをimportパスに追加
anytext_dir = os.path.join(os.path.dirname(__file__), "AnyText")
sys.path.insert(0, anytext_dir)
os.chdir(anytext_dir)  # models_yaml/ 等の相対パス解決用

import cv2
import numpy as np

# --- AnyTextModelを直接インスタンス化（modelscope pipeline不要）---
print("Loading AnyText model directly...")
from ms_wrapper import AnyTextModel

model = AnyTextModel(
    anytext_dir,
    use_fp16=True,
    use_translator=False,
    font_path=os.path.join(anytext_dir, "font", "Arial_Unicode.ttf"),
    cfg_path="models_yaml/anytext_sd15.yaml",
)
print("Model loaded!")


# =============================================
# モード1: text-generation（ゼロから画像生成）
# =============================================
print("\n=== Mode: text-generation ===")
result_gen, code, warning, debug = model.forward(
    {"prompt": 'A coffee shop sign that reads "HELLO"', "seed": 42},
    mode="gen",
    image_count=1,
    ddim_steps=20,
    image_width=512,
    image_height=512,
    cfg_scale=9.0,
    strength=1.0,
    eta=0.0,
    a_prompt="best quality, extremely detailed",
    n_prompt="low quality, watermark",
    show_debug=False,
    sort_priority="y",
    skip_blending=False,
    revise_pos=False,
)
if code >= 0 and result_gen:
    out_path = os.path.join(os.path.dirname(__file__), "test_gen_output.png")
    cv2.imwrite(out_path, result_gen[0][..., ::-1])  # RGB→BGR
    print(f"Generated: {out_path}")
else:
    print(f"Generation failed: {warning}")


# =============================================
# モード2: text-editing（既存画像にテキスト編集）
# =============================================
print("\n=== Mode: text-editing ===")
ref_path = os.path.join(anytext_dir, "example_images", "ref1.jpg")
edit_path = os.path.join(anytext_dir, "example_images", "edit1.png")

if os.path.exists(ref_path) and os.path.exists(edit_path):
    ref_img = cv2.imread(ref_path)[..., ::-1]  # BGR→RGB
    edit_img = cv2.imread(edit_path)

    result_edit, code, warning, debug = model.forward(
        {
            "prompt": 'A sign that reads "WORLD"',
            "seed": 123,
            "draw_pos": edit_img,
            "ori_image": ref_img,
        },
        mode="edit",
        image_count=1,
        ddim_steps=20,
        image_width=512,
        image_height=512,
        cfg_scale=9.0,
        strength=1.0,
        eta=0.0,
        a_prompt="best quality, extremely detailed",
        n_prompt="low quality, watermark",
        show_debug=False,
        sort_priority="y",
        skip_blending=False,
        revise_pos=True,
    )
    if code >= 0 and result_edit:
        out_path = os.path.join(os.path.dirname(__file__), "test_edit_output.png")
        cv2.imwrite(out_path, result_edit[0][..., ::-1])
        print(f"Edited: {out_path}")
    else:
        print(f"Editing failed: {warning}")
else:
    print("Skipping edit test (sample images not found)")

print("\nDone!")
