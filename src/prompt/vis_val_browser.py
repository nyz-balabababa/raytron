#!/usr/bin/env python3
"""验证集图片浏览器 —— 鼠标点选文件夹，键盘 ← → 翻页，M 键切换 mask 叠加"""
import json
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent.parent
VAL_LIST = ROOT / "test" / "tree_overcheck.txt"
PRED_JSON = ROOT / "test" / "prompt_test_output" / "train_tasks" / "pred_train_tasks.json"

DISPLAY_W = 1100
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
]


def rle_to_mask(rle, h, w):
    try:
        from pycocotools import mask as maskUtils
        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        return maskUtils.decode(rle_copy).astype(np.uint8)
    except ImportError:
        pass
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    if isinstance(counts, str):
        counts = [int(x) for x in counts.strip().split(",") if x.strip().isdigit()]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)
    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for run_len in counts:
        if val == 1:
            mask[pos:pos + run_len] = 1
        pos += run_len
        val = 1 - val
    return mask.reshape((h, w), order="F")


class Browser:
    def __init__(self):
        with open(VAL_LIST, encoding="utf-8") as f:
            self.all = [line.strip() for line in f if line.strip()]

        self.folders = sorted(set(p.split("/")[1] for p in self.all))
        self.images = []
        self.idx = 0
        self.folder = ""
        self.show_mask = False

        # 加载预测结果
        self.pred_lookup = {}
        if PRED_JSON.exists():
            with open(PRED_JSON, encoding="utf-8") as f:
                self.preds = json.load(f)
            for p in self.preds:
                img_path = p["image_path"].replace("\\", "/")
                if not img_path.startswith("test/"):
                    img_path = "test/" + img_path
                self.pred_lookup[img_path] = p["prompts"]
            print(f"已加载预测: {len(self.pred_lookup)} 张图")
        else:
            self.preds = {}

        self.win = tk.Tk()
        self.win.title("M=Mask  ← → 翻页  Q=退出")
        self.win.geometry(f"{DISPLAY_W + 20}x800")

        self.btn_frame = tk.Frame(self.win)
        self.btn_frame.pack(pady=4)
        self.btns = {}
        for f in self.folders:
            btn = tk.Button(self.btn_frame, text=f, width=10,
                            command=lambda name=f: self.switch_folder(name))
            btn.pack(side=tk.LEFT, padx=2)
            self.btns[f] = btn

        self.label = tk.Label(self.win, bg="black")
        self.label.pack(expand=True, fill=tk.BOTH)

        self.win.bind("<Left>", lambda e: self.nav(-1))
        self.win.bind("<Right>", lambda e: self.nav(1))
        self.win.bind("<KeyPress-a>", lambda e: self.nav(-1))
        self.win.bind("<KeyPress-d>", lambda e: self.nav(1))
        self.win.bind("<KeyPress-q>", lambda e: self.win.destroy())
        self.win.bind("<KeyPress-j>", lambda e: self.nav(10))
        self.win.bind("<KeyPress-k>", lambda e: self.nav(-10))
        self.win.bind("<m>", self.toggle_mask)
        self.win.bind("<M>", self.toggle_mask)

        self.switch_folder(self.folders[0])
        self.win.mainloop()

    def switch_folder(self, name):
        self.folder = name
        self.images = [p for p in self.all if p.startswith(f"test/{name}/")]
        self.idx = 0
        for f, btn in self.btns.items():
            btn.config(bg="#4a90d9" if f == name else "SystemButtonFace",
                       fg="white" if f == name else "black")
        self.show()

    def nav(self, delta):
        self.idx = max(0, min(self.idx + delta, len(self.images) - 1))
        self.show()

    def toggle_mask(self, event=None):
        self.show_mask = not self.show_mask
        print(f"  [Mask {'ON' if self.show_mask else 'OFF'}]")
        self.show()

    def show(self):
        if not self.images:
            self.label.config(image="", text="无图片")
            return

        rel_path = self.images[self.idx]
        path = ROOT / rel_path
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        std_v = np.std(gray)
        lap_v = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 缩放
        scale = min(DISPLAY_W / w, 680 / h)
        dw, dh = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((dw, dh), Image.LANCZOS)

        # mask 叠加
        prompt_info = ""
        preds = self.pred_lookup.get(rel_path, {})
        if preds and self.show_mask:
            hits = []
            pil_img = pil_img.convert("RGBA")
            for idx, (prompt, pred) in enumerate(preds.items()):
                if not pred.get("hit"):
                    continue
                rle = pred.get("rle")
                if rle is None:
                    continue
                mask = rle_to_mask(rle, rle["size"][0], rle["size"][1])
                if mask.sum() == 0:
                    continue
                # 缩放 mask 到显示尺寸
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((dw, dh), Image.NEAREST)
                mask_arr = np.array(mask_img) > 128
                # 彩色叠加
                color = COLORS[idx % len(COLORS)]
                overlay = Image.new("RGBA", (dw, dh), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                for y, x in zip(*np.where(mask_arr)):
                    draw.point((x, y), fill=(*color, 80))
                pil_img = Image.alpha_composite(pil_img, overlay)
                hits.append(f"{prompt}({pred.get('instances',0)}x {pred['score']:.2f})")

            prompt_info = "  |  " + "  ".join(hits) if hits else "  无命中"

        # 底栏
        draw = ImageDraw.Draw(pil_img)
        mask_status = " [MASK ON]" if self.show_mask else ""
        info = (f"[{self.folder}]{mask_status} {self.idx+1}/{len(self.images)}  |  "
                f"{Path(path).name}  |  {w}x{h}  |  亮度:{brightness:.0f}  std:{std_v:.0f}  lap:{lap_v:.0f}"
                f"{prompt_info}")
        try:
            font = ImageFont.truetype("consola.ttf", 12)
        except Exception:
            font = ImageFont.load_default()

        bar_h = 22
        bar = Image.new("RGBA", (dw, bar_h), (0, 0, 0, 160))
        pil_img = pil_img.convert("RGBA")
        pil_img.paste(bar, (0, dh - bar_h), bar)
        draw = ImageDraw.Draw(pil_img)
        draw.text((5, dh - bar_h + 2), info, fill=(200, 200, 200), font=font)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.label.config(image=self.photo)
        self.win.title(f"[{self.folder}]{mask_status} {self.idx+1}/{len(self.images)}  {Path(path).name}")


if __name__ == "__main__":
    Browser()
