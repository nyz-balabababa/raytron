#!/usr/bin/env python3
"""伪标签浏览器

功能:
  - 浏览原图
  - 按文件夹筛选
  - 按类别筛选
  - 按命中状态/高分/低分/漏检筛选
  - M 键切换 mask 叠加
  - B 键切换 bbox 叠加

用途:
  - 抽查高分命中样本
  - 抽查低分命中样本
  - 抽查指定类别的未命中样本
  - 检查 mask -> bbox 转换是否合理
"""

import json
import tkinter as tk
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent.parent.parent

# 可按需修改为你当前要看的任务/伪标签文件
TASK_JSON = ROOT / "test" / "json" / "val_d5&7_tasks(9).json"
PRED_JSON = ROOT / "test" / "prompt_test_output" / "val_d5&7_tasks(9)" / "pred_val_d5&7_tasks(9).json"

DISPLAY_W = 1200
DISPLAY_H = 760
MASK_ALPHA = 88
LOW_SCORE_TH = 0.55
HIGH_SCORE_TH = 0.75

COLORS = [
    (255, 80, 80), (80, 220, 80), (80, 140, 255),
    (255, 220, 80), (240, 80, 220), (80, 240, 240),
    (255, 140, 0), (170, 110, 255),
]


def rle_to_mask(rle, h=None, w=None):
    try:
        from pycocotools import mask as mask_utils

        rle_copy = dict(rle)
        if isinstance(rle_copy["counts"], str):
            rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
        return mask_utils.decode(rle_copy).astype(np.uint8)
    except ImportError:
        pass

    if h is None or w is None:
        h, w = rle["size"]
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


def mask_to_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        bboxes.append((x, y, bw, bh))
    return bboxes


def load_task_index(task_json):
    prompts_by_image = defaultdict(set)
    if not task_json.exists():
        return prompts_by_image

    with open(task_json, encoding="utf-8") as f:
        tasks = json.load(f)
    for item in tasks:
        img_path = item["image_path"].replace("\\", "/")
        if not img_path.startswith("test/"):
            img_path = "test/" + img_path
        prompts_by_image[img_path].add(item["text_prompt"])
    return prompts_by_image


def load_pred_index(pred_json):
    pred_lookup = {}
    prompt_set = set()
    if not pred_json.exists():
        return pred_lookup, prompt_set

    with open(pred_json, encoding="utf-8-sig") as f:
        preds = json.load(f)
    for item in preds:
        img_path = item["image_path"].replace("\\", "/")
        if not img_path.startswith("test/"):
            img_path = "test/" + img_path
        pred_lookup[img_path] = item["prompts"]
        prompt_set.update(item["prompts"].keys())
    return pred_lookup, prompt_set


class Browser:
    def __init__(self):
        self.task_lookup = load_task_index(TASK_JSON)
        self.pred_lookup, pred_prompts = load_pred_index(PRED_JSON)
        self.all_images = sorted(set(self.task_lookup.keys()) | set(self.pred_lookup.keys()))
        self.all_prompts = ["ALL"] + sorted(pred_prompts | {p for prompts in self.task_lookup.values() for p in prompts})
        self.filter_modes = [
            ("all", "全部"),
            ("hit", "命中"),
            ("high", "高分"),
            ("low", "低分"),
            ("miss", "漏检"),
        ]

        self.folders = ["ALL"] + sorted({p.split("/")[1] for p in self.all_images if "/" in p})
        self.images = []
        self.idx = 0
        self.folder = "ALL"
        self.current_prompt = "ALL"
        self.current_mode = "all"
        self.show_mask = True
        self.show_bbox = True

        print(f"已加载任务图像: {len(self.all_images)} 张")
        print(f"已加载预测: {len(self.pred_lookup)} 张图")
        print(f"类别: {', '.join(self.all_prompts[1:])}")

        self.win = tk.Tk()
        self.win.title("M=Mask  B=BBox  ← → 翻页  Q=退出")
        self.win.geometry(f"{DISPLAY_W + 40}x960")

        self._build_controls()

        self.label = tk.Label(self.win, bg="black")
        self.label.pack(expand=True, fill=tk.BOTH)

        self._bind_keys()
        self.refresh_filters()
        self.win.mainloop()

    def _build_controls(self):
        self.folder_frame = tk.Frame(self.win)
        self.folder_frame.pack(pady=4)
        self.folder_buttons = {}
        for folder in self.folders:
            btn = tk.Button(self.folder_frame, text=folder, width=10, command=lambda name=folder: self.switch_folder(name))
            btn.pack(side=tk.LEFT, padx=2)
            self.folder_buttons[folder] = btn

        self.prompt_frame = tk.Frame(self.win)
        self.prompt_frame.pack(pady=4)
        self.prompt_buttons = {}
        for prompt in self.all_prompts:
            btn = tk.Button(
                self.prompt_frame,
                text=prompt,
                width=16 if prompt != "ALL" else 8,
                command=lambda name=prompt: self.switch_prompt(name),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.prompt_buttons[prompt] = btn

        self.mode_frame = tk.Frame(self.win)
        self.mode_frame.pack(pady=4)
        self.mode_buttons = {}
        for mode_key, mode_label in self.filter_modes:
            btn = tk.Button(self.mode_frame, text=mode_label, width=8, command=lambda key=mode_key: self.switch_mode(key))
            btn.pack(side=tk.LEFT, padx=4)
            self.mode_buttons[mode_key] = btn

        hint = (
            "快捷键: ←/→ 或 A/D 翻页, J/K 快跳, M mask, B bbox, "
            "1-5 切筛选模式, [/ ] 切类别, Q 退出"
        )
        self.hint_label = tk.Label(self.win, text=hint, fg="#555")
        self.hint_label.pack(pady=2)

    def _bind_keys(self):
        self.win.bind("<Left>", lambda e: self.nav(-1))
        self.win.bind("<Right>", lambda e: self.nav(1))
        self.win.bind("<KeyPress-a>", lambda e: self.nav(-1))
        self.win.bind("<KeyPress-d>", lambda e: self.nav(1))
        self.win.bind("<KeyPress-j>", lambda e: self.nav(10))
        self.win.bind("<KeyPress-k>", lambda e: self.nav(-10))
        self.win.bind("<KeyPress-q>", lambda e: self.win.destroy())
        self.win.bind("<KeyPress-m>", self.toggle_mask)
        self.win.bind("<KeyPress-M>", self.toggle_mask)
        self.win.bind("<KeyPress-b>", self.toggle_bbox)
        self.win.bind("<KeyPress-B>", self.toggle_bbox)
        self.win.bind("<KeyPress-1>", lambda e: self.switch_mode("all"))
        self.win.bind("<KeyPress-2>", lambda e: self.switch_mode("hit"))
        self.win.bind("<KeyPress-3>", lambda e: self.switch_mode("high"))
        self.win.bind("<KeyPress-4>", lambda e: self.switch_mode("low"))
        self.win.bind("<KeyPress-5>", lambda e: self.switch_mode("miss"))
        self.win.bind("<KeyPress-bracketleft>", lambda e: self.cycle_prompt(-1))
        self.win.bind("<KeyPress-bracketright>", lambda e: self.cycle_prompt(1))

    def cycle_prompt(self, delta):
        idx = self.all_prompts.index(self.current_prompt)
        idx = (idx + delta) % len(self.all_prompts)
        self.switch_prompt(self.all_prompts[idx])

    def switch_folder(self, name):
        self.folder = name
        self.refresh_filters()

    def switch_prompt(self, name):
        self.current_prompt = name
        self.refresh_filters()

    def switch_mode(self, mode_key):
        self.current_mode = mode_key
        self.refresh_filters()

    def toggle_mask(self, event=None):
        self.show_mask = not self.show_mask
        self.show()

    def toggle_bbox(self, event=None):
        self.show_bbox = not self.show_bbox
        self.show()

    def refresh_filters(self):
        self.images = self._filter_images()
        self.idx = 0
        self._update_buttons()
        self.show()

    def _update_buttons(self):
        for name, btn in self.folder_buttons.items():
            active = name == self.folder
            btn.config(bg="#4a90d9" if active else "SystemButtonFace", fg="white" if active else "black")
        for name, btn in self.prompt_buttons.items():
            active = name == self.current_prompt
            btn.config(bg="#3aa675" if active else "SystemButtonFace", fg="white" if active else "black")
        for key, btn in self.mode_buttons.items():
            active = key == self.current_mode
            btn.config(bg="#d97b3a" if active else "SystemButtonFace", fg="white" if active else "black")

    def _match_folder(self, img_path):
        return self.folder == "ALL" or img_path.startswith(f"test/{self.folder}/")

    def _get_prompt_pred(self, img_path, prompt):
        return self.pred_lookup.get(img_path, {}).get(prompt)

    def _score_band_match(self, pred):
        score = float(pred.get("score", 0.0))
        if self.current_mode == "high":
            return pred.get("hit") and score >= HIGH_SCORE_TH
        if self.current_mode == "low":
            return pred.get("hit") and score < LOW_SCORE_TH
        if self.current_mode == "hit":
            return pred.get("hit")
        if self.current_mode == "miss":
            return not pred.get("hit")
        return True

    def _image_has_mode_match(self, img_path):
        if self.current_prompt == "ALL":
            preds = self.pred_lookup.get(img_path, {})
            if self.current_mode == "all":
                return True
            if self.current_mode == "miss":
                task_prompts = self.task_lookup.get(img_path, set())
                return any(not self._get_prompt_pred(img_path, prompt) or not self._get_prompt_pred(img_path, prompt).get("hit") for prompt in task_prompts)
            for pred in preds.values():
                if self._score_band_match(pred):
                    return True
            return False

        task_has_prompt = self.current_prompt in self.task_lookup.get(img_path, set()) or self.current_prompt in self.pred_lookup.get(img_path, {})
        if not task_has_prompt:
            return False

        pred = self._get_prompt_pred(img_path, self.current_prompt)
        if pred is None:
            return self.current_mode in {"all", "miss"}
        return self._score_band_match(pred)

    def _filter_images(self):
        filtered = []
        for img_path in self.all_images:
            if not self._match_folder(img_path):
                continue
            if not self._image_has_mode_match(img_path):
                continue
            filtered.append(img_path)
        return filtered

    def nav(self, delta):
        if not self.images:
            return
        self.idx = max(0, min(self.idx + delta, len(self.images) - 1))
        self.show()

    def _collect_display_preds(self, rel_path):
        preds = self.pred_lookup.get(rel_path, {})
        if self.current_prompt == "ALL":
            return preds
        if self.current_prompt in preds:
            return {self.current_prompt: preds[self.current_prompt]}
        return {}

    def show(self):
        if not self.images:
            self.label.config(image="", text="当前筛选结果为空")
            self.win.title("无图片")
            return

        rel_path = self.images[self.idx]
        path = ROOT / rel_path
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            self.label.config(image="", text=f"读取失败: {path}")
            return

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        std_v = np.std(gray)
        lap_v = cv2.Laplacian(gray, cv2.CV_64F).var()

        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        dw, dh = int(w * scale), int(h * scale)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((dw, dh), Image.LANCZOS).convert("RGBA")
        draw = ImageDraw.Draw(pil_img)

        display_preds = self._collect_display_preds(rel_path)
        hit_lines = []
        miss_lines = []
        prompt_notes = []

        for pred_idx, prompt in enumerate(sorted(display_preds.keys())):
            pred = display_preds[prompt]
            hit = bool(pred.get("hit"))
            score = float(pred.get("score", 0.0)) if hit else 0.0
            instances = int(pred.get("instances", 0)) if hit else 0
            prompt_notes.append(f"{prompt}: {'hit' if hit else 'miss'}")
            if not hit or pred.get("rle") is None:
                miss_lines.append(prompt)
                continue

            mask = rle_to_mask(pred["rle"], pred["rle"]["size"][0], pred["rle"]["size"][1])
            if mask.sum() == 0:
                miss_lines.append(f"{prompt}(empty)")
                continue

            color = COLORS[pred_idx % len(COLORS)]
            if self.show_mask:
                mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize((dw, dh), Image.NEAREST)
                mask_arr = np.array(mask_img) > 128
                overlay_arr = np.zeros((dh, dw, 4), dtype=np.uint8)
                overlay_arr[mask_arr] = [color[0], color[1], color[2], MASK_ALPHA]
                overlay = Image.fromarray(overlay_arr, "RGBA")
                pil_img = Image.alpha_composite(pil_img, overlay)
                draw = ImageDraw.Draw(pil_img)

            if self.show_bbox:
                mask_disp = cv2.resize(mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST)
                for x, y, bw, bh in mask_to_bboxes(mask_disp):
                    draw.rectangle([x, y, x + bw, y + bh], outline=color, width=2)

            hit_lines.append(f"{prompt}({instances}x {score:.2f})")

        info_parts = [
            f"[{self.folder}]",
            f"{self.idx + 1}/{len(self.images)}",
            Path(path).name,
            f"{w}x{h}",
            f"亮度:{brightness:.0f}",
            f"std:{std_v:.0f}",
            f"lap:{lap_v:.0f}",
            f"prompt:{self.current_prompt}",
            f"mode:{self.current_mode}",
        ]
        if hit_lines:
            info_parts.append("命中: " + " | ".join(hit_lines))
        if miss_lines:
            info_parts.append("未命中: " + " | ".join(miss_lines))
        info = "  |  ".join(info_parts)

        try:
            font = ImageFont.truetype("consola.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

        bar_h = 26
        bar = Image.new("RGBA", (dw, bar_h), (0, 0, 0, 170))
        pil_img.paste(bar, (0, dh - bar_h), bar)
        draw = ImageDraw.Draw(pil_img)
        draw.text((6, dh - bar_h + 4), info, fill=(230, 230, 230), font=font)

        self.photo = ImageTk.PhotoImage(pil_img)
        self.label.config(image=self.photo, text="")
        title = (
            f"[{self.folder}] {self.idx + 1}/{len(self.images)} "
            f"{Path(path).name} | prompt={self.current_prompt} mode={self.current_mode}"
        )
        self.win.title(title)


if __name__ == "__main__":
    Browser()
