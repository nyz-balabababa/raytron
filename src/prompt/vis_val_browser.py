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
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

ROOT = Path(__file__).resolve().parent.parent.parent

# 可按需修改为你当前要看的任务/伪标签文件
TASK_JSON = ROOT / "test" /"json" / "val_tasks.json"
PRED_JSON = ROOT / "test" / "sam3_label_output" /"pseudo_C.json"
OLD_PRED_JSON = ROOT / "test" / "sam3_label_old" / "val_tasks1" / "pred_val_tasks1.json"
NEW_PRED_JSON = ROOT / "test" / "sam3_label_output" / "pred_val_tasks.json"
DIFF_SAMPLES_TXT = ROOT / "test" / "label_analysis" / "pseudo_label_compare" / "divergent_samples.txt"
DIFF_TOPK = 200

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


def load_divergent_items(txt_path, topk=20):
    items = []
    if not txt_path.exists():
        return items

    with open(txt_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            image_path = parts[0].replace("\\", "/")
            if not image_path.startswith("test/"):
                image_path = "test/" + image_path
            prompt = parts[1]
            meta = {
                "old_hit": parts[2] if len(parts) > 2 else "",
                "new_hit": parts[3] if len(parts) > 3 else "",
                "iou": parts[4] if len(parts) > 4 else "",
                "contrast_std": parts[5] if len(parts) > 5 else "",
            }
            items.append({"image_path": image_path, "prompt": prompt, "meta": meta})
            if len(items) >= topk:
                break
    return items


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
    task_by_ann = {}
    if not task_json.exists():
        return prompts_by_image, task_by_ann

    with open(task_json, encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(
            f"TASK_JSON 格式不对: 期望任务数组，但拿到 {type(tasks).__name__}。"
            " 浏览提交结果时，TASK_JSON 应该指向 test_tasks.json。"
        )
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError(f"TASK_JSON 中存在非对象元素: {type(item).__name__}")
        img_path = item["image_path"].replace("\\", "/")
        if not img_path.startswith("test/"):
            img_path = "test/" + img_path
        prompt = item["text_prompt"]
        prompts_by_image[img_path].add(prompt)
        task_by_ann[item["ann_id"]] = {
            "image_path": img_path,
            "text_prompt": prompt,
        }
    return prompts_by_image, task_by_ann


def load_pred_index(pred_json, task_by_ann):
    pred_lookup = {}
    prompt_set = set()
    if not pred_json.exists():
        return pred_lookup, prompt_set

    with open(pred_json, encoding="utf-8-sig") as f:
        preds = json.load(f)

    # 兼容旧的伪标签格式：[{image_path, prompts}, ...]
    if isinstance(preds, list):
        for item in preds:
            if not isinstance(item, dict):
                continue
            img_path = item["image_path"].replace("\\", "/")
            if not img_path.startswith("test/"):
                img_path = "test/" + img_path
            pred_lookup[img_path] = item["prompts"]
            prompt_set.update(item["prompts"].keys())
        return pred_lookup, prompt_set

    # 兼容提交格式：{model_info, timing, predictions:[{ann_id, rle}, ...]}
    if isinstance(preds, dict) and isinstance(preds.get("predictions"), list):
        for item in preds["predictions"]:
            if not isinstance(item, dict):
                continue
            ann_id = item.get("ann_id")
            task = task_by_ann.get(ann_id)
            if task is None:
                continue
            img_path = task["image_path"]
            prompt = task["text_prompt"]
            rle = item.get("rle")
            hit = bool(rle)
            instances = 0
            if rle:
                mask = rle_to_mask(rle, rle["size"][0], rle["size"][1])
                hit = bool(mask.sum() > 0)
                instances = len(mask_to_bboxes(mask)) if hit else 0
            pred_lookup.setdefault(img_path, {})[prompt] = {
                "hit": hit,
                "score": 1.0 if hit else 0.0,
                "instances": instances,
                "rle": rle,
            }
            prompt_set.add(prompt)
        return pred_lookup, prompt_set

    raise ValueError(
        f"PRED_JSON 格式不支持: {type(preds).__name__}。"
        " 需要是伪标签数组，或提交版 predictions.json。"
    )


class Browser:
    def __init__(self):
        self.task_json_path = TASK_JSON
        self.pred_json_path = PRED_JSON
        self.old_pred_json_path = OLD_PRED_JSON
        self.new_pred_json_path = NEW_PRED_JSON
        self.diff_samples_path = DIFF_SAMPLES_TXT
        self.filter_modes = [
            ("all", "全部"),
            ("hit", "命中"),
            ("high", "高分"),
            ("low", "低分"),
            ("miss", "漏检"),
        ]

        self.task_lookup = {}
        self.task_by_ann = {}
        self.pred_lookup = {}
        self.old_pred_lookup = {}
        self.new_pred_lookup = {}
        self.all_images = []
        self.divergent_items = []
        self.all_prompts = ["ALL"]
        self.folders = ["ALL"]
        self.images = []
        self.idx = 0
        self.folder = "ALL"
        self.current_prompt = "ALL"
        self.current_mode = "all"
        self.show_mask = True
        self.show_bbox = True
        self.use_divergent_topk = False
        self.use_current_compare = False

        print(f"已加载任务图像: {len(self.all_images)} 张")
        print(f"已加载预测: {len(self.pred_lookup)} 张图")
        print(f"类别: {', '.join(self.all_prompts[1:])}")
        if self.divergent_items:
            print(f"已加载分歧样本 Top-{len(self.divergent_items)}: {DIFF_SAMPLES_TXT}")

        self.win = tk.Tk()
        self.win.title("M=Mask  B=BBox  ← → 翻页  Q=退出")
        self.win.geometry(f"{DISPLAY_W + 40}x960")

        self._build_controls()

        self.label = tk.Label(self.win, bg="black")
        self.label.pack(expand=True, fill=tk.BOTH)

        self._bind_keys()
        self.reload_data(reset_state=True)
        self.win.mainloop()

    def _build_controls(self):
        self.file_frame = tk.Frame(self.win)
        self.file_frame.pack(pady=4, fill=tk.X)
        tk.Button(self.file_frame, text="选择任务JSON", width=12, command=self.choose_task_json).pack(side=tk.LEFT, padx=4)
        tk.Button(self.file_frame, text="选择当前新伪标签JSON", width=16, command=self.choose_pred_json).pack(side=tk.LEFT, padx=4)
        tk.Button(self.file_frame, text="选择旧伪标签", width=12, command=self.choose_old_pred_json).pack(side=tk.LEFT, padx=4)
        tk.Button(self.file_frame, text="选择新伪标签", width=12, command=self.choose_new_pred_json).pack(side=tk.LEFT, padx=4)
        tk.Button(self.file_frame, text="重新加载", width=10, command=self.reload_data).pack(side=tk.LEFT, padx=4)

        self.task_path_label = tk.Label(self.file_frame, text="", anchor="w", fg="#333")
        self.task_path_label.pack(side=tk.LEFT, padx=8)

        self.pred_path_label = tk.Label(self.win, text="", anchor="w", fg="#333")
        self.pred_path_label.pack(pady=2, fill=tk.X)
        self.compare_path_label = tk.Label(self.win, text="", anchor="w", fg="#333")
        self.compare_path_label.pack(pady=2, fill=tk.X)

        self.controls_frame = tk.Frame(self.win)
        self.controls_frame.pack(pady=2, fill=tk.X)

        self.folder_frame = None
        self.prompt_frame = None
        self.mode_frame = None
        self.extra_frame = None
        self.folder_buttons = {}
        self.prompt_buttons = {}
        self.mode_buttons = {}

        self._rebuild_dynamic_controls()

    def _rebuild_dynamic_controls(self):
        for frame in [self.folder_frame, self.prompt_frame, self.mode_frame, self.extra_frame]:
            if frame is not None:
                frame.destroy()

        self.folder_frame = tk.Frame(self.controls_frame)
        self.folder_frame.pack(pady=4)
        self.folder_buttons = {}
        for folder in self.folders:
            btn = tk.Button(self.folder_frame, text=folder, width=10, command=lambda name=folder: self.switch_folder(name))
            btn.pack(side=tk.LEFT, padx=2)
            self.folder_buttons[folder] = btn

        self.prompt_frame = tk.Frame(self.controls_frame)
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

        self.mode_frame = tk.Frame(self.controls_frame)
        self.mode_frame.pack(pady=4)
        self.mode_buttons = {}
        for mode_key, mode_label in self.filter_modes:
            btn = tk.Button(self.mode_frame, text=mode_label, width=8, command=lambda key=mode_key: self.switch_mode(key))
            btn.pack(side=tk.LEFT, padx=4)
            self.mode_buttons[mode_key] = btn

        self.extra_frame = tk.Frame(self.controls_frame)
        self.extra_frame.pack(pady=4)
        self.divergent_button = tk.Button(
            self.extra_frame,
            text=f"分歧Top-{DIFF_TOPK}",
            width=12,
            command=self.toggle_divergent_mode,
        )
        self.divergent_button.pack(side=tk.LEFT, padx=4)
        self.current_compare_button = tk.Button(
            self.extra_frame,
            text="当前JSON对比",
            width=12,
            command=self.toggle_current_compare_mode,
        )
        self.current_compare_button.pack(side=tk.LEFT, padx=4)
        self.copy_path_button = tk.Button(
            self.extra_frame,
            text="复制图片路径",
            width=12,
            command=self.copy_current_image_path,
        )
        self.copy_path_button.pack(side=tk.LEFT, padx=4)

        hint = (
            "快捷键: ←/→ 或 A/D 翻页, J/K 快跳, M mask, B bbox, "
            "1-5 切筛选模式, 6 切分歧Top, 7 切当前JSON对比, C 复制路径, [/ ] 切类别, Q 退出"
        )
        self.hint_label = tk.Label(self.win, text=hint, fg="#555")
        self.hint_label.pack(pady=2)

    def _update_path_labels(self):
        self.task_path_label.config(text=f"TASK: {self.task_json_path}")
        self.pred_path_label.config(text=f"CURRENT: {self.pred_json_path}")
        self.compare_path_label.config(text=f"OLD: {self.old_pred_json_path}    |    DEFAULT_NEW: {self.new_pred_json_path}")

    def choose_task_json(self):
        selected = filedialog.askopenfilename(
            title="选择任务 JSON",
            initialdir=str(self.task_json_path.parent),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if selected:
            self.task_json_path = Path(selected)
            self.reload_data(reset_state=True)

    def choose_pred_json(self):
        selected = filedialog.askopenfilename(
            title="选择预测/伪标签 JSON",
            initialdir=str(self.pred_json_path.parent),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if selected:
            self.pred_json_path = Path(selected)
            self.reload_data(reset_state=True)

    def choose_old_pred_json(self):
        selected = filedialog.askopenfilename(
            title="选择旧伪标签 JSON",
            initialdir=str(self.old_pred_json_path.parent),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if selected:
            self.old_pred_json_path = Path(selected)
            self.reload_data(reset_state=True)

    def choose_new_pred_json(self):
        selected = filedialog.askopenfilename(
            title="选择新伪标签 JSON",
            initialdir=str(self.new_pred_json_path.parent),
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if selected:
            self.new_pred_json_path = Path(selected)
            self.reload_data(reset_state=True)

    def reload_data(self, reset_state=False):
        try:
            task_lookup, task_by_ann = load_task_index(self.task_json_path)
            pred_lookup, pred_prompts = load_pred_index(self.pred_json_path, task_by_ann)
            old_pred_lookup, _ = load_pred_index(self.old_pred_json_path, task_by_ann)
            new_pred_lookup, _ = load_pred_index(self.new_pred_json_path, task_by_ann)
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            return

        self.task_lookup = task_lookup
        self.task_by_ann = task_by_ann
        self.pred_lookup = pred_lookup
        self.old_pred_lookup = old_pred_lookup
        self.new_pred_lookup = new_pred_lookup
        self.all_images = sorted(set(self.task_lookup.keys()) | set(self.pred_lookup.keys()))
        self.divergent_items = load_divergent_items(self.diff_samples_path, DIFF_TOPK)
        self.all_prompts = ["ALL"] + sorted(pred_prompts | {p for prompts in self.task_lookup.values() for p in prompts})
        self.folders = ["ALL"] + sorted({p.split("/")[1] for p in self.all_images if "/" in p})

        if reset_state:
            self.folder = "ALL"
            self.current_mode = "all"
            self.use_divergent_topk = False
            self.use_current_compare = False
            if self.current_prompt not in self.all_prompts:
                self.current_prompt = "ALL"
        else:
            if self.folder not in self.folders:
                self.folder = "ALL"
            if self.current_prompt not in self.all_prompts:
                self.current_prompt = "ALL"

        print(f"已加载任务图像: {len(self.all_images)} 张")
        print(f"已加载预测: {len(self.pred_lookup)} 张图")
        print(f"类别: {', '.join(self.all_prompts[1:])}")
        if self.divergent_items:
            print(f"已加载分歧样本 Top-{len(self.divergent_items)}: {self.diff_samples_path}")

        self._update_path_labels()
        self._rebuild_dynamic_controls()
        self.refresh_filters()

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
        self.win.bind("<KeyPress-6>", lambda e: self.toggle_divergent_mode())
        self.win.bind("<KeyPress-7>", lambda e: self.toggle_current_compare_mode())
        self.win.bind("<KeyPress-c>", lambda e: self.copy_current_image_path())
        self.win.bind("<KeyPress-C>", lambda e: self.copy_current_image_path())
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

    def toggle_divergent_mode(self):
        if not self.divergent_items and not self.use_divergent_topk:
            return
        self.use_divergent_topk = not self.use_divergent_topk
        if self.use_divergent_topk:
            self.use_current_compare = False
        self.refresh_filters()

    def toggle_current_compare_mode(self):
        if not self.pred_lookup and not self.use_current_compare:
            return
        self.use_current_compare = not self.use_current_compare
        if self.use_current_compare:
            self.use_divergent_topk = False
        self.refresh_filters()

    def refresh_filters(self):
        if self.use_divergent_topk:
            self.images = list(self.divergent_items)
        elif self.use_current_compare:
            self.images = self._build_current_compare_items()
        else:
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
        if hasattr(self, "divergent_button"):
            active = self.use_divergent_topk
            self.divergent_button.config(bg="#8f54d9" if active else "SystemButtonFace", fg="white" if active else "black")
        if hasattr(self, "current_compare_button"):
            active = self.use_current_compare
            self.current_compare_button.config(bg="#b25c2f" if active else "SystemButtonFace", fg="white" if active else "black")

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

    def _build_current_compare_items(self):
        items = []
        for img_path in sorted(self.pred_lookup.keys()):
            if not self._match_folder(img_path):
                continue
            preds = self.pred_lookup.get(img_path, {})
            for prompt, pred in preds.items():
                if self.current_prompt != "ALL" and prompt != self.current_prompt:
                    continue
                if not self._score_band_match(pred):
                    continue
                items.append(
                    {
                        "image_path": img_path,
                        "prompt": prompt,
                        "meta": {"source": "current_json_compare"},
                    }
                )
        return items

    def nav(self, delta):
        if not self.images:
            return
        self.idx = max(0, min(self.idx + delta, len(self.images) - 1))
        self.show()

    def get_current_rel_path(self):
        if not self.images:
            return None
        current = self.images[self.idx]
        if isinstance(current, dict):
            return current["image_path"]
        return current

    def copy_current_image_path(self):
        rel_path = self.get_current_rel_path()
        if not rel_path:
            return
        full_path = str((ROOT / rel_path).resolve())
        try:
            self.win.clipboard_clear()
            self.win.clipboard_append(full_path)
            self.win.update()
            self.win.title(f"已复制路径: {full_path}")
        except Exception as exc:
            messagebox.showerror("复制失败", str(exc))

    def _collect_display_preds(self, rel_path):
        focus_prompt = None
        if (self.use_divergent_topk or self.use_current_compare) and self.images:
            current = self.images[self.idx]
            if isinstance(current, dict):
                focus_prompt = current.get("prompt")
        preds = self.pred_lookup.get(rel_path, {})
        if focus_prompt:
            if focus_prompt in preds:
                return {focus_prompt: preds[focus_prompt]}
            return {}
        if self.current_prompt == "ALL":
            return preds
        if self.current_prompt in preds:
            return {self.current_prompt: preds[self.current_prompt]}
        return {}

    def _collect_divergent_preds(self, rel_path, prompt, use_current_new=False):
        out = {}
        old_pred = self.old_pred_lookup.get(rel_path, {}).get(prompt)
        if use_current_new:
            new_pred = self.pred_lookup.get(rel_path, {}).get(prompt)
        else:
            new_pred = self.new_pred_lookup.get(rel_path, {}).get(prompt)
        if old_pred is not None:
            out["OLD"] = old_pred
        if new_pred is not None:
            out["NEW"] = new_pred
        return out

    def show(self):
        if not self.images:
            self.label.config(image="", text="当前筛选结果为空")
            self.win.title("无图片")
            return

        current = self.images[self.idx]
        if isinstance(current, dict):
            rel_path = current["image_path"]
            focus_prompt = current.get("prompt")
            focus_meta = current.get("meta", {})
        else:
            rel_path = current
            focus_prompt = None
            focus_meta = {}
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

        if self.use_divergent_topk and focus_prompt:
            display_preds = self._collect_divergent_preds(rel_path, focus_prompt, use_current_new=False)
        elif self.use_current_compare and focus_prompt:
            display_preds = self._collect_divergent_preds(rel_path, focus_prompt, use_current_new=True)
        else:
            display_preds = self._collect_display_preds(rel_path)
        hit_lines = []
        miss_lines = []

        for pred_idx, prompt in enumerate(sorted(display_preds.keys())):
            pred = display_preds[prompt]
            hit = bool(pred.get("hit"))
            score = float(pred.get("score", 0.0)) if hit else 0.0
            instances = int(pred.get("instances", 0)) if hit else 0
            if not hit or pred.get("rle") is None:
                miss_lines.append(prompt)
                continue

            mask = rle_to_mask(pred["rle"], pred["rle"]["size"][0], pred["rle"]["size"][1])
            if mask.sum() == 0:
                miss_lines.append(f"{prompt}(empty)")
                continue

            if self.use_divergent_topk or self.use_current_compare:
                color = (255, 80, 80) if prompt == "OLD" else (80, 220, 80)
            else:
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

            line_name = f"{prompt}:{focus_prompt}" if (self.use_divergent_topk or self.use_current_compare) and focus_prompt else prompt
            hit_lines.append(f"{line_name}({instances}x {score:.2f})")

        info_parts = [
            f"[{'分歧Top' if self.use_divergent_topk else ('当前JSON对比' if self.use_current_compare else self.folder)}]",
            f"{self.idx + 1}/{len(self.images)}",
            Path(path).name,
            rel_path,
            f"{w}x{h}",
            f"亮度:{brightness:.0f}",
            f"std:{std_v:.0f}",
            f"lap:{lap_v:.0f}",
            f"prompt:{focus_prompt or self.current_prompt}",
            f"mode:{'divergent' if self.use_divergent_topk else ('current_compare' if self.use_current_compare else self.current_mode)}",
        ]
        if self.use_divergent_topk and focus_meta:
            info_parts.append(
                "分歧: "
                f"old={focus_meta.get('old_hit', '')} "
                f"new={focus_meta.get('new_hit', '')} "
                f"iou={focus_meta.get('iou', '')} "
                f"cstd={focus_meta.get('contrast_std', '')}"
            )
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
            f"[{'分歧Top' if self.use_divergent_topk else ('当前JSON对比' if self.use_current_compare else self.folder)}] {self.idx + 1}/{len(self.images)} "
            f"{Path(path).name} | prompt={focus_prompt or self.current_prompt} "
            f"mode={'divergent' if self.use_divergent_topk else ('current_compare' if self.use_current_compare else self.current_mode)}"
        )
        self.win.title(title)


if __name__ == "__main__":
    Browser()
