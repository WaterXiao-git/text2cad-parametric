"""
赛题任务规格：把官方 prompt.json 与从中文题干里解析出的尺寸约束合并成结构化任务。

为什么需要手工解析尺寸：
    官方 prompt.json 只有布尔标签 size_constrained，
    但真正的数值（如「整体高度不超过90毫米」）写在中文 prompt 正文里。
    要拿满 FDM 物理合规分，必须逐题把数值抠出来并强制满足。
    这里由人工核对 40 道题干后录入，作为建模与校验的硬指标。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# 从中文题干解析出的目标尺寸（单位 mm）。
# max_* 表示硬上限；target_* 表示「建议约」的软目标（允许 ±12% 容差）。
SIZE_HINTS: dict[str, dict[str, float]] = {
    # ---------- constrained ----------
    "hid_con_001": {"max_height": 90},
    "hid_con_002": {"max_diameter": 85},
    "hid_con_003": {"max_extent": 50},
    "hid_con_004": {"max_extent": 60},
    "hid_con_005": {"max_width": 100, "max_depth": 100},
    "hid_con_006": {"max_length": 140},
    "hid_con_007": {"max_thickness": 20},
    "hid_con_008": {"max_width": 120},
    "hid_con_009": {"max_height": 60},
    "hid_con_010": {"max_width": 150},
    # ---------- decorative ----------
    "hid_dec_001": {"target_width": 120},
    "hid_dec_002": {"target_diameter": 90},
    "hid_dec_003": {"target_width": 180},
    "hid_dec_004": {"target_diameter": 90, "target_height": 150},
    "hid_dec_005": {"target_width": 110},
    "hid_dec_006": {"target_width": 120},
    "hid_dec_007": {"target_height": 110},
    "hid_dec_008": {"target_height": 130},
    "hid_dec_009": {"target_width": 100},
    "hid_dec_010": {"target_height": 120},
    # ---------- functional ----------
    "hid_fun_001": {"target_length": 60, "count": 2},
    "hid_fun_002": {"target_width": 90},
    "hid_fun_003": {"target_width": 80},
    "hid_fun_004": {"target_width": 180},
    "hid_fun_005": {"max_width": 150},
    "hid_fun_006": {"target_height": 150},
    "hid_fun_007": {"target_length": 100},
    "hid_fun_008": {"target_width": 100},
    "hid_fun_009": {"max_width": 80},
    "hid_fun_010": {"max_width": 120},
    # ---------- small_object ----------
    "hid_sma_001": {"max_extent": 70},
    "hid_sma_002": {"target_height": 45},
    "hid_sma_003": {"target_length": 80},
    "hid_sma_004": {"target_diameter": 60},
    "hid_sma_005": {"max_diameter": 90},
    "hid_sma_006": {"target_width": 70},
    "hid_sma_007": {"target_length": 100},
    "hid_sma_008": {"target_width": 80},
    "hid_sma_009": {"target_length": 70},
    "hid_sma_010": {"max_height": 130},
}


@dataclass
class Task:
    """一道赛题的完整规格。"""

    prompt_id: str
    category: str
    prompt: str
    prompt_en: str
    prompt_intent: str
    constraints: dict = field(default_factory=dict)
    size: dict = field(default_factory=dict)

    @property
    def needs_single_component(self) -> bool:
        return bool(self.constraints.get("single_component"))

    @property
    def needs_no_floating(self) -> bool:
        return bool(self.constraints.get("no_floating_parts"))

    @property
    def needs_stable_base(self) -> bool:
        return bool(self.constraints.get("stable_base"))

    @property
    def needs_watertight(self) -> bool:
        return bool(self.constraints.get("watertight_required"))

    @property
    def needs_size(self) -> bool:
        return bool(self.constraints.get("size_constrained")) or bool(self.size)


def load_tasks(subset: str = "test") -> list[Task]:
    """读取 data/prompts_{subset}.json 并合并尺寸提示。"""
    path = DATA / f"prompts_{subset}.json"
    if not path.exists():
        raise FileNotFoundError(f"缺少 {path}，请先运行 src/extract_data.py")
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for r in raw:
        tasks.append(
            Task(
                prompt_id=r["prompt_id"],
                category=r.get("category", ""),
                prompt=r.get("prompt", ""),
                prompt_en=r.get("prompt_en", ""),
                prompt_intent=r.get("prompt_intent", ""),
                constraints=r.get("constraints", {}),
                size=SIZE_HINTS.get(r["prompt_id"], {}),
            )
        )
    return tasks


def get_task(prompt_id: str, subset: str = "test") -> Optional[Task]:
    for t in load_tasks(subset):
        if t.prompt_id == prompt_id:
            return t
    return None


if __name__ == "__main__":
    ts = load_tasks("test")
    print(f"共 {len(ts)} 道题")
    for t in ts[:3]:
        print(f"\n[{t.prompt_id}] {t.category}  size={t.size}")
        print(f"  {t.prompt}")
        print(f"  constraints={t.constraints}")
