"""
按官方 constraints 逐项校验生成的网格，并给出可执行的整改建议。

官方 5 项结构化约束（见 docs/spec_dataware.txt）：
    single_component    单体
    no_floating_parts   无悬浮件
    stable_base         稳定底部
    watertight_required 需水密
    size_constrained    尺寸受限

本模块把这 5 项 + FDM 工艺指标（悬垂、壁厚、支撑）统一成一份体检报告，
用于「生成 → 自检 → 修正」闭环，也是技术报告里量化效果的数据来源。
"""

from __future__ import annotations

import numpy as np
import trimesh

from dataclasses import dataclass, field
from typing import Optional

from printability import (
    PrintConfig,
    bed_contact,
    overhang_area_ratio,
    printability_metrics,
    printability_score,
)

__all__ = ["CheckResult", "validate_task", "format_report", "SOFT_TOL"]


# 「建议约 XX 毫米」类软目标允许的上下浮动比例
SOFT_TOL = 0.12

# 打印就位（print-in-place）机构题：题干明确要求「留可见间隙」的活动件，
# 因此必然存在多个连通体。判定标准改为「附加件的包围盒必须完全嵌套在
# 主体包围盒内」——嵌套即为机构，散落在外的才算悬浮碎块。
PRINT_IN_PLACE = {"hid_con_008", "hid_fun_009", "hid_sma_001", "hid_sma_005"}


@dataclass
class CheckResult:
    name: str
    required: bool
    passed: bool
    detail: str = ""
    value: Optional[float] = None


def _check_size(task_size: dict, extents: np.ndarray, required: bool = True) -> CheckResult:
    """
    校验尺寸约束。

    max_*   ：硬上限，超出即不通过
    target_*：软目标，允许 ±SOFT_TOL

    required 取自官方 prompt.json 的 `size_constrained` 标签，**不是**取自
    「我们是否录了尺寸」。原因：装饰类题目的官方标签常常是 false，题干里也
    根本没有数字（例如「石拱桥微缩摆件，单孔石拱跨越两侧台座」），
    taskspec.SIZE_HINTS 里那些 target_* 是团队自己定的合理量级。
    把自定的量级当成官方硬约束来判「失败」，会虚报违规——
    这类题正确姿势是「参考」项：仍然量、仍然报，但不计入官方约束通过率。
    """
    if not task_size:
        return CheckResult("size_constrained", False, True, "无尺寸约束")

    w, d, h = float(extents[0]), float(extents[1]), float(extents[2])
    # 回转体直径取 XY 包络的最大边，而不是包围盒对角线：
    # 对角线会把圆柱/六棱柱的直径高估 20%~40%，与题干「外径约90」严重不符
    diag = float(max(w, d))
    msgs, ok = [], True

    hard = {
        "max_height": h, "max_width": w, "max_depth": d,
        "max_length": max(w, d), "max_thickness": min(w, d),
        "max_extent": max(w, d, h), "max_diameter": diag,
    }
    for key, actual in hard.items():
        if key in task_size:
            lim = float(task_size[key])
            passed = actual <= lim + 1e-6
            ok &= passed
            msgs.append(f"{key}: {actual:.1f}/{lim:.0f}mm {'OK' if passed else '超限'}")

    soft = {
        "target_height": h, "target_width": w, "target_depth": d,
        "target_length": max(w, d), "target_extent": max(w, d, h),
        "target_diameter": diag,
    }
    for key, actual in soft.items():
        if key in task_size:
            lim = float(task_size[key])
            lo, hi = lim * (1 - SOFT_TOL), lim * (1 + SOFT_TOL)
            passed = lo <= actual <= hi
            ok &= passed
            msgs.append(f"{key}: {actual:.1f}≈{lim:.0f}mm {'OK' if passed else '偏离'}")

    if not msgs:
        return CheckResult("size_constrained", False, True, "无可比尺寸")
    detail = "; ".join(msgs)
    if not required:
        detail += "（官方未强制尺寸，按参考项记录）"
    return CheckResult("size_constrained", bool(required), bool(ok), detail)


def validate_task(
    mesh: trimesh.Trimesh,
    task,
    cfg: PrintConfig = PrintConfig(),
    compute_support: bool = True,
    compute_thickness: bool = True,
) -> dict:
    """
    对单个任务的输出网格做全量体检。

    返回包含 checks（逐项约束）、printability（工艺指标）、
    score（综合分）与 suggestions（整改建议）的字典。
    """
    checks: list[CheckResult] = []
    suggestions: list[str] = []

    # ---- 连通性：单体 & 无悬浮件 -------------------------------------
    comps = mesh.split(only_watertight=False)
    n_comp = len(comps)
    largest = max((len(c.faces) for c in comps), default=0)
    largest_frac = largest / max(len(mesh.faces), 1)

    if task.needs_single_component:
        ok = n_comp == 1
        checks.append(CheckResult("single_component", True, ok,
                                  f"连通体数量={n_comp}", float(n_comp)))
        if not ok:
            suggestions.append(
                f"存在 {n_comp} 个连通体：用布尔并集合并，或在主体与附件间增加连接颈/加强筋。"
            )

    if task.needs_no_floating:
        if task.prompt_id in PRINT_IN_PLACE and n_comp > 1:
            # 以包围盒体积最大者作为「主体」参考（活动件面数可能更多但体量小）
            order = sorted(comps, key=lambda c: -float(np.prod(c.bounds[1] - c.bounds[0])))
            mb = order[0].bounds
            # 装配判定：附加体与主体在 XY 投影上必须重叠（叠在一起才算
            # 装配到位的机构；XY 都不重叠的就是散落在外的悬浮碎块）
            nested = True
            for c in order[1:]:
                cb, ob = c.bounds, order[0].bounds
                overlap = (min(cb[1][0], ob[1][0]) - max(cb[0][0], ob[0][0]) > 0.5
                           and min(cb[1][1], ob[1][1]) - max(cb[0][1], ob[0][1]) > 0.5)
                if not overlap:
                    nested = False
                    break
            checks.append(CheckResult(
                "no_floating_parts", True, nested,
                f"打印就位机构: {n_comp} 个体, XY投影相互重叠={nested}",
                float(largest_frac)))
            if not nested:
                suggestions.append("打印就位机构的附加体与主体在 XY 投影上不重叠，"
                                   "疑似悬浮碎块，请检查装配关系。")
        else:
            # 悬浮件 = 除了最大主体之外的零碎连通体
            ok = n_comp == 1 or largest_frac >= 0.98
            checks.append(CheckResult("no_floating_parts", True, ok,
                                      f"主体占比={largest_frac:.3f}, 碎块={n_comp - 1}",
                                      float(largest_frac)))
            if not ok:
                suggestions.append(
                    f"有 {n_comp - 1} 个碎块（主体仅占 {largest_frac:.1%}）："
                    "删除独立碎块或把它们与主体连通。"
                )

    # ---- 稳定底部 ------------------------------------------------------
    bc = bed_contact(mesh, (0.0, 0.0, 1.0))
    ca, ha = bc["contact_area_mm2"], bc["hull_area_mm2"]
    ratio = ca / ha if ha > 1e-9 else 0.0
    if task.needs_stable_base:
        ok = bool(bc["com_inside"]) and ratio >= 0.15 and ca >= 100.0
        checks.append(CheckResult(
            "stable_base", True, ok,
            f"接触面={ca:.0f}mm² 支撑凸包={ha:.0f}mm² 占比={ratio:.2f} "
            f"质心在支撑区内={bc['com_inside']}", float(ratio)))
        if not ok:
            suggestions.append(
                f"底部不稳（接触 {ca:.0f}mm²，占比 {ratio:.2f}）："
                "加宽/加平底座，或切出一个平面底面，确保质心投影落在支撑区内。"
            )

    # ---- 水密性 --------------------------------------------------------
    wt = bool(mesh.is_watertight)
    if task.needs_watertight:
        checks.append(CheckResult("watertight_required", True, wt,
                                  f"watertight={wt}", 1.0 if wt else 0.0))
        if not wt:
            suggestions.append("网格非水密：执行补洞/非流形修复，或改用泊松重建兜底。")
    else:
        checks.append(CheckResult("watertight_required", False, True,
                                  f"未强制（当前 watertight={wt}）", 1.0 if wt else 0.0))

    # ---- 尺寸 ----------------------------------------------------------
    # required 用官方的 size_constrained 标签，而不是「我们录没录尺寸」：
    # 自定的量级只能当参考项，不能当官方硬约束（详见 _check_size 的说明）。
    sz = _check_size(task.size, np.asarray(mesh.extents, dtype=float),
                     required=bool(task.constraints.get("size_constrained")))
    checks.append(sz)
    if sz.required and not sz.passed:
        suggestions.append(f"尺寸不达标（{sz.detail}）：按比例整体缩放后再复检。")

    # ---- 工艺指标 ------------------------------------------------------
    pm = printability_metrics(
        mesh, cfg=cfg, compute_support=compute_support,
        compute_thickness=compute_thickness,
    )
    pscore, pparts = printability_score(pm, cfg)

    ohr = pm["overhang_area_ratio"]
    if ohr > 0.10:
        suggestions.append(
            f"悬垂面积占比 {ohr:.1%} 偏高：把朝下的水平面改成 ≤45° 的斜/锥面（自支撑），"
            "或加过渡圆角，可免支撑打印。"
        )
    tf = pm.get("thin_fraction", 0.0)
    if tf > 0.10:
        suggestions.append(
            f"壁厚过薄样本占 {tf:.1%}：加厚薄壁到 ≥{cfg.min_wall_mm:.1f}mm（约 2 倍喷嘴直径）。"
        )

    passed_req = [c.passed for c in checks if c.required]
    return {
        "prompt_id": task.prompt_id,
        "category": task.category,
        "checks": checks,
        "n_required": len(passed_req),
        "n_passed": int(sum(passed_req)),
        "all_constraints_passed": bool(all(passed_req)) if passed_req else True,
        "printability": pm,
        "printability_score": float(pscore),
        "printability_parts": pparts,
        "suggestions": suggestions,
    }


def format_report(res: dict) -> str:
    """把体检结果渲染成可读文本，供日志与技术报告使用。"""
    lines = [f"[{res['prompt_id']}] {res['category']}"]
    lines.append(
        f"  约束: {res['n_passed']}/{res['n_required']} 通过"
        f"   可打印性综合分: {res['printability_score']:.3f}"
    )
    for c in res["checks"]:
        flag = "强制" if c.required else "参考"
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"    [{mark}] ({flag}) {c.name}: {c.detail}")
    pm = res["printability"]
    lines.append(
        f"  工艺: 悬垂={pm['overhang_area_ratio']:.3f}"
        f"  连通体={pm['n_components']}"
        f"  水密={pm['watertight']}"
        f"  薄壁占比={pm.get('thin_fraction', float('nan')):.3f}"
        f"  尺寸={pm['extents_mm'][0]:.1f}x{pm['extents_mm'][1]:.1f}x{pm['extents_mm'][2]:.1f}mm"
    )
    for s in res["suggestions"]:
        lines.append(f"  -> 建议: {s}")
    return "\n".join(lines)
