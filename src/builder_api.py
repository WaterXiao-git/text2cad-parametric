"""
建模器公共契约（所有 builders_*.py 必须遵守）。

为什么需要统一契约
------------------
官方评测（docs/spec_dataware.txt）对提交模型有硬性门槛：
  * 模型单位统一为毫米(mm)，GLB 为主模型；
  * 每题 constraints 里的 single_component / no_floating_parts /
    stable_base / watertight_required / size_constrained 会被解析检查；
  * FDM 物理合规由主办方解析模型文件直接计算（悬垂、壁厚、连通性等）；
  * 人工复核会判定「生成极简几何体/纯方块骗物理分」为作弊。

因此每个建模器必须同时满足：
  1. 几何正确还原题意（多视图 MLLM 看图打语义分，形状必须像题里说的东西）；
  2. 全部结构化约束达标（单体、无悬浮、稳定底、水密、尺寸）；
  3. FDM 工艺友好（悬垂面积占比尽量低、壁厚达标、可免支撑）；
  4. 形体有细节层次，不能是几个方块拼一下糊弄。

统一约定
--------
  * 单位：毫米
  * 朝向：Z 轴向上，底面落在 Z=0（这就是「默认打印朝向」）
  * 居中：XY 关于原点对称
  * 返回：trimesh.Trimesh，且必须是水密、单连通实体

注册方式
--------
    from builder_api import register, self_check

    @register("hid_con_007")
    def build(task):
        ...
        return mesh
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cadlib import finalize, union                       # noqa: E402
from taskspec import Task, get_task, load_tasks           # noqa: E402
from validate import format_report, validate_task         # noqa: E402

__all__ = ["REGISTRY", "register", "self_check", "Task", "finalize", "union",
           "load_tasks", "get_task", "format_report", "validate_task", "np", "trimesh"]

REGISTRY: dict[str, "callable"] = {}


def register(prompt_id: str):
    """把建模函数按 prompt_id 注册进全局表。"""

    def deco(fn):
        if prompt_id in REGISTRY:
            raise KeyError(f"prompt_id 重复注册: {prompt_id}")
        REGISTRY[prompt_id] = fn
        return fn

    return deco


def self_check(mesh: trimesh.Trimesh, task: Task, verbose: bool = False,
               compute_support: bool = False) -> tuple[bool, str]:
    """
    快速自检：返回 (是否全部强制约束通过, 报告文本)。

    默认关掉支撑体积计算（最耗时），建模迭代阶段先用悬垂面积占比代替；
    最终提交前再用 compute_support=True 做一次全量体检。
    """
    res = validate_task(mesh, task, compute_support=compute_support,
                        compute_thickness=False)
    if verbose:
        return res["all_constraints_passed"], format_report(res)
    return res["all_constraints_passed"], format_report(res)


# ==========================================================================
# 完整示例：hid_con_007 轻载承重挂钩
# --------------------------------------------------------------------------
# 题干：生成一个轻载承重挂钩：背板有两个安装孔，前方挂钩向上弯曲；
#       总厚度不超过20毫米，钩尖圆润，背板不得镂空。
# 要点：
#   * 背板贴墙竖直放置，挂钩朝前(+Y)伸出并向上弯 —— 这就是「默认打印朝向」；
#     但背板平躺打印更稳，所以这里把「贴墙面」做成底面，即背板平放、钩朝上，
#     既满足 stable_base 又天然免支撑。
#   * 总厚度约束指的是墙面法向(原朝向)的厚度，平放后对应 Z 向高度，仍 ≤20mm。
# ==========================================================================
if __name__ == "__main__":  # 示例仅在该文件被直接运行时演示
    import math

    from cadlib import cyl, extrude, loft_solid, rounded_box, rounded_rect, subtract
    from shapely.geometry import Polygon

    def build_hook_example(task):
        W, H, T = 60.0, 90.0, 14.0          # 背板宽、高(平放后为Z向)、总厚度
        plate = rounded_box(W, T, H, r=4.0)

        # 两个安装孔：沿厚度方向贯通背板，平放后孔轴沿 Z
        holes = [cyl(2.6, T * 3, -T, 40, x=x, y=H - 18.0) for x in (-18.0, 18.0)]

        # 向上弯的钩体：用放样做渐变曲率，钩尖收圆
        prof = []
        n = 14
        for i in range(n + 1):
            t = i / n
            a = math.pi * 0.5 * t                       # 0 -> 90°上翘
            reach = 34.0 * math.sin(a)
            z = H + 26.0 * (1 - math.cos(a))
            half = 7.0 - 1.8 * t                        # 越靠钩尖越细
            prof.append((Polygon([(-half, -reach), (half, -reach),
                                  (half, reach), (-half, reach)]), z))
        hook = loft_solid(prof, n=40)

        part = union([plate] + [])
        part = subtract(part, holes)
        part = union([part, hook])
        return finalize(part)

    t = get_task("hid_con_007", "test")
    m = build_hook_example(t)
    ok, rep = self_check(m, t, verbose=True)
    print(rep)
    print("PASS" if ok else "FAIL")
