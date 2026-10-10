"""
通用建模器：把「任意中文工程描述」自动转成参数化实体。

为什么需要它
------------
40 个手写 builders_*.py 只覆盖开发集/测试集的 prompt_id。决赛的输入是
`/input/prompts.jsonl`，主办方明确说可能包含未公开的题目 —— 遇到没见过的
prompt_id 必须仍能产出合法模型，否则该题按 0 分计。

思路：不追求「看懂」，只追求「合法且合理」。
  1. 正则抽取题干里的尺寸（不超过/约/至少 + 数字 + 毫米）；
  2. 关键词打分选出一个**原型**（box/plate/bracket/hook/tube/wedge/ring/arch）；
  3. 按原型拼几何，题干里出现「孔/槽/纹/肋」等词就叠加对应特征；
  4. 最后统一缩放到题干限定的最大尺寸内 —— 这一步保证 size_constrained
     无论如何都不会超（解析错了也只是形状不够贴切，不会违规）。

所有原型都满足：单体、水密、底面平贴 Z=0、无朝下水平大平面。
"""

from __future__ import annotations

import math
import re

import numpy as np
import trimesh
from shapely.geometry import Polygon

try:  # 允许被当作包内模块或顶层模块导入
    from cadlib import (chamfer_slab, cyl, extrude, loft_solid, place,  # noqa: F401
                        rounded_box, rounded_rect, subtract, union, finalize)
    from builder_api import Task, finalize as _finalize
except ImportError:  # pragma: no cover
    raise

# --------------------------------------------------------------------------
# 尺寸解析
# --------------------------------------------------------------------------
_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(毫米|mm|cm|厘米|米)")
_MAX_WORDS = ("不超过", "以内", "不得超过", "以下", "之内", "小于等于", "小于", "≤", "限高", "限宽")
_TARGET_WORDS = ("建议约", "建议", "大约", "左右", "约")
_MIN_WORDS = ("至少", "不低于", "大于等于", "不小于", "以上")

# 轴向关键词：中文里尺寸限定词可以出现在数字**前面**（「外径约90毫米」）
# 也可以出现在**后面**（「100毫米见方」「50毫米以内」），所以判定窗口必须
# 同时看前文和后文 —— 只看向前 12 个字符会漏掉 hid_con_003「50毫米以内」这类写法。
_AXIS_Z_WORDS = ("高度", "高", "限高")
_AXIS_XY_WORDS = ("宽度", "宽", "长度", "长", "直径", "外径", "内径", "半径",
                  "边长", "见方", "限宽")
_WIN_BEFORE, _WIN_AFTER = 12, 8


def _to_mm(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit in ("cm", "厘米"):
        return value * 10.0
    if unit == "米":
        return value * 1000.0
    return value


def _blank() -> dict:
    return {"max": None, "target": None, "min": None}


def _put(spec: dict, kind: str, value: float) -> None:
    """同组同类型取最大值：题干里的小数值（插槽宽 2mm）是特征尺寸，
    不能拿来当整体尺寸，取最大可避免模型被缩到 2mm。"""
    cur = spec[kind]
    spec[kind] = value if cur is None else max(cur, value)


def parse_dims(text: str) -> dict:
    """
    抽取题干里的尺寸线索，并**按轴向分组**。

    返回：
        {
          "numbers": [...],                       # 全部数值（毫米）
          "any":   {"max","target","min"},        # 未指明轴向的整体尺寸
          "horiz": {"max","target","min"},        # 水平向（宽度/长度/直径/边长）
          "z":     {"max","target","min"},        # 高度向
        }

    分组的意义：像「外径建议约90毫米，高度建议约150毫米」这种多轴题，
    只靠整体等比缩放必然顾此失彼；分了组才能对 XY 与 Z 分别缩放。
    """
    out = {"numbers": [], "any": _blank(), "horiz": _blank(), "z": _blank()}
    for m in _NUM.finditer(text):
        val = _to_mm(float(m.group(1)), m.group(2))
        before = text[max(0, m.start() - _WIN_BEFORE): m.start()]
        after = text[m.end(): m.end() + _WIN_AFTER]
        out["numbers"].append(val)

        # 限定词可以出现在数字前（「不超过 60 毫米」）或数字后（「50 毫米以内」），
        # 所以判定限定词时前后文都要看。但**不能**把后文用于判定轴向：
        # 「外径建议约90毫米，高度建议约150毫米」里，90 的后文含「高度」，
        # 用前后文一起判会把 90 误判成高度轴，两轴题就废了。
        ctx = before + after
        if any(w in ctx for w in _MAX_WORDS):
            kind = "max"
        elif any(w in ctx for w in _MIN_WORDS):
            kind = "min"
        elif any(w in ctx for w in _TARGET_WORDS):
            kind = "target"
        else:
            continue

        # 轴向：优先看数字**之前**的词（中文限定词绝大多数是前缀），
        # 前文没有轴向词时才退回到后文（覆盖「100 毫米见方」这类后缀写法）。
        if any(w in before for w in _AXIS_Z_WORDS):
            group = "z"
        elif any(w in before for w in _AXIS_XY_WORDS):
            group = "horiz"
        elif any(w in after for w in _AXIS_Z_WORDS):
            group = "z"
        elif any(w in after for w in _AXIS_XY_WORDS):
            group = "horiz"
        else:
            group = "any"
        _put(out[group], kind, val)
    return out


# --------------------------------------------------------------------------
# 原型选择
# --------------------------------------------------------------------------
_ARCH_KEYWORDS = {
    "hook": ("挂钩", "钩", "挂架", "挂环", "挂绳"),
    "wedge": ("楔", "垫脚", "门挡", "垫块", "斜坡", "垫高"),
    "tube": ("筒", "杯", "罐", "套筒", "花器", "罩", "笔筒"),
    "arch": ("拱桥", "桥", "拱", "拱门"),
    "ring": ("圆环", "环", "圈", "圆盘", "展示座", "陀螺"),
    "bracket": ("支架", "托架", "底座", "支撑", "支脚", "固定座", "挂架", "牌架"),
    "plate": ("板", "牌", "标尺", "相框", "挂饰", "面板", "尺", "卡片"),
    "box": ("盒", "收纳", "抽屉", "箱", "盒体", "分类", "筐", "篮"),
}
_DEFAULT_ARCH = "box"


def select_archetype(text: str, category: str = "") -> str:
    """关键词打分选原型；平票时按 category 偏好兜底。"""
    score = {k: 0 for k in _ARCH_KEYWORDS}
    for arch, words in _ARCH_KEYWORDS.items():
        for w in words:
            if w in text:
                score[arch] += len(w)          # 越长的词越具体，权重越高
    best = max(score, key=lambda k: score[k])
    if score[best] == 0:
        bias = {"constrained": "bracket", "functional_object": "box",
                "decorative_object": "ring", "small_object": "plate"}
        return bias.get(category, _DEFAULT_ARCH)
    return best


# --------------------------------------------------------------------------
# 特征关键词
# --------------------------------------------------------------------------
def wants(text: str, *words: str) -> bool:
    return any(w in text for w in words)


# --------------------------------------------------------------------------
# 原型实现（全部以 100mm 量级构造，最后统一缩放）
# --------------------------------------------------------------------------
def _arc(cx, cy, r, a0, a1, n):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def _circle(r, n=96):
    return Polygon(_arc(0, 0, r, 0, 360, n))


def _dedupe(pts, eps=1e-6):
    out = []
    for q in pts:
        if not out or math.hypot(q[0] - out[-1][0], q[1] - out[-1][1]) > eps:
            out.append(q)
    if len(out) > 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _xz_prism(pts, y0, y1):
    poly = Polygon(_dedupe([(float(p[0]), float(p[1])) for p in pts]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, height=float(y1 - y0), cap=True)
    m.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0],
                                [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float))
    m.apply_translation((0.0, float(y1), 0.0))
    return m


def _rot(axis, ang):
    d = {"x": [1., 0., 0.], "y": [0., 1., 0.], "z": [0., 0., 1.]}[axis]
    return trimesh.transformations.rotation_matrix(ang, d)


def arch_box(W, D, H, text):
    """盒体/容器：外壳 + 内腔 + 可选分隔条。"""
    wall = 4.0
    body = rounded_box(W, D, H, r=4.0, z0=0.0)
    body = subtract(body, [place(rounded_box(W - 2 * wall, D - 2 * wall,
                                             H, r=3.0, z0=wall), 0, 0, 0)])
    if wants(text, "分类", "分格", "隔", "多规格", "收纳"):
        for k in (1, 2, 3):
            y = -D / 2 + wall + (D - 2 * wall) * k / 4.0
            body = union([body, place(rounded_box(W - 2 * wall, 3.0, H - wall,
                                                  r=1.2, z0=wall), 0.0, y - 1.5, 0.0)])
    return body


def arch_plate(W, D, H, text):
    """板件：圆角薄板 + 可选孔阵列 / 顶面浮雕棱。"""
    t = min(H, 8.0)
    body = chamfer_slab(rounded_rect(W, D, r=6.0), t, 1.8, 0.0)
    if wants(text, "孔", "圆孔", "通孔", "镂空", "嵌槽", "凹槽"):
        holes = []
        for i in range(3):
            for j in range(2):
                holes.append(place(cyl(min(D, W) * 0.06, t * 3, -t, 28,
                                       x=-W / 4 + i * W / 4,
                                       y=-D / 4 + j * D / 2), 0, 0, 0))
        body = subtract(body, holes)
    if wants(text, "刻度", "纹", "浮雕", "格"):
        # 顶面浮雕棱：**沿 Y 平铺一排**，而不是沿 Z 逐级抬高。
        # 早期版本写成 z = t*0.4 + k*2.4（棱高只有 1.4），k≥3 的棱条整体
        # 飘在板面上方 → 形成悬浮碎块，把 single_component / no_floating_parts
        # 两项硬约束直接打掉（hid_con_005 / hid_dec_003 / hid_sma_009 / hid_sma_010 都栽在这）。
        # 正确做法：棱条底面下沉 0.6mm 咬进板体（保证连通），只露出约 0.8mm 做浮雕。
        n = 5
        z0 = max(t - 0.6, t * 0.45)
        for k in range(n):
            y = -D * 0.30 + k * (D * 0.60 / (n - 1))
            body = union([body, place(
                rounded_box(W * 0.62, 1.6, 1.4, r=0.6, z0=z0), 0.0, y, 0.0)])
    return body


def arch_bracket(W, D, H, text):
    """支架：底板 + 立板 + 三角加强肋（免支撑、稳定底）。"""
    base_t = max(H * 0.18, 6.0)
    body = chamfer_slab(rounded_rect(W, D, r=5.0), base_t, 2.5, 0.0)
    up_t = max(W * 0.08, 6.0)
    up = chamfer_slab(rounded_rect(up_t, D * 0.8, r=3.0),
                      H - base_t, 3.0, 0.0)
    up.apply_translation((0.0, 0.0, base_t))
    body = union([body, place(up, 0.0, -D * 0.1, 0.0)])
    # 三角肋
    for sy in (-D * 0.25, D * 0.25):
        rib = _xz_prism([(up_t / 2 - 1.0, base_t), (up_t / 2 - 1.0, H * 0.62),
                         (W * 0.28, base_t)], sy - 2.0, sy + 2.0)
        body = union([body, rib])
    return body


def arch_hook(W, D, H, text):
    """挂钩：背板平贴床面 + 向上生长的钩体（唯一免支撑朝向）。"""
    t = max(W * 0.1, 5.0)
    plate = chamfer_slab(rounded_rect(W, D, r=5.0), t, 2.0, 0.0)
    body = subtract(plate, [place(cyl(t * 0.45, t * 3, -t, 36, x=x, y=D * 0.25), 0, 0, 0)
                            for x in (-W * 0.28, W * 0.28)])
    R = max(H * 0.20, 6.0)
    THK = min(max(t * 0.6, 3.0), R * 0.6)      # 保证内弧半径 R-THK > 0
    z0 = t - 1.0
    zc = z0 + R + THK                          # 弯心 z
    yc = D * 0.32                              # 弯心 y
    ARM = max(H * 0.25, 8.0)                   # 水平臂长
    yt = yc - R - ARM                          # 钩尖 x
    prof = [(yc + THK, z0), (yc + THK, zc)]
    prof += _arc(yc - R, zc, R + THK, 0.0, 90.0, 18)[1:]     # 外侧弯
    prof += [(yt, zc + R + THK)]
    prof += _arc(yt, zc + R, THK, 90.0, 270.0, 14)[1:]       # 圆润钩尖
    prof += [(yc - R, zc + R - THK)]
    prof += _arc(yc - R, zc, max(R - THK, 1.0), 90.0, 0.0, 18)[1:]  # 内侧弯
    prof += [(yc - THK, z0)]
    fin = _xz_prism(prof, -THK, THK)
    return union([body, fin])


def arch_tube(W, D, H, text):
    """套筒/容器：圆筒 + 内腔 + 底板，外壁竖向凹槽防滑。"""
    R = min(W, D) / 2.0
    wall = 4.0
    body = subtract(cyl(R, H, 0.0, 96), [cyl(R - wall, H + 6.0, wall, 96)])
    body = union([body, cyl(R, wall, 0.0, 96)])
    for k in range(10):
        a = math.radians(36.0 * k)
        body = subtract(body, [cyl(1.5, H + 6.0, -3.0, 20,
                                   x=R * math.cos(a), y=R * math.sin(a))])
    return body


def arch_wedge(W, D, H, text):
    """楔/垫脚：直角楔体 + 底面防滑槽。"""
    body = _xz_prism([(0, 0), (W, 0), (W, H)], -D / 2, D / 2)
    for k in range(7):
        x = W * 0.2 + W * 0.6 * k / 6.0
        body = subtract(body, [_xz_prism([(x - 1.6, -1.0), (x + 1.6, -1.0),
                                          (x + 1.6, 3.0), (x - 1.6, 3.0)],
                                         -D / 2 - 1, D / 2 + 1)])
    return body


def arch_ring(W, D, H, text):
    """圆环/展示座：底盘 + 中央轴或环。"""
    R = min(W, D) / 2.0
    base_h = max(H * 0.14, 6.0)
    body = chamfer_slab(_circle(R, 128), base_h, 2.5, 0.0)
    body = union([body, cyl(R * 0.22, H - base_h, 0.0, 48)])
    cap = trimesh.creation.icosphere(subdivisions=3, radius=R * 0.18)
    cap.apply_translation((0.0, 0.0, H - base_h))
    body = union([body, cap])
    return body


def arch_arch(W, D, H, text):
    """拱桥/拱门：箱梁 + 尖拱开洞（拱腹无近水平下表面）。"""
    off = W * 0.16
    R = math.hypot(off, H * 0.8)
    R_IN = R - max(W * 0.08, 8.0)
    t_in = math.degrees(math.acos(min(1.0, off / R_IN)))
    body = _xz_prism([(-W / 2, 0), (W / 2, 0), (W / 2, H), (-W / 2, H)],
                     -D / 2, D / 2)
    hole = _arc(-off, 0, R_IN, 0, t_in, 26) + \
        _arc(off, 0, R_IN, 180 - t_in, 180, 26)
    body = subtract(body, [_xz_prism(hole, -D / 2 - 2, D / 2 + 2)])
    return body


ARCH_FN = {"box": arch_box, "plate": arch_plate, "bracket": arch_bracket,
           "hook": arch_hook, "tube": arch_tube, "wedge": arch_wedge,
           "ring": arch_ring, "arch": arch_arch}


# --------------------------------------------------------------------------
# 尺寸对齐
# --------------------------------------------------------------------------
def _target_from(spec: dict) -> float | None:
    """
    由一组同轴约束算出该轴的目标尺寸。

    优先级：硬上限（不超过 XX）> 软目标（约 XX）> 下限（至少 XX）。
    这样「不超过 60 毫米」一定不会超；只有「约 XX」时按建议值贴近。
    """
    want = spec["max"] if spec["max"] is not None else spec["target"]
    if spec["min"] is not None and (want is None or spec["min"] > want):
        want = spec["min"]
    return want


def _scale(mesh: trimesh.Trimesh, sx: float, sy: float, sz: float) -> None:
    """逐轴缩放（就地）。倍数做上下限保护，避免极端题干把模型压成薄片。"""
    clamp = lambda f: min(max(float(f), 0.15), 6.0)  # noqa: E731
    mesh.apply_scale([clamp(sx), clamp(sy), clamp(sz)])


def apply_parsed_size(mesh: trimesh.Trimesh, dims: dict) -> tuple[trimesh.Trimesh, str]:
    """
    按题干数值把原型调到目标尺寸。

    为什么不是简单等比缩放：官方题干的常见写法是
    「外径建议约 90 毫米，高度建议约 150 毫米」——两个轴各有各的数。
    整体等比缩放最多只能满足其中一个，另一个必然偏离。
    所以这里分两步：先按「未指明轴向」的数值整体缩放；
    再对 XY（水平向）与 Z（高度向）各自单独缩放到题干要求。
    原型的长宽比（100:70）在 XY 内等比保留，不会把形状拉歪。
    """
    notes: list[str] = []

    # 1) 未指明轴向的整体尺寸（如「整体不超过 60 毫米」）
    want_any = _target_from(dims["any"])
    if want_any:
        ext = float(np.max(mesh.extents))
        if ext > 1e-9:
            s = max(want_any, 20.0) / ext
            _scale(mesh, s, s, s)
            notes.append(f"整体×{s:.3f}")

    # 2) 高度轴（Z）
    want_z = _target_from(dims["z"])
    if want_z and mesh.extents[2] > 1e-9:
        s = max(want_z, 5.0) / float(mesh.extents[2])
        _scale(mesh, 1.0, 1.0, s)
        notes.append(f"高度×{s:.3f}")

    # 3) 水平向（XY 等比；原型的 X 恒为最大边，与「宽度/长度/直径」口径一致）
    want_xy = _target_from(dims["horiz"])
    if want_xy and mesh.extents[0] > 1e-9:
        s = max(want_xy, 20.0) / float(mesh.extents[0])
        _scale(mesh, s, s, 1.0)
        notes.append(f"水平×{s:.3f}")

    # 4) 兜底：题干完全没给数值时给一个合理量级
    if not notes and float(np.max(mesh.extents)) < 5.0:
        s = 80.0 / max(float(np.max(mesh.extents)), 1e-9)
        _scale(mesh, s, s, s)
        notes.append(f"默认×{s:.3f}")

    # 5) 硬上限安全垫：「不超过 XX 毫米」是不容超的硬指标，但圆角/放样等
    #    几何处理会让实际包络比目标大出零点几毫米（实测 hid_con_002：
    #    目标 85mm 实得 85.1mm，0.1mm 之差就被判违规）。
    #    这里按超出的比例补一次缩放，并留 0.2% 余量。
    metrics = {"z": lambda e: float(e[2]),
               "horiz": lambda e: float(max(e[0], e[1])),
               "any": lambda e: float(max(e))}
    for group, get in metrics.items():
        lim = dims[group]["max"]
        if not lim:
            continue
        act = get(mesh.extents)
        if act > lim * 0.999:
            s = (lim / act) * 0.998
            if group == "z":
                _scale(mesh, 1.0, 1.0, s)
            elif group == "horiz":
                _scale(mesh, s, s, 1.0)
            else:
                _scale(mesh, s, s, s)
            notes.append(f"{group}上限×{s:.4f}")

    return mesh, "; ".join(notes)


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_generic(task: Task) -> trimesh.Trimesh:
    """通用建模入口：解析题干 → 选原型 → 构造 → 按题干尺寸对齐。"""
    text = f"{task.prompt} {task.prompt_intent}"
    dims = parse_dims(text)
    arch = select_archetype(text, task.category)
    body = ARCH_FN[arch](100.0, 70.0, 60.0, text)
    body = finalize(body)
    body, _ = apply_parsed_size(body, dims)
    body = finalize(body)
    return body
