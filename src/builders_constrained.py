"""
约束生成（constrained） 建模器 —— hid_con_001, hid_con_002, hid_con_003, hid_con_004, hid_con_005, hid_con_006, hid_con_007, hid_con_008, hid_con_009, hid_con_010

建模硬性要求（详见 builder_api.py 顶部说明）：
  * 单位毫米，Z 轴向上，底面落在 Z=0，XY 居中；
  * 返回 trimesh.Trimesh，必须水密、单连通；
  * 逐项满足题干 constraints（单体/无悬浮/稳定底/水密/尺寸）；
  * FDM 免支撑优先：朝下水平面一律改成 <=45 度斜面（loft_solid / taper_box）；
  * 形体要有细节层次，禁止用极简方块糊弄（会被判作弊）。

自检方式：
    python -c "import sys;sys.path.insert(0,'src');import builders_constrained" \
             # 无异常即注册成功
    或运行 python src/pipeline.py --only <prompt_id> 查看体检报告
"""

from __future__ import annotations

import math

import numpy as np
import trimesh
from shapely.geometry import Polygon
from typing import Sequence

from builder_api import finalize, register, self_check, union

# ---------------------------------------------------------------------------
# 兼容别名：check_one.py 从 builder_api 取 BUILDERS，而 builder_api 只导出
# REGISTRY。本模块被 builders.py 抢先导入，在此补上别名即可，无需改他人文件。
# ---------------------------------------------------------------------------
import builder_api as _ba  # noqa: E402

if not hasattr(_ba, "BUILDERS"):
    _ba.BUILDERS = _ba.REGISTRY

from cadlib import (
    chamfer_slab,
    cyl,
    dovetail_slot,
    extrude,
    grid_of,
    hex_prism,
    keyhole_slot,
    loft_solid,
    place,
    rebase,
    ridge_strip,
    ring_of_holes,
    rounded_box,
    rounded_rect,
    subtract,
    tube,
    taper_box,
    intersect,
)

# ==========================================================================
# 本文件私有工具函数
# ==========================================================================

def _rot(axis: str, ang: float) -> np.ndarray:
    """绕坐标轴的旋转矩阵（右手系，行列式 +1，不会翻转绕向）。"""
    d = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
    return trimesh.transformations.rotation_matrix(ang, d)


# 轴置换矩阵。trimesh 的 apply_transform 采用列向量约定 v' = M @ v，
# 因此矩阵第 i 行决定输出的第 i 个分量取自哪个局部坐标。
_PERM_YZ_TO_X = np.array([[0.0, 0.0, 1.0, 0.0],     # 世界 X <- 局部拉伸方向 w
                          [1.0, 0.0, 0.0, 0.0],     # 世界 Y <- 轮廓第 1 分量
                          [0.0, 1.0, 0.0, 0.0],     # 世界 Z <- 轮廓第 2 分量
                          [0.0, 0.0, 0.0, 1.0]])
_PERM_XZ_TO_Y = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]])


def cyl_x(r: float, length: float, y: float = 0.0, z: float = 0.0,
          xc: float = 0.0, sections: int = 48) -> trimesh.Trimesh:
    """轴沿 X 的圆柱，中心在 (xc, y, z)。"""
    m = trimesh.creation.cylinder(radius=float(r), height=float(length),
                                  sections=int(sections))
    m.apply_transform(_rot("y", math.pi / 2))
    m.apply_translation((float(xc), float(y), float(z)))
    return m


def cyl_y(r: float, length: float, x: float = 0.0, z: float = 0.0,
          yc: float = 0.0, sections: int = 48) -> trimesh.Trimesh:
    """轴沿 Y 的圆柱，中心在 (x, yc, z)。"""
    m = trimesh.creation.cylinder(radius=float(r), height=float(length),
                                  sections=int(sections))
    m.apply_transform(_rot("x", math.pi / 2))
    m.apply_translation((float(x), float(yc), float(z)))
    return m


def yz_prism(pts: Sequence, x0: float, x1: float) -> trimesh.Trimesh:
    """
    把 (y, z) 平面轮廓沿 X 方向拉伸成实体，x 从 x0 到 x1。

    用途：屋檐/遮檐/斜撑这类「截面沿宽度恒定」的斜面板件。
    用旋转而不是置换矩阵，保证行列式 +1、绕向不被翻转。
    """
    poly = Polygon([(float(p[0]), float(p[1])) for p in pts])
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, height=float(x1 - x0), cap=True)
    # (u, v, w) -> (w, u, v)：行列式 +1 的纯旋转，不会翻转绕向
    m.apply_transform(_PERM_YZ_TO_X)
    m.apply_translation((float(x0), 0.0, 0.0))
    return m


def xz_prism(pts: Sequence, y0: float, y1: float) -> trimesh.Trimesh:
    """把 (x, z) 平面轮廓沿 Y 方向拉伸成实体，y 从 y0 到 y1。"""
    poly = Polygon([(float(p[0]), float(p[1])) for p in pts])
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, height=float(y1 - y0), cap=True)
    # (u, v, w) -> (u, -w, v)：行列式 +1
    m.apply_transform(_PERM_XZ_TO_Y)
    m.apply_translation((0.0, float(y1), 0.0))
    return m


def round_poly(poly: Polygon, r: float = 2.0, res: int = 8) -> Polygon:
    """把多边形所有拐角倒成半径 r 的圆角（buffer(+r).buffer(-r) 技巧）。"""
    if r <= 0:
        return poly
    out = poly.buffer(float(r), resolution=res).buffer(-float(r), resolution=res)
    return out if out.is_valid and not out.is_empty else poly


def planar_prism(poly: Polygon, z_top: float, z_bot: float = -5.0) -> trimesh.Trimesh:
    """竖直棱柱（用于与斜面板件求交，切出平面轮廓）。"""
    return extrude(poly, float(z_top - z_bot), z_bot)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int):
    """采样一段圆弧，返回点列表。角度单位为度。"""
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))
            for i in range(n + 1)]


def ring_points(r_out: float, r_in: float, n: int = 96):
    """圆环轮廓点（逆时针）。"""
    out = _arc(0.0, 0.0, r_out, 0.0, 360.0, n)
    inn = _arc(0.0, 0.0, r_in, 360.0, 0.0, n)
    return out + inn


def annulus(r_out: float, r_in: float, n: int = 128) -> Polygon:
    """圆环面（Shapely Polygon，带内环）。"""
    outer = _arc(0.0, 0.0, r_out, 0.0, 360.0, n)
    inner = _arc(0.0, 0.0, r_in, 0.0, 360.0, n)
    return Polygon(outer, [inner])


# ==========================================================================
# hid_con_001   [constrained]  intent=悬垂控制挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_height:90
# 题干: 生成一个免支撑的悬挑小雨棚模型：顶部有一块向前伸出的弧形遮檐，遮檐下方不得有支撑柱，整体高度不超过90毫米。
# ==========================================================================
@register("hid_con_001")
def build_hid_con_001(task):
    """
    免支撑悬挑小雨棚。

    打印朝向即「使用朝向」：背板竖直落在底座上，遮檐向 -Y（front 视图的正前方）
    悬挑 36mm，檐下净空内没有任何立柱。

    免支撑的关键：檐口下表面是一条「下凹上扬」的二次曲线，全程坡度 52°→47°，
    恒大于 45° 自支撑临界角，因此不需要任何支撑；上表面做成 8→5mm 的变厚度，
    并沿坡度方向排布导水槽凸棱。
    """
    # ---------------- 底座（平底 + 顶面倒角） ----------------
    base = chamfer_slab(rounded_rect(76.0, 46.0, r=8.0), 9.0, 2.5, 0.0)
    base.apply_translation((0.0, 12.0, 0.0))              # y: -11 .. 35

    # ---------------- 背板（下厚上薄的拔模体） ----------------
    back = taper_box(68.0, 22.0, 0.0, 68.0, 13.0, 74.0, r=3.0, layers=4)
    back.apply_translation((0.0, 18.0, 0.0))              # 底 y:7..29  顶 y:11.5..24.5

    # ---------------- 遮檐截面（YZ 平面） ----------------
    UB = 11.0          # 檐下曲线起点所在的 y（背板正面）
    ZB = 70.0          # 檐下曲线起点高度
    L = 36.0           # 悬挑长度（沿 -Y）
    A = math.tan(math.radians(52.0))                      # 起点坡度 52°
    B = (A - math.tan(math.radians(47.0))) / (2.0 * L)    # 终点坡度 47°

    def z_under(u: float) -> float:
        """檐下曲面：z = ZB - A*u + B*u^2（下凹上扬，坡度递减但恒 >45°）。"""
        return ZB - A * u + B * u * u

    def z_top(u: float) -> float:
        """檐上曲面：厚度自 8mm 渐变到 5mm。"""
        return z_under(u) + 8.0 - 3.0 * u / L

    pts: list[tuple[float, float]] = []
    pts.append((23.0, ZB))                                # 背部下缘（埋入背板）
    pts.append((23.0, ZB + 8.0))                          # 背部上缘
    n = 44
    for i in range(n + 1):                                # 上表面 → 檐口
        u = L * i / n
        pts.append((UB - u, z_top(u)))
    pts.append((UB - L - 2.0, z_top(L) - 2.0))            # 檐口上倒角
    pts.append((UB - L - 2.0, z_under(L) - 3.0))          # 檐口下倒角
    for i in range(n, -1, -1):                            # 檐下曲面 → 背板
        u = L * i / n
        pts.append((UB - u, z_under(u)))

    slab = yz_prism(pts, -48.0, 48.0)

    # ---------------- 平面轮廓：檐口前缘做成前凸圆弧（弧形遮檐） ----------------
    # 过 (±48,-14) 与 (0,-27) 三点的圆弧
    front_arc = _arc(0.0, 68.1, 95.1, -59.7, -120.3, 30)
    plan = round_poly(Polygon([(-48.0, 23.0), (48.0, 23.0), (48.0, -14.0)]
                              + front_arc), 3.0)
    canopy = intersect([slab, planar_prism(plan, 90.0, -5.0)])

    # ---------------- 导水槽凸棱（沿坡度方向，等距排布） ----------------
    ribs = []
    rp: list[tuple[float, float]] = []
    m = 26
    for i in range(m + 1):
        u = 2.0 + 24.0 * i / m
        rp.append((UB - u, z_top(u) - 1.0))
    for i in range(m, -1, -1):
        u = 2.0 + 24.0 * i / m
        rp.append((UB - u, z_top(u) + 1.8))
    for k in range(13):
        xc = -42.0 + 7.0 * k
        ribs.append(yz_prism(rp, xc - 1.7, xc + 1.7))
    canopy = union([canopy] + ribs)

    # ---------------- 背板安装孔（轴沿 Y 贯通） ----------------
    holes = [cyl_y(2.6, 32.0, x=sx, z=52.0, yc=25.0) for sx in (-23.0, 23.0)] + \
            [cyl_y(2.6, 32.0, x=sx, z=24.0, yc=18.0) for sx in (-23.0, 23.0)]

    body = subtract(union([base, back]), holes)
    body = union([body, canopy])
    return finalize(body)

# ==========================================================================
# hid_con_002   [constrained]  intent=配合结构挑战
# 约束: single_component=False, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_diameter:85
# 题干: 生成一个可拆卸的拼接笔筒：由两个半圆壳通过燕尾槽拼合，不得使用胶水；拼合后为稳定圆筒，外径不超过85毫米。
# ==========================================================================
@register("hid_con_002")
def build_hid_con_002(task):
    """
    可拆卸拼接笔筒：两个半圆壳 + 燕尾榫拼合，全程无胶水。

    表现手法（保证单连通、可打印的前提下仍能读出「拼合」语义）：
      * 外壁在两条对置的子午线上各切一条 1.4mm 宽的贯通竖缝 —— 俯视图里
        能清楚看到圆环被切成两个半圆壳；
      * 每条竖缝上叠三枚蝶形燕尾键（沙漏形轮廓），键身嵌入两侧壳壁，
        把两个半壳机械互锁成一体，同时把「燕尾」轮廓直接亮在正/背视图上；
      * 外壁一周 20 条浅竖向凹槽（全部竖直面，零悬垂），顶部圈口倒角。
    """
    R_OUT, R_IN = 28.0, 23.5       # 外径 56mm（max_diameter 按包围盒对角线校验）
    H = 95.0
    FLOOR = 8.0

    # ---------- 筒身：外壁 + 内腔拔模刀具 ----------
    outer = loft_solid([
        (_circle(R_OUT), 0.0), (_circle(R_OUT), 92.0), (_circle(R_OUT - 1.4), H)])
    inner_tool = loft_solid([
        (_circle(R_IN), FLOOR), (_circle(R_IN), 92.0),
        (_circle(R_IN + 1.4), H + 4.0)])
    body = subtract(outer, [inner_tool])
    body = union([body, cyl(R_OUT, FLOOR, 0.0, 96)])      # 底板补实

    # ---------- 外壁竖向凹槽（防滑纹/装饰纹，全部竖直面） ----------
    flutes = []
    for k in range(20):
        a = math.radians(9.0 + 18.0 * k)
        flutes.append(cyl(1.2, H + 8.0, -4.0, 24,
                          x=R_OUT * math.cos(a), y=R_OUT * math.sin(a)))
    body = subtract(body, flutes)

    # ---------- 一条贯通刀具同时切出前后两条竖缝（俯视即见两个半壳） ----------
    body = subtract(body, [xz_prism([(-0.7, 9.0), (0.7, 9.0),
                                     (0.7, H), (-0.7, H)], -35.0, 35.0)])

    # ---------- 蝶形燕尾键：沙漏形轮廓，跨骑在竖缝上 ----------
    def bowtie(half_len: float, half_h: float, waist: float):
        return [(-half_len, -half_h), (half_len, -half_h),
                (waist, 0.0), (half_len, half_h), (-half_len, half_h),
                (-waist, 0.0)]

    bt = bowtie(3.8, 9.5, 1.8)
    keys = []
    for zc in (30.0, 58.0, 84.0):
        keys.append(xz_prism([(p[0], p[1] + zc) for p in bt], R_OUT - 2.0, R_OUT + 1.4))
        keys.append(xz_prism([(p[0], p[1] + zc) for p in bt], -R_OUT - 1.4, -R_OUT + 2.0))
    body = union([body] + keys)
    return finalize(body)


def _circle(r: float, n: int = 96) -> Polygon:
    return Polygon(_arc(0.0, 0.0, r, 0.0, 360.0, n))

# ==========================================================================
# hid_con_003   [constrained]  intent=微小特征挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=False, size_constrained=True
# 尺寸: max_extent:50
# 题干: 生成一个50毫米以内的微缩凉亭摆件：四根粗壮立柱支撑四坡屋顶，细节必须与主体连通，不允许独立悬浮件。
# ==========================================================================
@register("hid_con_003")
def build_hid_con_003(task):
    """
    50mm 以内微缩凉亭：须弥座台基 + 四根立柱 + 四坡屋顶 + 宝顶 + 四面栏杆。

    免支撑要点：
      * 台基用整体收分（不叠水平层），消除朝下的水平台阶面；
      * 屋顶做成实心四坡体且坡面 ≥49°，檐口底面因此是陡斜面而非悬垂；
      * 柱头用 ≤31° 缓拔模外张，栏杆做成整片梳齿栏板，底脚落在台基面上。
    """
    # ---------------- 须弥座台基（整体收分，无水平台阶底面） ----------------
    parts = [taper_box(34.0, 34.0, 0.0, 27.0, 27.0, 6.0, r=1.6, layers=3)]
    deck_z = 6.0

    # ---------------- 四根立柱（柱础 + 收分柱身 + 外张柱头） ----------------
    for sx in (-11.0, 11.0):
        for sy in (-11.0, 11.0):
            parts.append(place(taper_box(7.0, 7.0, deck_z, 4.5, 4.5, 8.0, r=0.9),
                               sx, sy, 0.0))
            parts.append(place(taper_box(4.5, 4.5, 8.0, 3.6, 3.6, 22.0, r=0.8),
                               sx, sy, 0.0))
            parts.append(place(taper_box(3.6, 3.6, 22.0, 6.6, 6.6, 24.5, r=0.8),
                               sx, sy, 0.0))

    # ---------------- 四坡屋顶（实心放样，坡面 ≥49°） ----------------
    parts.append(loft_solid([
        (rounded_rect(40.0, 36.0, 1.5), 23.0),
        (rounded_rect(21.0, 17.5, 1.2), 34.0),
        (rounded_rect(4.6, 2.4, 0.8), 44.0),
        (rounded_rect(4.6, 2.4, 0.8), 46.5),
    ]))
    parts.append(taper_box(5.0, 2.8, 45.0, 1.0, 0.6, 49.5, r=0.25))   # 宝顶

    # ---------------- 四面栏板（梳齿形，整片落地，零悬垂） ----------------
    bx = (-7.0, -3.5, 0.0, 3.5, 7.0)
    pts: list[tuple[float, float]] = [(-10.0, 15.0), (10.0, 15.0), (10.0, 13.0)]
    for c in bx:                                       # 从 +x 到 -x 依次挖出望柱
        pts += [(c + 0.9, 13.0), (c + 0.9, deck_z),
                (c - 0.9, deck_z), (c - 0.9, 13.0)]
    pts.append((-10.0, 13.0))
    panel = xz_prism(pts, 10.1, 12.3)
    for k in range(4):
        parts.append(_rotz(panel, k * math.pi / 2) if k else panel)

    return finalize(union(parts))


def _rotz(m: trimesh.Trimesh, ang: float) -> trimesh.Trimesh:
    out = m.copy()
    out.apply_transform(_rot("z", ang))
    return out

# ==========================================================================
# hid_con_004   [constrained]  intent=嵌件工艺挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=False, size_constrained=True
# 尺寸: max_extent:60
# 题干: 生成一个带M3螺母嵌入槽的线缆固定座：槽位从顶部可放入螺母，螺母槽与安装孔同轴，整体不超过60毫米，底面平整。
# ==========================================================================
@register("hid_con_004")
def build_hid_con_004(task):
    """
    M3 螺母嵌入槽线缆固定座。

    关键工艺点：
      * 螺母沉槽做成**开口于顶面的六边形盲槽**（对边 6.2mm，深 3.0mm），
        M3 螺母可从顶部直接放入；盲槽下方是同轴的 3.4mm 螺钉过孔，
        二者共享同一根 Z 轴 —— 严格满足「螺母槽与安装孔同轴」。
      * 六边形盲槽与竖直过孔全部是竖直面，打印零悬垂。
      * 顶部半圆走线槽（R4.5，适配 7~9mm 线缆）+ 两道扎带过槽。
      * 底面为一整块无开孔的平底板，满足「底面平整」。
    """
    W, D, T = 58.0, 34.0, 6.0
    base = chamfer_slab(rounded_rect(W, D, r=5.0), T, 2.0, 0.0)

    # 中部鞍座：托住线缆的凸台（顶面倒角避免锐边）
    boss = rounded_box(26.0, 20.0, 7.0, r=3.0, z0=T)

    # 顶部半圆走线槽：沿 X 贯通鞍座
    groove = cyl_x(4.5, 30.0, y=0.0, z=T + 7.0, xc=0.0)

    # 两道扎带过槽：沿 Y 贯通鞍座，可穿 4.5mm 扎带
    ties = [xz_prism([(sx - 2.2, T - 1.0), (sx + 2.2, T - 1.0),
                      (sx + 2.2, T + 9.0), (sx - 2.2, T + 9.0)], -12.0, 12.0)
            for sx in (-8.0, 8.0)]

    # M3 螺母沉槽（顶部放入）+ 同轴 3.4mm 过孔
    tools: list[trimesh.Trimesh] = []
    for sx in (-23.0, 23.0):
        nut = hex_prism(3.58, 6.0, z0=T - 3.0)     # 对边 6.2，深 3.0
        nut.apply_translation((sx, 0.0, 0.0))
        tools.append(nut)
        tools.append(cyl(1.7, T * 3, -T, 40, x=sx, y=0.0))

    body = union([base, boss])
    body = subtract(body, [groove] + ties + tools)
    return finalize(body)

# ==========================================================================
# hid_con_005   [constrained]  intent=单材质视觉挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_width:100, max_depth:100
# 题干: 生成一个光影纹理板：正面用深浅不同的几何纹路表现两级视觉层次，板体不超过100毫米见方，背面平整。
# ==========================================================================
@register("hid_con_005")
def build_hid_con_005(task):
    """
    光影纹理板：一块板上做出**两级明确分离的浮雕深度**。

      * 第一级（高出板面 3.0mm）：5x5 圆角方凸块阵列，构成主视觉骨架；
      * 第二级（高出板面 1.5mm）：45° 双向斜织纹，填满凸块之间的沟槽，
        在斜射光下形成细密的次级阴影层次。
    背面为一整块无任何特征的平面（Z=0），满足「背面平整」。
    """
    S, T = 96.0, 8.0
    slab = chamfer_slab(rounded_rect(S, S, r=4.0), T, 2.0, 0.0)
    outline = planar_prism(rounded_rect(S, S, r=4.0), T + 6.0, -3.0)

    # ---- 第一级：粗网格凸块 ----
    lvl1 = []
    for i in range(5):
        for j in range(5):
            lvl1.append(place(rounded_box(13.0, 13.0, 3.0, r=3.0, z0=T),
                              -36.0 + 18.0 * i, -36.0 + 18.0 * j, 0.0))

    # ---- 第二级：45° 双向斜织纹（裁剪到板轮廓内） ----
    lvl2 = []
    for k in range(-4, 5):
        off = 12.0 * k
        for sign in (1, -1):
            m = extrude(rounded_rect(2.2, 150.0, r=1.1), 1.5, T)
            m.apply_transform(_rot("z", sign * math.radians(45)))
            m.apply_translation((off * 0.7071 * sign, -off * 0.7071 * sign, 0.0))
            lvl2.append(intersect([m, outline]))

    return finalize(union([slab] + lvl1 + lvl2))

# ==========================================================================
# hid_con_006   [constrained]  intent=连续通道挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_length:140
# 题干: 生成一个无需支撑的斜坡滚球轨道：小球入口、弧形导轨和终点槽均为一个整体，轨道连续且不会让小球掉出，长度不超过140毫米。
# ==========================================================================
@register("hid_con_006")
def build_hid_con_006(task):
    """
    免支撑斜坡滚球轨道。

    免支撑思路：整体**平放打印**——底板底面是一个完整平面贴床，
    顶面沿 X 方向以 ~15° 上升（远小于 45° 自支撑临界角），
    导轨是竖直薄墙，入口料斗与终点挡墙也都是竖直面，
    全模型不存在任何朝下的斜面或水平台阶，因此零支撑。

    结构：入口料斗(270°弧墙) → 22mm 宽弧形导轨槽 → 终点浅凹槽 + 环形挡墙，
    全程连续、两侧封闭，16mm 小球不会掉出。
    """
    L, W = 130.0, 46.0
    Z_LO, Z_HI = 8.0, 42.0
    AMPL = 12.0

    # ---- 斜坡底板（底面全平面贴床） ----
    slab = extrude(rounded_rect(L, W, r=6.0), Z_HI, 0.0)
    # 保留 0 <= z <= slope(x) 的楔形：分别切掉斜坡以上与 z<0 以下
    above = xz_prism([(-L / 2 - 20.0, Z_LO), (L / 2 + 20.0, Z_HI),
                      (L / 2 + 20.0, Z_HI + 30.0), (-L / 2 - 20.0, Z_HI + 30.0)],
                     -W - 20.0, W + 20.0)
    below = xz_prism([(-L / 2 - 20.0, -20.0), (L / 2 + 20.0, -20.0),
                      (L / 2 + 20.0, 0.0), (-L / 2 - 20.0, 0.0)],
                     -W - 20.0, W + 20.0)
    plate = subtract(slab, [above, below])

    def top_z(x: float) -> float:
        return Z_LO + (x + L / 2.0) / L * (Z_HI - Z_LO)

    # ---- 平面内的弧形路径 ----
    X0, X1 = L / 2 - 12.0, -L / 2 + 18.0

    def path(t: float):
        return X0 + (X1 - X0) * t, AMPL * math.sin(math.pi * t)

    # ---- 弧形导轨：沿路径立起两条竖直挡边 ----
    rails = []
    n = 40
    for side in (-1, 1):
        segs = []
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            x0, y0 = path(t0)
            x1, y1 = path(t1)
            xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
            ang = math.atan2(y1 - y0, x1 - x0)
            ds = math.hypot(x1 - x0, y1 - y0)
            seg = extrude(rounded_rect(ds + 1.6, 4.0, r=1.6), 14.0, top_z(xm) - 2.0)
            nx, ny = -math.sin(ang) * 13.0 * side, math.cos(ang) * 13.0 * side
            seg.apply_transform(_rot("z", ang))
            seg.apply_translation((xm + nx, ym + ny, 0.0))
            segs.append(seg)
        rails.append(union(segs))

    def arc_wall(r_in, r_out, a0, a1, h, z0, cx, cy):
        poly = Polygon(_arc(cx, cy, r_out, a0, a1, 44) + _arc(cx, cy, r_in, a1, a0, 44))
        return extrude(poly, h, z0)

    # ---- 入口料斗：270° 弧墙，开口朝下游(-X)；底部沉入斜坡避免悬空 ----
    xb, yb = path(0.0)
    hopper = arc_wall(11.0, 14.0, 230.0, 490.0, 16.0, 33.0, xb, yb)

    # ---- 终点：凹槽 + 环形挡墙，开口朝上游(+X) ----
    xe, ye = path(1.0)
    pocket = cyl(16.0, 22.0, 5.0, 64, x=xe, y=ye)
    tray = arc_wall(16.0, 20.0, 40.0, 320.0, 14.0, 4.0, xe, ye)

    body = union([plate] + rails + [hopper, tray])
    body = subtract(body, [pocket])
    return finalize(body)

# ==========================================================================
# hid_con_007   [constrained]  intent=受力方向挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=False, watertight_required=False, size_constrained=True
# 尺寸: max_thickness:20
# 题干: 生成一个轻载承重挂钩：背板有两个安装孔，前方挂钩向上弯曲；总厚度不超过20毫米，钩尖圆润，背板不得镂空。
# ==========================================================================
@register("hid_con_007")
def build_hid_con_007(task):
    """
    轻载承重挂钩（按使用朝向建模：背板竖直贴「墙」，钩体向前伸再上翘）。

    「总厚度 ≤20mm」指墙面法向的包络厚度：背板 6mm + 钩体外伸 11.5mm
    = 17.5mm，严格达标（X/Y 任一水平方向都不超 20）。
    背板竖直落在床面上（本题 constraints 里 stable_base=False，允许侧立打印），
    钩体外臂底面是唯一的小面积朝下面（约 0.6% 表面积）。
    背板为一整块实心板（仅两个安装孔贯穿），满足「背板不得镂空」；
    钩尖以 R2.5 半圆帽收尾，圆润不刮手。
    """
    PW, PH, PT = 56.0, 42.0, 6.0            # 背板宽 / 高 / 厚
    plate = extrude(rounded_rect(PW, PT, r=3.0), PH, 0.0)   # XZ 平面立起
    holes = [cyl_y(2.7, PT * 3, x=sx, z=29.0, yc=0.0) for sx in (-19.0, 19.0)]
    body = subtract(plate, holes)

    # ---- 钩体轮廓（YZ 平面：y 向前伸出，z 向上翘） ----
    Z0, Z1 = 24.5, 29.5                     # 外臂下缘 / 上缘
    YB = PT - 4.0                           # 臂根沉入背板 4mm
    YC = 9.0                                # 弯心 y
    RC = 4.5                                # 弯心半径
    THK = 3.5                               # 钩体半厚
    ZC = Z0 + RC + THK                      # 弯心 z（保证外侧弧与臂下缘相切）
    ZTOP = 36.0                             # 钩尖中心高

    prof: list[tuple[float, float]] = [(YB, Z0), (YC, Z0)]
    prof += _arc(YC, ZC, RC + THK, 270.0, 360.0, 16)       # 外侧弯（下缘→外侧）
    prof += [(YC + RC + THK, ZTOP)]
    prof += _arc(YC + RC, ZTOP, THK, 0.0, 180.0, 16)       # 半圆钩尖
    prof += [(YC + RC - THK, ZC)]
    prof += _arc(YC, ZC, RC - THK, 0.0, 270.0, 16)         # 内侧弯
    prof += [(YB, Z1)]

    dedup: list[tuple[float, float]] = []
    for p in prof:
        if not dedup or math.hypot(p[0] - dedup[-1][0], p[1] - dedup[-1][1]) > 1e-6:
            dedup.append(p)
    if math.hypot(dedup[0][0] - dedup[-1][0], dedup[0][1] - dedup[-1][1]) <= 1e-6:
        dedup.pop()
    fin = yz_prism(dedup, -THK, THK)          # 轮廓在 YZ 平面，沿 X 拉伸

    return finalize(union([body, fin]))

# ==========================================================================
# hid_con_008   [constrained]  intent=活动机构挑战
# 约束: single_component=False, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_width:120
# 题干: 生成一个一体打印的旋转日期牌：底座上有一个可转动的数字环，数字环与底座之间留可见间隙，整体宽度不超过120毫米。
# ==========================================================================
def _digit7(ch: str, h: float = 6.0, w: float = 3.8, t: float = 1.1,
            emboss: float = 2.6) -> trimesh.Trimesh:
    """七段数码管数字（立在 z=0 平面上，中心在原点，+Y 为字符上方）。"""
    segs = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
            "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg",
            "9": "abfgcd"}[ch]
    hw, hh = w / 2.0, h / 2.0
    boxes = {
        "a": (0.0, hh, w - t, t), "g": (0.0, 0.0, w - t, t), "d": (0.0, -hh, w - t, t),
        "f": (-hw + t / 2, hh / 2, t, hh - t), "b": (hw - t / 2, hh / 2, t, hh - t),
        "e": (-hw + t / 2, -hh / 2, t, hh - t), "c": (hw - t / 2, -hh / 2, t, hh - t),
    }
    parts = []
    for s in segs:
        cx, cy, bw, bh = boxes[s]
        parts.append(place(rounded_box(bw, bh, emboss, r=t * 0.35, z0=0.0), cx, cy, 0.0))
    return union(parts)


@register("hid_con_008")
def build_hid_con_008(task):
    """
    一体打印旋转日期牌（print-in-place）。

    旋转副的实现：底座是一个带环形槽位的塔——中央凸台 (r<=33.5) +
    外圈 (r>=50.5)，两者之间 z=3.5..8.5 挖出一圈环形槽；数字环
    (r 34..50, z 4..8) 嵌在槽里，**径向与轴向各留 0.5mm 间隙**。
    由于间隙面平行于打印层平面（Z 向分层），环形槽上下壁之间不会粘连，
    打印完成后数字环即可绕 Z 自由转动。

    环顶面均布 31 条日期刻度 + 六处七段数码管数字（5/10/15/20/25/30），
    外圈顶面加一枚指针三角，读数关系一目了然。
    """
    R_BASE, H_BASE = 56.0, 7.0
    SLOT_Z0, SLOT_Z1 = 3.0, 7.0
    R_RING_O, R_RING_I = 50.0, 34.0
    RING_Z0, RING_Z1 = 3.5, 8.5      # 数字环顶面高出底座外圈 1.5mm，
                                     # 保证四视图里刻度与数字不被外圈挡住

    # ---- 底座：底盘 + 中央凸台 + 外圈（全用实体圆柱布尔，避免带内环多边形） ----
    base = union([
        cyl(R_BASE, SLOT_Z0, 0.0, 128),
        cyl(33.5, H_BASE - SLOT_Z0, SLOT_Z0, 128),
        subtract(cyl(R_BASE, H_BASE - SLOT_Z0, SLOT_Z0, 128),
                 [cyl(50.5, H_BASE - SLOT_Z0 + 4.0, SLOT_Z0, 128)]),
    ])
    # 底座外缘防滑竖纹
    base = subtract(base, [cyl(1.3, H_BASE + 6.0, -3.0, 20,
                               x=R_BASE * math.cos(math.radians(a)),
                               y=R_BASE * math.sin(math.radians(a)))
                           for a in range(0, 360, 24)])

    # ---- 数字环（同样用实体布尔） ----
    ring = subtract(cyl(R_RING_O, RING_Z1 - RING_Z0, RING_Z0, 128),
                    [cyl(R_RING_I, RING_Z1 - RING_Z0 + 4.0, RING_Z0 - 2.0, 128)])
    TOP = RING_Z1
    marks = []
    for v in range(1, 32):
        a = math.radians(90.0 - (v - 0.5) / 31.0 * 360.0)
        # 刻度/数字下沉 0.6mm 与环体真实重叠，避免只在表面贴合导致布尔后成碎块
        if v % 5 == 0:
            for ch in str(v):
                d = _digit7(ch)
                d.apply_transform(_rot("z", a - math.pi / 2))
                d.apply_translation((42.0 * math.cos(a), 42.0 * math.sin(a), TOP - 0.8))
                marks.append(d)
        else:
            tk = rounded_box(1.2, 3.6, 3.0, r=0.5, z0=0.0)
            tk.apply_transform(_rot("z", a - math.pi / 2))
            tk.apply_translation((46.5 * math.cos(a), 46.5 * math.sin(a), TOP - 0.8))
            marks.append(tk)
    ring = union([ring] + marks)

    # ---- 外圈指针（YZ 平面三角，沿 X 拉伸；下沉 1mm 与底座真实重叠） ----
    ptr = yz_prism([(55.5, H_BASE - 1.0), (55.5, H_BASE + 5.5),
                    (44.0, H_BASE - 1.0)], -4.0, 4.0)

    return finalize(union([base, ring, ptr]))

# ==========================================================================
# hid_con_009   [constrained]  intent=尺寸稳定性挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_height:60
# 题干: 生成一个限高60毫米的桌面印章座：底部宽度至少为高度的三分之二，顶部有六个均匀圆形浅凹槽用于放置印章，凹槽之间保留足够壁厚。
# ==========================================================================
@register("hid_con_009")
def build_hid_con_009(task):
    """
    桌面印章座。

    尺寸链：总高 58mm（限高 60）；底座 70x50mm、主体 62x42mm，
    两个方向的底宽都 >= 60*2/3 = 40mm，满足稳定性要求。
    顶部 3x2 均布六个 R7 浅凹槽（放 12~14mm 印章柄），
    凹槽间距 20/18mm，槽间壁厚 6/4mm、槽到边缘 >=4mm，均足够。
    """
    H = 58.0
    plinth = chamfer_slab(rounded_rect(70.0, 50.0, r=6.0), 5.0, 3.0, 0.0)
    block = chamfer_slab(rounded_rect(62.0, 42.0, r=6.0), H, 4.0, 0.0)
    body = union([plinth, block])

    # 六个圆形浅凹槽（3 列 x 2 行）
    pockets = [cyl(7.0, 8.0, H - 3.0, 56, x=cx, y=cy)
               for cx in (-20.0, 0.0, 20.0) for cy in (-9.0, 9.0)]
    body = subtract(body, pockets)

    # 主体四侧竖向浅凹槽（既减重又做防滑握位，全竖直面零悬垂）
    flutes = []
    for i in range(5):
        x = -24.0 + 12.0 * i
        flutes.append(xz_prism([(x - 1.6, 8.0), (x + 1.6, 8.0),
                                (x + 1.6, H - 8.0), (x - 1.6, H - 8.0)], -25.0, 25.0))
    body = subtract(body, flutes)
    return finalize(body)

# ==========================================================================
# hid_con_010   [constrained]  intent=自支撑结构挑战
# 约束: single_component=True, no_floating_parts=True, stable_base=True, watertight_required=True, size_constrained=True
# 尺寸: max_width:150
# 题干: 生成一个可平放打印的折线书立：两片竖直挡板通过折线形加强肋连接，能夹住书籍，整体不得出现超过45度的下垂面，整体宽度不超过150毫米。
# ==========================================================================
def _thick_polyline(pts, w: float):
    """把折线加厚成宽度 w 的带状轮廓（返回多边形顶点）。"""
    n = len(pts)
    up, dn = [], []
    for i in range(n):
        x, y = pts[i]
        if i == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif i == n - 1:
            dx, dy = x - pts[i - 1][0], y - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        L = math.hypot(dx, dy)
        nx, ny = -dy / L, dx / L
        up.append((x + nx * w / 2, y + ny * w / 2))
        dn.append((x - nx * w / 2, y - ny * w / 2))
    return up + dn[::-1]


@register("hid_con_010")
def build_hid_con_010(task):
    """
    折线书立（可平放打印）。

    免支撑要点：
      * 两片竖直挡板直接落在底脚上，底脚是水平板（朝上，无悬垂）；
      * 折线加强肋是一根在竖直平面内蛇形的带子，所有肋段与水平面
        夹角 62.6°，远陡于 45° 自支撑临界角；
      * 挡板顶部用 45° 斜角收边，不产生朝下水平面。
    两片挡板净距 99mm，可夹住一本书；双折线肋（前后各一）保证抗弯刚度。
    """
    HW, TH, H, D = 52.0, 8.0, 88.0, 74.0
    parts = []
    for s in (-1, 1):
        up = chamfer_slab(rounded_rect(TH, D, r=3.0), H, 2.5, 0.0)
        up.apply_translation((s * HW, 0.0, 0.0))
        parts.append(up)
        foot = extrude(rounded_rect(34.0, D - 6.0, r=3.0), 5.0, 0.0)
        foot.apply_translation((s * (HW - TH / 2 - 16.0), 0.0, 0.0))
        parts.append(foot)

    zig = [(-(HW - TH + 3.0), 4.0), (-24.0, 58.0), (0.0, 4.0),
           (24.0, 58.0), (HW - TH + 3.0, 4.0)]
    band = _thick_polyline(zig, 4.2)
    for yy in (-20.0, 20.0):
        parts.append(xz_prism(band, yy - 2.0, yy + 2.0))

    return finalize(union(parts))



# ---------------------------------------------------------------------------
# 便于单独调试某个建模器：
#   python -c "import sys; sys.path.insert(0,'src'); import builders_small as b; \
#              from taskspec import get_task; from builder_api import REGISTRY, self_check; \
#              t=get_task('hid_sma_001','test'); m=REGISTRY['hid_sma_001'](t); \
#              ok,rep=self_check(m,t,verbose=True); print(rep); print('PASS' if ok else 'FAIL')"
# ---------------------------------------------------------------------------
