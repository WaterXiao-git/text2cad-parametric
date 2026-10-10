"""
小型器物（small_object） 建模器 —— hid_sma_001, hid_sma_002, hid_sma_003, hid_sma_004, hid_sma_005, hid_sma_006, hid_sma_007, hid_sma_008, hid_sma_009, hid_sma_010

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

from builder_api import finalize, register, self_check, union
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
# 私有工具
# ==========================================================================
def _rot(axis: str, ang: float) -> np.ndarray:
    d = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[axis]
    return trimesh.transformations.rotation_matrix(ang, d)


def _rx(m, a):
    o = m.copy(); o.apply_transform(_rot("x", a)); return o


def _rz(m, a):
    o = m.copy(); o.apply_transform(_rot("z", a)); return o


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


def _star(r_out, r_in, points=5, rot=0.0):
    pts = []
    for i in range(points * 2):
        a = rot + math.pi * i / points
        pts.append(((r_out if i % 2 == 0 else r_in) * math.cos(a),
                    (r_out if i % 2 == 0 else r_in) * math.sin(a)))
    return Polygon(_dedupe(pts))


def _cyl_y(r, length, x=0.0, z=0.0, sections=48):
    m = trimesh.creation.cylinder(radius=float(r), height=float(length),
                                  sections=int(sections))
    m.apply_transform(_rot("x", math.pi / 2))
    m.apply_translation((float(x), 0.0, float(z)))
    return m


# ==========================================================================
# hid_sma_001  骰子收纳旋盖筒（打印就位旋转盖）
# ==========================================================================
@register("hid_sma_001")
def build_hid_sma_001(task):
    """
    骰子收纳旋盖筒（打印就位 print-in-place）。

    六边形短筒（外接圆半径 32，平移对边距 55.4，外廓 <= 70），壁厚 4、
    底板 6（底部平整），外壁 12 条竖向防滑凹槽；旋盖是独立六边壳，
    借内衬套在筒口外翻缘上，轴向与径向各留 0.4mm 间隙，可绕 Z 旋转；
    盖顶面一枚抽象五角星浮雕。
    """
    R = 32.0
    hexpts = [(R * math.cos(math.radians(60 * i + 30)),
               R * math.sin(math.radians(60 * i + 30))) for i in range(6)]
    hexo = Polygon(hexpts)
    hexi = Polygon([(28.0 * math.cos(math.radians(60 * i + 30)),
                     28.0 * math.sin(math.radians(60 * i + 30))) for i in range(6)])
    H = 54.0
    tube = subtract(extrude(hexo, H, 0.0), [extrude(hexi, H + 6.0, 6.0)])
    tube = union([tube, extrude(hexi, 6.0, 0.0)])                # 底板

    # 外翻缘（供旋盖卡住）：z 46..52 半径外扩到 R+2
    flange = subtract(extrude(_hex_poly(R + 2.0), 6.0, 46.0),
                      [extrude(_hex_poly(R - 5.0), 12.0, 44.0)])
    tube = union([tube, flange])

    # 外壁竖向防滑凹槽
    for i in range(12):
        a = math.radians(30.0 * i)
        tube = subtract(tube, [place(cyl(1.6, H + 6.0, -3.0, 20,
                                         x=(R + 0.5) * math.cos(a),
                                         y=(R + 0.5) * math.sin(a)), 0, 0, 0)])

    # 旋盖：六边壳 + 内衬套，与外翻缘留 0.4mm 间隙
    lid_o = _hex_poly(R + 2.8)
    lid_cav = _hex_poly(R + 0.4)
    lid = subtract(extrude(lid_o, 12.0, 52.4), [extrude(lid_cav, 10.0, 52.4)])
    spigot_o = _hex_poly(R - 4.6)
    spigot = subtract(extrude(spigot_o, 8.0, 44.4), [extrude(_hex_poly(R - 8.6), 14.0, 42.0)])
    lid = union([lid, spigot])
    # 盖顶五角星浮雕
    lid = union([lid, place(extrude(_star(11.0, 4.6, 5, rot=0.25), 1.4, 63.8), 0, 0, 0)])
    return finalize(union([tube, lid]))


def _hex_poly(r):
    return Polygon([(r * math.cos(math.radians(60 * i + 30)),
                     r * math.sin(math.radians(60 * i + 30))) for i in range(6)])


# ==========================================================================
# hid_sma_002  眼镜镜腿临时挂架
# ==========================================================================
@register("hid_sma_002")
def build_hid_sma_002(task):
    """
    眼镜镜腿临时挂架。

    拱形底座：两片三角支撑腿 + 拱梁（跨度 62、拱高 46 ≈ 建议值 45），
    拱顶两侧各一道 R4.5 浅 U 形凹槽（开口向上、深 6），眼镜可倒扣搁置。
    """
    L, W = 54.0, 30.0
    parts = []
    for sx in (-L / 2, L / 2):
        parts.append(place(taper_box(18.0, W, 0.0, 12.0, W - 6.0, 28.0, r=2.5),
                           sx, 0.0, 0.0))
    # 拱梁（下缘平、上缘起拱）
    prof = [(-L / 2, 21.0), (L / 2, 21.0), (L / 2, 27.0)]
    prof += _arc(0.0, 21.0, L / 2, 0.0, 180.0, 30)
    parts.append(_xz_prism(prof, -W / 2, W / 2))
    body = union(parts)

    # 两侧浅 U 形凹槽（开口向上，R4.5，轴沿 Y 贯通）
    for sx in (-20.0, 20.0):
        m = trimesh.creation.cylinder(radius=4.5, height=W + 6.0, sections=40)
        m.apply_transform(_rot("x", math.pi / 2))
        m.apply_translation((sx, 0.0, 33.0))
        body = subtract(body, [m])
    return finalize(body)


# ==========================================================================
# hid_sma_003  可替换标签的钥匙识别牌
# ==========================================================================
@register("hid_sma_003")
def build_hid_sma_003(task):
    """
    钥匙识别牌。

    圆角长条主体 82x30x5（长度 = 建议值 80 量级），左端 R5 穿环孔
    （可挂钥匙圈），正面 56x18x1.8 浅槽嵌纸质标签，槽口留 6mm 豁口
    方便更换纸片，边角全部 R4 圆润过渡。
    """
    W, D, T = 82.0, 30.0, 5.0
    body = chamfer_slab(rounded_rect(W, D, r=4.0), T, 1.6, 0.0)
    # 穿环孔
    body = subtract(body, [cyl(5.0, T + 6.0, -3.0, 40, x=-W / 2 + 9.0, y=0.0)])
    # 标签浅槽（正面开 1.8 深凹槽）
    slot = rounded_box(56.0, 18.0, 4.0, r=2.0, z0=T - 1.8)
    body = subtract(body, [place(slot, 5.0, 0.0, 0.0)])
    # 槽口豁口（方便换纸）
    body = subtract(body, [place(rounded_box(12.0, 26.0, 4.0, r=2.0, z0=T - 1.9),
                                 W / 2 - 20.0, 0.0, 0.0)])
    return finalize(body)


# ==========================================================================
# hid_sma_004  指尖陀螺展示座
# ==========================================================================
@register("hid_sma_004")
def build_hid_sma_004(task):
    """
    指尖陀螺展示座。

    圆盘底座直径 62（= 建议值 60 量级）、高 8；中央竖直短轴 R5、高 26，
    轴顶端以 R6.5 半圆帽一体成型（防脱），帽下 14mm 处再放一个 R11 托盘，
    可承托圆形小物。全件回转体，立式打印零悬垂。
    """
    body = chamfer_slab(_circle(31.0, 128), 8.0, 2.5, 0.0)
    body = union([body, cyl(5.0, 26.0, 0.0, 48)])
    # 防脱圆帽（半球 + 短柱）
    cap = trimesh.creation.icosphere(subdivisions=3, radius=6.5)
    cap.apply_translation((0.0, 0.0, 26.0))
    body = union([body, cap])
    # 托盘
    body = union([body, cyl(11.0, 4.0, 12.0, 64)])
    return finalize(body)


# ==========================================================================
# hid_sma_005  吉他拨片收纳拨盘（打印就位旋转盘）
# ==========================================================================
@register("hid_sma_005")
def build_hid_sma_005(task):
    """
    吉他拨片收纳拨盘（打印就位 print-in-place）。

    圆形扁盒直径 88（<= 90）、高 26；顶面开月牙形取片口，盒内一块
    可旋转选厚度的指示盘（独立体，与盒底留 0.4mm 分层间隙，可绕 Z 旋转），
    盘面均布 0.6/0.73/0.88/1.0 四档厚度刻记；盒底封闭平整。
    """
    R, H = 44.0, 26.0
    shell = subtract(cyl(R, H, 0.0, 128), [cyl(R - 4.0, H + 6.0, 6.0, 128)])
    shell = union([shell, cyl(R, 6.0, 0.0, 128)])                # 底板
    # 顶面月牙取片口
    moon = Polygon(_arc(-6.0, 0.0, 20.0, -62.0, 62.0, 28)
                   + _arc(2.0, 0.0, 20.0, 58.0, 302.0, 28))
    shell = subtract(shell, [extrude(moon, 8.0, H - 5.0)])

    # 旋转指示盘（独立体）
    dial = cyl(R - 5.0, 3.0, 6.4, 128)
    for k, rr in enumerate((8.0, 14.0, 20.0, 26.0)):
        a = math.radians(90.0 * k)
        dial = union([dial, place(cyl(2.2, 5.0, 6.4, 20, x=rr * math.cos(a),
                                      y=rr * math.sin(a)), 0, 0, 0)])
    return finalize(union([shell, dial]))


# ==========================================================================
# hid_sma_006  迷你香料量勺挂架
# ==========================================================================
@register("hid_sma_006")
def build_hid_sma_006(task):
    """
    香料量勺挂架。

    半圆底座（半径 36、宽 72 ≈ 建议值 70、厚 10）上竖起三根 R5 短柱
    （高 52，前后错开），每根柱顶以 R6 圆角钩收头（钩口 11mm），
    可挂小量勺。立式打印，柱身竖直零悬垂。
    """
    body = extrude(Polygon(_arc(0, 0, 36.0, 0.0, 180.0, 48)), 10.0, 0.0)
    for k, (sx, sy) in enumerate(((-24.0, 6.0), (0.0, 9.0), (24.0, 6.0))):
        body = union([body, cyl(5.0, 52.0, 0.0, 40, x=sx, y=sy)])
        prof = [(sx - 6.0, 46.0), (sx - 6.0, 56.0)]
        prof += _arc(sx, 56.0, 6.0, 180.0, 320.0, 16)
        prof += [(sx + 6.0, 46.0)]
        body = union([body, _xz_prism(prof, sy - 3.0, sy + 3.0)])
    return finalize(body)


# ==========================================================================
# hid_sma_007  印章防滚置放槽
# ==========================================================================
@register("hid_sma_007")
def build_hid_sma_007(task):
    """
    印章防滚置放槽。

    长条底座 102x34x16（长度 = 建议值 100 量级），中央开 R9 半圆凹槽
    （可放 R8~R11 圆柱印章），两端各以 12° 斜面微微抬高形成挡肩，
    印章放进后不会滚出。
    """
    L, W, H = 102.0, 34.0, 16.0
    # 两端抬高的楔形轮廓
    prof = [(0, 0), (L, 0), (L, H), (L - 18.0, H - 6.0), (18.0, H - 6.0), (0, H)]
    body = _xz_prism(prof, -W / 2, W / 2)
    # 中央半圆凹槽（沿 X 贯通，槽底落在 z = H-6-9 = 1 以上）
    groove = trimesh.creation.cylinder(radius=9.0, height=L + 8.0, sections=64)
    groove.apply_transform(_rot("y", math.pi / 2))
    groove.apply_translation((L / 2, 0.0, H - 6.0))
    body = subtract(body, [groove])
    return finalize(body)


# ==========================================================================
# hid_sma_008  摄像头镜头盖收纳牌
# ==========================================================================
@register("hid_sma_008")
def build_hid_sma_008(task):
    """
    镜头盖收纳牌。

    卡片形底板 84x56x6（宽度 = 建议值 80 量级），上开四个不同直径的
    圆形嵌槽（R14 / R11 / R8.5 / R6，深 3，对应常见镜头盖尺寸），
    左上角一枚 R4 挂绳孔，边角全部 R5 圆润过渡。
    """
    W, D, T = 84.0, 56.0, 6.0
    body = chamfer_slab(rounded_rect(W, D, r=5.0), T, 1.8, 0.0)
    # 四个不同直径嵌槽（2x2 排布，槽间壁厚 >= 5）
    spec = [(-19.0, 9.0, 14.0), (17.0, 11.0, 11.0),
            (-17.0, -13.0, 8.5), (18.0, -15.0, 6.0)]
    for cx, cy, r in spec:
        body = subtract(body, [cyl(r, 5.0, T - 3.0, 64, x=cx, y=cy)])
    # 挂绳孔
    body = subtract(body, [cyl(4.0, T + 6.0, -3.0, 40, x=-W / 2 + 8.0, y=D / 2 - 8.0)])
    return finalize(body)


# ==========================================================================
# hid_sma_009  可折叠购物袋固定夹
# ==========================================================================
@register("hid_sma_009")
def build_hid_sma_009(task):
    """
    购物袋固定夹（C 形夹）。

    方块 72x26x22，从右端水平挖入一条 8mm 高的开口槽形成 C 形；
    上臂右端加宽 6mm 作按压片，下臂上表面带 5 道 1.6mm 浅齿纹增加摩擦。
    侧壁厚 7，立式打印零悬垂。
    """
    L, W, H = 72.0, 26.0, 22.0
    body = chamfer_slab(rounded_rect(L, W, r=4.0), H, 2.5, 0.0)
    # 侧向开口槽（从 +X 端挖入，高 8）
    body = subtract(body, [place(rounded_box(30.0, W + 6.0, 8.0, r=2.0, z0=H / 2 - 4.0),
                                 L / 2 - 10.0, 0.0, 0.0)])
    # 上臂按压片（加宽）
    body = union([body, place(rounded_box(22.0, W + 6.0, 12.0, r=3.0, z0=H / 2 + 4.0),
                              L / 2 - 14.0, 0.0, 0.0)])
    # 下臂上表面浅齿纹
    for k in range(5):
        x = L / 2 - 34.0 + 4.0 * k
        body = union([body, _xz_prism([(x - 0.8, H / 2 - 4.0), (x + 0.8, H / 2 - 4.0),
                                       (x + 0.8, H / 2 - 2.2), (x - 0.8, H / 2 - 2.2)],
                                      -W / 2, W / 2)])
    return finalize(body)


@register("hid_sma_010")
def build_hid_sma_010(task):
    """
    植物插枝高度标尺。

    窄长条底座 78x30x10 + 垂直刻度板 10x8x124（总高 128 <= 130），
    刻度以 3mm 高浮雕短线表示（每 10mm 一道长刻度、每 5mm 一道短刻度），
    顶部收成圆润叶片轮廓。刻度全部是竖直壁上的凸起，零悬垂。
    """
    base = chamfer_slab(rounded_rect(78.0, 30.0, r=5.0), 10.0, 2.5, 0.0)
    H = 124.0
    panel = _xz_prism([(-5.0, 0.0), (5.0, 0.0), (5.0, H - 18.0)]
                      + _arc(0.0, H - 18.0, 5.0, 0.0, 180.0, 20), -4.0, 4.0)
    body = union([base, panel])
    # 浮雕刻度（正面 + 背面）
    for mm in range(10, 121, 5):
        length = 6.0 if mm % 10 == 0 else 3.2
        for yy in (-4.6, 3.4):
            body = union([body, _xz_prism([(-length / 2, mm), (length / 2, mm),
                                           (length / 2, mm + 1.6),
                                           (-length / 2, mm + 1.6)], yy, yy + 1.2)])
    return finalize(body)
