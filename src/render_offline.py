"""
纯 numpy + Pillow 的离线四视图渲染器（**不依赖 Blender / OpenGL / GPU**）。

为什么需要它
------------
官方《决赛复现提交要求》2.3 规定 run_infer 的输出结构里每题必须包含
`renders/{front,side,top,perspective}.png`。但初赛阶段用的 Blender Cycles
在决赛的断网隔离容器里不可用（python:3.11-slim 里没有 Blender，也无法在线装），
因此必须自带一个纯 Python 的软件光栅化渲染器。

实现
----
* 正交投影（front/side/top）+ 透视投影（perspective）；
* 画家级 z-buffer 逐三角形光栅化（numpy 向量化，按屏幕包围盒裁剪）；
* 背面剔除用三维法向判定（n·视线方向 < 0 为可见面），避免绘制内壁；
* 光照 = 半球环境光 + 主光 + 补光 + 轻微镜面高光，用网格自身 PBR 底色着色；
* 2 倍超采样（SSAA）后降采样，得到接近离线渲染的平滑边缘。

之所以能"免支撑式"地快：逐面开销主要是 numpy 调用开销而非像素量，
所以 2x 超采样几乎不增加耗时。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

__all__ = ["render_four_views", "render_view"]

DEFAULT_SIZE = 768
SSAA = 2                    # 超采样倍数
MARGIN = 0.86               # 模型占画面比例

# 背景：浅灰到白的中性渐变（多模态模型看图时对比度友好）
BG_TOP = np.array([247.0, 248.0, 250.0])
BG_BOTTOM = np.array([218.0, 223.0, 231.0])

# 三盏灯：主光（左上偏前）、补光（右下）、顶光
_KEY = np.array([-0.42, -0.66, 0.62])
_FILL = np.array([0.62, 0.34, 0.20])
_TOP = np.array([0.05, 0.10, 0.99])


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _basis(direction, up_hint):
    """由「视线方向 f（相机→物体）」与「上方向提示」构造 (right, up, forward)。"""
    f = _unit(np.asarray(direction, dtype=float))
    u_h = _unit(np.asarray(up_hint, dtype=float))
    r = _unit(np.cross(f, u_h))
    u = _unit(np.cross(r, f))
    return r, u, f


def _background(h: int, w: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, h, dtype=float)[:, None, None]
    bg = (BG_TOP[None, None, :] * (1 - t) + BG_BOTTOM[None, None, :] * t)
    return np.repeat(bg, w, axis=1).copy()


def _base_color(mesh: trimesh.Trimesh) -> np.ndarray:
    """取网格自身的 PBR 底色（pipeline.paint 写入），失败则用中性灰蓝。"""
    try:
        mat = getattr(mesh.visual, "material", None)
        c = getattr(mat, "baseColorFactor", None)
        if c is not None:
            arr = np.asarray(c, dtype=float).reshape(-1)[:3]
            if arr.size == 3:
                # baseColorFactor 是线性值，转到 sRGB 显示空间
                return np.clip(np.where(arr <= 0.0031308, arr * 12.92,
                                        1.055 * np.power(arr, 1 / 2.4) - 0.055), 0, 1)
    except Exception:  # noqa: BLE001
        pass
    return np.array([0.52, 0.56, 0.60])


def _shade_faces(normals, view_dirs, base_rgb) -> np.ndarray:
    """逐面兰伯特光照 + 环境光 + 高光，返回 (F,3) 的 0~1 颜色。

    view_dirs：(F,3) 每个面指向相机的单位向量（正交投影时为常向量）。
    """
    key, fill, top = _unit(_KEY), _unit(_FILL), _unit(_TOP)
    view = view_dirs

    ndl_k = np.clip(normals @ key, 0, None)
    ndl_f = np.clip(normals @ fill, 0, None)
    ndl_t = np.clip(normals @ top, 0, None)

    # 半球环境光：朝上的面亮、朝下的面暗
    hemi = 0.5 + 0.5 * normals[:, 2:3]

    diff = (0.20 * hemi
            + 0.55 * ndl_k[:, None]
            + 0.12 * ndl_f[:, None] ** 1.4
            + 0.07 * ndl_t[:, None])
    color = base_rgb[None, :] * diff

    # 轻微镜面高光（Blinn-Phong），让圆角/斜面有金属感的层次
    half = _unit(key[None, :] + view)
    spec = np.clip(np.einsum("ij,ij->i", normals, half), 0, None) ** 28.0
    color = color + 0.14 * spec[:, None]

    # 边缘压暗，提升轮廓可读性（多视图语义评分更易识别结构）
    rim = np.clip(1.0 - np.clip(np.einsum("ij,ij->i", normals, view), 0, None),
                  0, 1) ** 3.0
    color = color * (1.0 - 0.16 * rim[:, None])

    return np.clip(color, 0.0, 1.0)


def _rasterize(uv, depth, faces, face_rgb, h, w, bg) -> np.ndarray:
    """z-buffer 光栅化。uv 为像素坐标 (V,2)，depth 为深度 (V,)，
    faces 为待绘制三角形索引 (n,3)，face_rgb 为与 faces 一一对应的颜色 (n,3)。"""
    img = bg
    zbuf = np.full((h, w), np.inf, dtype=float)

    for i, f in enumerate(faces):
        p = uv[f]
        d = depth[f]
        x0 = int(np.floor(p[:, 0].min()))
        x1 = int(np.ceil(p[:, 0].max())) + 1
        y0 = int(np.floor(p[:, 1].min()))
        y1 = int(np.ceil(p[:, 1].max())) + 1
        if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
            continue
        x0, x1 = max(x0, 0), min(x1, w)
        y0, y1 = max(y0, 0), min(y1, h)
        if x0 >= x1 or y0 >= y1:
            continue

        area = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(area) < 1e-9:
            continue
        inv = 1.0 / area

        xs = np.arange(x0, x1, dtype=float) + 0.5
        ys = np.arange(y0, y1, dtype=float) + 0.5
        X, Y = np.meshgrid(xs, ys)

        w0 = ((p[1, 0] - X) * (p[2, 1] - Y) - (p[2, 0] - X) * (p[1, 1] - Y)) * inv
        w1 = ((p[2, 0] - X) * (p[0, 1] - Y) - (p[0, 0] - X) * (p[2, 1] - Y)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * d[0] + w1 * d[1] + w2 * d[2]
        sub_z = zbuf[y0:y1, x0:x1]
        update = inside & (z < sub_z)
        if not update.any():
            continue
        sub_z[update] = z[update]
        sub_img = img[y0:y1, x0:x1]
        sub_img[update] = face_rgb[i] * 255.0

    return img


def render_view(mesh: trimesh.Trimesh, direction, up_hint, *,
                perspective: bool = False, size: int = DEFAULT_SIZE,
                ssaa: int = SSAA, base_rgb: Optional[np.ndarray] = None) -> np.ndarray:
    """渲染单个视角，返回 (size,size,3) 的 uint8 图像。"""
    V = np.asarray(mesh.vertices, dtype=float)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if len(F) == 0:
        raise ValueError("网格没有面")

    r, u, f = _basis(direction, up_hint)

    lo, hi = V.min(axis=0), V.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0

    if perspective:
        dist = 3.1 * max(radius, 1e-6)
        cam_pos = center - f * dist
        rel = V - cam_pos
        depth = rel @ f
        depth = np.maximum(depth, 1e-6 * max(radius, 1e-6))
        # 归一化屏幕坐标（再统一缩放，保证任意尺度都恰好填满画面）
        sx = (rel @ r) / depth
        sy = (rel @ u) / depth
    else:
        cam_pos = None
        rel = V - center
        depth = rel @ f
        sx = rel @ r
        sy = rel @ u

    span_w = float(sx.max() - sx.min())
    span_h = float(sy.max() - sy.min())
    span = max(span_w, span_h, 1e-9)
    W = size * ssaa
    scale = MARGIN * W / span
    cx = (sx.max() + sx.min()) / 2.0
    cy = (sy.max() + sy.min()) / 2.0

    uv = np.empty((len(V), 2), dtype=float)
    uv[:, 0] = (sx - cx) * scale + W / 2.0
    uv[:, 1] = W / 2.0 - (sy - cy) * scale

    # ---- 背面剔除（三维法向，正交/透视都适用）----
    tri = V[F]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = _unit(normals)
    centroids = tri.mean(axis=1)
    if perspective:
        to_cam = _unit(cam_pos[None, :] - centroids)
        visible = np.einsum("ij,ij->i", normals, to_cam) > 0
    else:
        to_cam = np.broadcast_to(-f, normals.shape)
        visible = (normals @ f) < 0
    keep = np.nonzero(visible)[0]
    if keep.size == 0:
        keep = np.arange(len(F))

    rgb = _shade_faces(normals, to_cam,
                       _base_color(mesh) if base_rgb is None else base_rgb)
    img = _rasterize(uv, depth, F[keep], rgb[keep], W, W, _background(W, W))

    # ---- 降采样（SSAA）----
    if ssaa > 1:
        from PIL import Image
        out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        out = out.resize((size, size), Image.LANCZOS)
        return np.asarray(out)
    return np.clip(img, 0, 255).astype(np.uint8)


VIEWS = {
    "front": dict(direction=(0.0, 1.0, 0.0), up_hint=(0.0, 0.0, 1.0), perspective=False),
    "side": dict(direction=(-1.0, 0.0, 0.0), up_hint=(0.0, 0.0, 1.0), perspective=False),
    "top": dict(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0), perspective=False),
    "perspective": dict(direction=(0.72, 0.66, -0.42), up_hint=(0.0, 0.0, 1.0),
                        perspective=True),
}


def render_four_views(mesh: trimesh.Trimesh, outdir, *, size: int = DEFAULT_SIZE,
                      verbose: bool = False) -> list[str]:
    """渲染 front/side/top/perspective 四张图到 outdir，返回写出的文件路径。

    自适应超采样：面数超过 4 万时关闭 2x SSAA，把单题渲染耗时压在 ~4 秒内
    （实测 hid_dec_010 有 7.6 万面，2x SSAA 会让该题渲染拖到 14 秒）。
    """
    from PIL import Image

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    base_rgb = _base_color(mesh)
    written: list[str] = []
    ssaa = SSAA if len(mesh.faces) <= 40000 else 1

    for name, cfg in VIEWS.items():
        img = render_view(mesh, cfg["direction"], cfg["up_hint"],
                          perspective=cfg["perspective"], size=size,
                          ssaa=ssaa, base_rgb=base_rgb)
        fp = out / f"{name}.png"
        Image.fromarray(img).save(fp, optimize=True)
        written.append(str(fp))
        if verbose:
            print(f"      [render] {name}.png {img.shape[1]}x{img.shape[0]}")
    return written


if __name__ == "__main__":
    import sys
    import time

    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/submission/hid_con_001/model.glb"
    dst = sys.argv[2] if len(sys.argv) > 2 else "work/offline_render"
    obj = trimesh.load(src, force="scene", process=False)
    geo = (list(obj.geometry.values())[0]
           if isinstance(obj, trimesh.Scene) else obj)
    t0 = time.time()
    files = render_four_views(geo, dst, verbose=True)
    print(f"耗时 {time.time() - t0:.2f}s -> {files}")
