"""
FDM 可制造性（Printability）几何内核。

该模块不依赖任何赛题数据，是「文本 → 高可打印性网格」管线中的后处理与评分核心。

设计要点
--------
1. **单位与尺度**：AI 生成的网格通常是无量纲的（bbox 可能是 1.0 或 -1..1）。
   所有可打印性指标（悬垂角、最小壁厚、支撑体积）都依赖真实物理尺寸，
   因此必须先 `normalize_scale` 到毫米级，否则指标毫无意义。
2. **水密性优先**：切片器（Cura/PrusaSlicer）要求流形、水密网格。
   `repair_watertight` 先做拓扑修复，失败时退回到「点云 + 泊松重建」，
   泊松重建天然产出水密闭合曲面，是强兜底。
3. **朝向与语义的冲突**：最优打印朝向往往要大幅倾斜以消除悬垂，
   但语义评分（CLIP）希望物体「正着放」。因此 `OrientationSearch` 默认
   限制在语义竖直方向附近的小角度倾斜（`max_tilt_deg`）内搜索，
   在两者之间取平衡，而不是无约束全局旋转。

作者：为 CCF BDCI 2026「文本生成3D方案赛」编写
"""

from __future__ import annotations

import numpy as np
import trimesh

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

__all__ = [
    "PrintConfig",
    "normalize_scale",
    "repair_watertight",
    "mesh_topology_report",
    "overhang_mask",
    "overhang_area_ratio",
    "support_volume",
    "thickness_profile",
    "bed_contact",
    "printability_metrics",
    "printability_score",
]


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class PrintConfig:
    """FDM 工艺参数。默认值按 0.4mm 喷嘴的桌面机设定。"""

    nozzle_mm: float = 0.4
    layer_height_mm: float = 0.2
    # 自支撑临界角（相对水平面）。小于该角的朝下面需要支撑。
    max_overhang_deg: float = 45.0
    # 最小壁厚，经验值取 2 倍喷嘴直径（0.8mm），低于此值挤出不稳定。
    min_wall_mm: float = 0.8
    # 最小可分辨特征尺寸
    min_feature_mm: float = 0.4
    # 打印平台尺寸 (X, Y, Z)
    bed_size_mm: Tuple[float, float, float] = (220.0, 220.0, 250.0)
    # 归一化后的目标最大边长（毫米）
    target_size_mm: float = 100.0

    @property
    def cos_overhang(self) -> float:
        """
        悬垂判据阈值。

        面法向 n 与建造方向 up 的关系：
          - n·up >= 0            ：朝上表面，自支撑
          - n·up <  0            ：朝下表面，是否自支撑取决于倾角
        面与水平面夹角 alpha = acos(|n·up|)。
        朝下面自支撑条件为 alpha >= max_overhang_deg，
        即 |n·up| <= cos(max_overhang_deg)，即 n·up >= -cos(max_overhang_deg)。
        故悬垂(需支撑)条件：n·up < -cos(max_overhang_deg)。
        """
        return float(np.cos(np.deg2rad(self.max_overhang_deg)))


# --------------------------------------------------------------------------
# 尺度归一化
# --------------------------------------------------------------------------
def normalize_scale(
    mesh: trimesh.Trimesh,
    target_size_mm: float = 100.0,
    inplace: bool = False,
) -> trimesh.Trimesh:
    """
    将网格缩放到物理尺寸：最长边 = target_size_mm，并把底面贴到 Z=0、居中于 XY。

    无量纲的 AI 输出必须先过这一步，否则壁厚/悬垂等指标全部失真。
    """
    m = mesh if inplace else mesh.copy()
    extents = m.extents
    max_extent = float(np.max(extents))
    if max_extent <= 0 or not np.isfinite(max_extent):
        return m
    s = target_size_mm / max_extent

    # 先居中再缩放，保证变换可预测
    m.apply_translation(-m.centroid)
    m.apply_scale(s)

    # 落到平台上：最低点 Z=0，XY 居中
    m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0][2]])
    return m


# --------------------------------------------------------------------------
# 水密性修复
# --------------------------------------------------------------------------
def _pymeshlab_repair(mesh: trimesh.Trimesh, max_hole_size: int = 200) -> Optional[trimesh.Trimesh]:
    """用 MeshLab 滤镜做拓扑修复：去孤立点、修非流形、补洞、统一法向。"""
    try:
        import pymeshlab as pml
    except Exception:
        return None

    try:
        ms = pml.MeshSet()
        pm = pml.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm, "m")

        for fname, kwargs in (
            ("meshing_remove_unreferenced_vertices", {}),
            ("meshing_repair_non_manifold_vertices", {}),
            ("meshing_repair_non_manifold_edges", {}),
            ("meshing_close_holes", {"maxholesize": max_hole_size}),
            ("meshing_remove_unreferenced_vertices", {}),
            ("meshing_re_orient_faces_coherently", {}),
        ):
            try:
                ms.apply_filter(fname, **kwargs)
            except Exception:
                continue

        cur = ms.current_mesh()
        v = np.asarray(cur.vertex_matrix(), dtype=np.float64)
        f = np.asarray(cur.face_matrix(), dtype=np.int64)
        if len(v) < 4 or len(f) < 4:
            return None
        out = trimesh.Trimesh(vertices=v, faces=f, process=False)
        out.remove_unreferenced_vertices()
        return out
    except Exception:
        return None


def _poisson_reconstruct(mesh: trimesh.Trimesh, depth: int = 9,
                         samples: int = 200_000) -> Optional[trimesh.Trimesh]:
    """
    兜底：表面采样 → 泊松重建。

    泊松重建输出的是闭合的隐式等值面，天然水密，
    对于满身破洞、自相交的生成式网格是最可靠的重建手段。
    """
    try:
        import pymeshlab as pml
    except Exception:
        return None

    try:
        ms = pml.MeshSet()
        pm = pml.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm, "m")

        ms.apply_filter("generate_sampling_poisson_disk", samplenum=samples)
        # 泊松重建需要法向
        try:
            ms.apply_filter("compute_normal_for_point_clouds")
        except Exception:
            pass
        ms.apply_filter(
            "generate_surface_reconstruction_screened_poisson",
            depth=depth,
            samplespernode=1.5,
        )
        cur = ms.current_mesh()
        v = np.asarray(cur.vertex_matrix(), dtype=np.float64)
        f = np.asarray(cur.face_matrix(), dtype=np.int64)
        if len(v) < 4 or len(f) < 4:
            return None
        out = trimesh.Trimesh(vertices=v, faces=f, process=False)
        out.remove_unreferenced_vertices()
        return out
    except Exception:
        return None


def repair_watertight(
    mesh: trimesh.Trimesh,
    poisson_depth: int = 9,
    max_hole_size: int = 200,
    verbose: bool = False,
) -> Tuple[trimesh.Trimesh, dict]:
    """
    尽力产出水密、流形、单连通的网格。

    策略（逐级降级）：
      1. 已经是水密单连通 → 原样返回
      2. MeshLab 拓扑修复
      3. 泊松重建兜底

    返回 (mesh, info)；info 记录每一步的结果，便于写技术报告与定位问题。
    """
    info: dict = {"input_watertight": bool(mesh.is_watertight),
                  "input_components": len(mesh.split(only_watertight=False))}

    m = mesh.copy()
    m.remove_unreferenced_vertices()
    try:
        m.remove_degenerate_faces()
    except Exception:
        pass
    try:
        m.fix_normals()
    except Exception:
        pass

    if m.is_watertight and len(m.split(only_watertight=False)) == 1:
        info["method"] = "none_needed"
        return m, info

    # 步骤 2：拓扑修复
    rep = _pymeshlab_repair(m, max_hole_size=max_hole_size)
    if rep is not None and len(rep.faces) > 4:
        try:
            rep.fix_normals()
        except Exception:
            pass
        if rep.is_watertight:
            info["method"] = "pymeshlab_repair"
            info["watertight"] = True
            info["components"] = len(rep.split(only_watertight=False))
            return rep, info
        m = rep  # 继续兜底，用修复后的结果重建

    # 步骤 3：泊松重建兜底
    pois = _poisson_reconstruct(m, depth=poisson_depth)
    if pois is not None and len(pois.faces) > 4:
        try:
            pois.fix_normals()
        except Exception:
            pass
        info["method"] = "poisson"
        info["watertight"] = bool(pois.is_watertight)
        info["components"] = len(pois.split(only_watertight=False))
        return pois, info

    info["method"] = "failed"
    info["watertight"] = bool(m.is_watertight)
    return m, info


# --------------------------------------------------------------------------
# 拓扑质量
# --------------------------------------------------------------------------
def mesh_topology_report(mesh: trimesh.Trimesh) -> dict:
    """流形性 / 水密性 / 连通性 / 法向一致性等硬指标。"""
    comps = mesh.split(only_watertight=False)
    return {
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(getattr(mesh, "is_volume", False)),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "n_components": len(comps),
        "largest_component_frac": float(
            max((len(c.faces) for c in comps), default=0) / max(len(mesh.faces), 1)
        ),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else float("nan"),
        "area_mm2": float(mesh.area),
        "extents_mm": [float(x) for x in mesh.extents],
    }


# --------------------------------------------------------------------------
# 悬垂
# --------------------------------------------------------------------------
def overhang_mask(mesh: trimesh.Trimesh,
                  up: Iterable[float] = (0.0, 0.0, 1.0),
                  cfg: PrintConfig = PrintConfig(),
                  bed_eps: Optional[float] = None) -> np.ndarray:
    """
    标记需要支撑的三角面。

    判据（推导见 PrintConfig.cos_overhang）：
        面法向 n 满足 n·up < -cos(max_overhang_deg)  → 需支撑

    重要修正：**贴平台的面不算悬垂**。
    位于最低处（z <= zmin + bed_eps）的朝下面直接由打印平台承托，
    不需要支撑。不做这一步会把「平底盒子」的底面误判为 100% 悬垂，
    与真实切片结果严重不符。
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)
    n = np.asarray(mesh.face_normals, dtype=np.float64)

    if mesh.is_watertight and mesh.volume is not None and mesh.volume < 0:
        # 法向朝内时判据反转
        mask = (n @ up) > cfg.cos_overhang
    else:
        mask = (n @ up) < -cfg.cos_overhang

    if bed_eps is None:
        bed_eps = max(cfg.layer_height_mm, 1e-6)
    # 贴平台的面由平台承托，排除
    zc = np.asarray(mesh.triangles_center, dtype=np.float64) @ up
    if len(zc):
        mask &= zc > (float(zc.min()) + bed_eps)
    return mask


def overhang_area_ratio(mesh: trimesh.Trimesh,
                        up: Iterable[float] = (0.0, 0.0, 1.0),
                        cfg: PrintConfig = PrintConfig()) -> float:
    """悬垂面积占总表面积的比例，0=无需支撑，越小越好。"""
    mask = overhang_mask(mesh, up, cfg)
    a = np.asarray(mesh.area_faces, dtype=np.float64)
    total = float(a.sum())
    if total <= 0:
        return 1.0
    return float(a[mask].sum() / total)


# --------------------------------------------------------------------------
# 支撑体积（体素近似）
# --------------------------------------------------------------------------
def support_volume(mesh: trimesh.Trimesh,
                   up: Iterable[float] = (0.0, 0.0, 1.0),
                   cfg: PrintConfig = PrintConfig(),
                   pitch: Optional[float] = None,
                   max_voxels: int = 2_000_000) -> float:
    """
    估算需要生成的支撑体积（mm^3）。

    做法：把模型体素化，对每个悬垂面，沿 -up 方向向下逐体素行进，
    直到遇到实体或平台，途经的空体素计入支撑体积。
    这是「支撑体积」的标准近似，比单纯用悬垂面积更贴近实际耗材与后处理成本。
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    if pitch is None:
        pitch = max(cfg.nozzle_mm * 1.5, 0.6)

    try:
        vox = mesh.voxelized(pitch=pitch)
    except Exception:
        return float("nan")
    mat = vox.matrix
    if mat.size > max_voxels:
        # 退回到更粗的体素，保证在笔记本 CPU 上也算得动
        pitch = pitch * (mat.size / max_voxels) ** (1 / 3)
        vox = mesh.voxelized(pitch=pitch)
        mat = vox.matrix

    if not mat.any():
        return 0.0

    mask = overhang_mask(mesh, up, cfg)
    if not mask.any():
        return 0.0

    # 悬垂面重心 → 体素索引
    face_centers = mesh.triangles_center[mask]
    idx_f = vox.points_to_indices(face_centers)
    # 只保留唯一体素，避免重复累计
    try:
        uniq = np.unique(idx_f, axis=0)
    except Exception:
        uniq = idx_f

    # 沿 -up 方向（体素网格里即向下）行进
    # 用整数步进：把 up 量化到体素网格主轴
    axis = int(np.argmax(np.abs(up)))
    step = int(np.sign(up[axis]))
    if step == 0:
        step = 1

    H = mat.shape[axis]
    support = np.zeros(mat.shape, dtype=bool)
    counts = np.zeros(mat.shape, dtype=np.int32)

    for vi in uniq:
        i, j, k = vi
        cur = [i, j, k]
        n_step = 0
        while 0 <= cur[axis] < H:
            cur[axis] -= step
            if not (0 <= cur[axis] < H):
                break
            if mat[tuple(cur)]:
                break  # 撞到实体，支撑到此为止
            ci = tuple(cur)
            if not support[ci]:
                support[ci] = True
            counts[ci] += 1
            n_step += 1
            if n_step > H:
                break

    cell_vol = float(pitch ** 3)
    return float(support.sum() * cell_vol)


# --------------------------------------------------------------------------
# 壁厚分析
# --------------------------------------------------------------------------
def thickness_profile(mesh: trimesh.Trimesh,
                      n_samples: int = 4000,
                      max_ray_mm: Optional[float] = None,
                      rng: Optional[np.random.Generator] = None) -> dict:
    """
    估算壁厚分布：从表面点沿 -法向发射射线，量到对面（内壁）的距离。

    返回统计量与「过薄顶点比例」。过薄意味着 FDM 挤出时无法成形或强度不足。
    """
    rng = rng or np.random.default_rng(0)
    n = min(n_samples, max(len(mesh.vertices), 1))
    sel = rng.choice(len(mesh.vertices), size=n, replace=False)

    pts = np.asarray(mesh.vertices, dtype=np.float64)[sel]
    nrm = np.asarray(mesh.vertex_normals, dtype=np.float64)[sel]

    if max_ray_mm is None:
        max_ray_mm = float(np.max(mesh.extents)) * 1.5 + 1.0

    origins = pts + nrm * 1e-4          # 略微抬起，避免自命中
    dirs = -nrm

    thick = np.full(n, np.nan)
    try:
        locs, ray_idx, tri_idx = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=dirs, multiple_hits=False
        )
        if len(ray_idx) > 0:
            d = np.linalg.norm(locs - origins[ray_idx], axis=1)
            # 同一条射线可能命中多次，取最近的一次（真正的内壁）
            order = np.lexsort((d, ray_idx))
            ray_idx_s, d_s = ray_idx[order], d[order]
            first = np.ones(len(ray_idx_s), dtype=bool)
            first[1:] = ray_idx_s[1:] != ray_idx_s[:-1]
            thick[ray_idx_s[first]] = d_s[first]
    except Exception:
        pass

    # 未命中的（开放网格/射线逃逸）按最大厚度处理，避免误判
    thick = np.where(np.isnan(thick), max_ray_mm, thick)

    cfg_min = PrintConfig().min_wall_mm
    return {
        "thickness_samples": int(n),
        "thickness_min": float(np.min(thick)),
        "thickness_p05": float(np.percentile(thick, 5)),
        "thickness_median": float(np.median(thick)),
        "thin_fraction": float(np.mean(thick < cfg_min)),
        "min_wall_mm": cfg_min,
        "thickness_hist": np.histogram(thick, bins=16)[0].tolist(),
    }


# --------------------------------------------------------------------------
# 平台接触与稳定性
# --------------------------------------------------------------------------
def bed_contact(mesh: trimesh.Trimesh,
                up: Iterable[float] = (0.0, 0.0, 1.0),
                tol_mm: float = 0.5) -> dict:
    """
    底面接触情况：接触面积、接触凸包面积、质心是否落在支撑多边形内。

    FDM 首层附着与抗倾倒依赖足够大的接触面；质心投影一旦越界就会打印失败。
    """
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    z = np.asarray(mesh.vertices, dtype=np.float64) @ up
    zmin = float(z.min())
    low = z <= zmin + tol_mm

    contact_pts = np.asarray(mesh.vertices, dtype=np.float64)[low]
    if len(contact_pts) < 3:
        return {"contact_area_mm2": 0.0, "hull_area_mm2": 0.0,
                "com_inside": False, "n_contact_verts": int(low.sum())}

    # 投影到垂直于 up 的平面，算二维凸包面积
    basis_u = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(basis_u) < 1e-6:
        basis_u = np.cross(up, [0.0, 1.0, 0.0])
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(up, basis_u)

    p2 = np.column_stack([contact_pts @ basis_u, contact_pts @ basis_v])

    contact_area = 0.0
    try:
        f_low = low[mesh.faces].all(axis=1)
        if f_low.any():
            contact_area = float(mesh.area_faces[f_low].sum())
    except Exception:
        pass

    hull_area = 0.0
    com_inside = False
    try:
        from scipy.spatial import ConvexHull, Delaunay
        hull = ConvexHull(p2)
        hull_area = float(hull.volume)  # 二维 ConvexHull 的 volume 即面积
        com2 = np.array([mesh.center_mass @ basis_u, mesh.center_mass @ basis_v])
        tri = Delaunay(p2[hull.vertices])
        com_inside = bool(tri.find_simplex(com2) >= 0)
    except Exception:
        pass

    return {
        "contact_area_mm2": float(contact_area),
        "hull_area_mm2": float(hull_area),
        "com_inside": com_inside,
        "n_contact_verts": int(low.sum()),
    }


# --------------------------------------------------------------------------
# 综合指标与评分
# --------------------------------------------------------------------------
def printability_metrics(mesh: trimesh.Trimesh,
                         up: Iterable[float] = (0.0, 0.0, 1.0),
                         cfg: PrintConfig = PrintConfig(),
                         compute_support: bool = True,
                         compute_thickness: bool = True) -> dict:
    """汇总全部可打印性指标。"""
    up = np.asarray(up, dtype=np.float64)
    up = up / (np.linalg.norm(up) + 1e-12)

    rep = mesh_topology_report(mesh)
    rep["overhang_area_ratio"] = overhang_area_ratio(mesh, up, cfg)
    rep["n_overhang_faces"] = int(overhang_mask(mesh, up, cfg).sum())

    if compute_support:
        sv = support_volume(mesh, up, cfg)
        rep["support_volume_mm3"] = float(sv)
        vol = rep.get("volume_mm3", float("nan"))
        rep["support_ratio"] = float(sv / vol) if (vol and np.isfinite(vol) and vol > 0) else float("nan")
    else:
        rep["support_volume_mm3"] = float("nan")
        rep["support_ratio"] = float("nan")

    if compute_thickness:
        rep.update(thickness_profile(mesh))

    rep.update({f"bed_{k}": v for k, v in bed_contact(mesh, up).items()})

    h = float((np.asarray(mesh.vertices) @ up).ptp())
    rep["height_mm"] = h
    fits = (mesh.extents[0] <= cfg.bed_size_mm[0]
            and mesh.extents[1] <= cfg.bed_size_mm[1]
            and h <= cfg.bed_size_mm[2])
    rep["fits_bed"] = bool(fits)
    return rep


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def printability_score(metrics: dict, cfg: PrintConfig = PrintConfig()) -> Tuple[float, dict]:
    """
    把各项指标折算成 0..1 的综合可打印性分数（越高越好）。

    权重设计思路：
      - 硬约束（水密/单连通/可放入平台）不达标会直接导致打印失败，权重最高
      - 悬垂与支撑体量决定后处理成本与失败率，是打印性的主要矛盾
      - 壁厚、底面附着、稳定性为次要但不可忽略项
    由于官方精确权重未知，这里采用「先满足硬约束、再优化主要矛盾」的稳健配比。
    """
    w = {
        "watertight": 0.20,
        "single_component": 0.10,
        "overhang": 0.24,
        "support": 0.14,
        "thickness": 0.12,
        "bed": 0.10,
        "stability": 0.06,
        "fits_bed": 0.04,
    }

    s_watertight = 1.0 if metrics.get("watertight") else 0.0
    s_single = 1.0 if metrics.get("n_components", 9) <= 1 else _clip01(
        1.0 - (metrics["n_components"] - 1) / 4.0
    )

    ohr = metrics.get("overhang_area_ratio", 1.0)
    s_overhang = _clip01(1.0 - ohr / 0.25)  # 悬垂面积占比 25% 计 0 分

    sr = metrics.get("support_ratio", float("nan"))
    if sr is None or not np.isfinite(sr):
        s_support = _clip01(1.0 - ohr)  # 退化情况用悬垂比例近似
    else:
        s_support = _clip01(1.0 - sr / 0.60)

    tf = metrics.get("thin_fraction", 0.0)
    s_thickness = _clip01(1.0 - tf / 0.30)

    ca = metrics.get("bed_contact_area_mm2", 0.0) or 0.0
    ha = metrics.get("bed_hull_area_mm2", 0.0) or 0.0
    denom = max(ha, 1e-6)
    s_bed = _clip01(ca / denom) if ha > 0 else 0.0

    s_stab = 1.0 if metrics.get("bed_com_inside") else 0.0
    s_fits = 1.0 if metrics.get("fits_bed") else 0.0

    parts = {
        "watertight": s_watertight,
        "single_component": s_single,
        "overhang": s_overhang,
        "support": s_support,
        "thickness": s_thickness,
        "bed": s_bed,
        "stability": s_stab,
        "fits_bed": s_fits,
    }
    total = float(sum(w[k] * parts[k] for k in w))
    return total, parts
