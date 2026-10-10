"""
四视图渲染（在 Blender 内以 `blender -b -P render_views.py -- <model> <outdir>` 运行）。

为什么单独用 Blender 渲染：
    赛题明确要求提交 front/side/top/perspective 四张渲染图，且初赛的
    「多视图语义与视觉」环节是用**多模态大模型看图打分**。也就是说，
    这四张图就是语义分的直接输入，渲染质量会实打实影响得分。
    open3d 的离屏渲染在 Windows 上不支持 EGL，因此改用 Blender 的 Cycles
    做离线渲染：光照、材质、抗锯齿都远好于简易光栅化。

用法:
    blender -b -P src/render_views.py -- model.glb outdir 1024
"""

import os
import sys

import bpy


# --------------------------------------------------------------------------
# 参数解析（Blender 的自定义参数放在 `--` 之后）
# --------------------------------------------------------------------------
def parse_args():
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
    else:
        args = []
    model = args[0] if len(args) > 0 else None
    outdir = args[1] if len(args) > 1 else "renders"
    size = int(args[2]) if len(args) > 2 else 1024

    # 必须转成绝对路径：Blender 对相对路径会按「当前驱动器 + blend 文件目录」解析，
    # 实测会把 work/renders 解析到 C:\work\renders 而不是项目目录下。
    if model:
        model = os.path.abspath(model)
    outdir = os.path.abspath(outdir)
    return model, outdir, size


# --------------------------------------------------------------------------
# 场景搭建
# --------------------------------------------------------------------------
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_model(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".glb" or ext == ".gltf":
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    else:
        raise ValueError(f"不支持的模型格式: {ext}")

    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError("导入后没有找到网格对象")
    return objs


def _world_bbox(objs):
    import mathutils
    mn = [1e9] * 3
    mx = [-1e9] * 3
    for o in objs:
        for c in o.bound_box:
            wc = o.matrix_world @ mathutils.Vector(c)
            for i in range(3):
                mn[i] = min(mn[i], wc[i])
                mx[i] = max(mx[i], wc[i])
    return mn, mx


def normalize_object(objs, target_size: float = 1.6):
    """
    修正朝向 + 居中 + 缩放到统一包围球，保证四个视角构图一致。

    两个必须处理的坑（实测确认）：

    1) **轴向问题**。trimesh 导出的 GLB 是 Y-up，Blender 的 gltf 导入器
       实测映射为 blender = (x, z, -y)，即模型会"躺倒"：
       trimesh 的 (X=10, Y=20, Z=100) 导入后变成 (X=10, Y=100, Z=20)。
       若不修正，top 视图会拍到侧面，语义分直接崩。
       绕 X 轴 +90° 可还原：(x, z, -y) → (x, y, z)。

    2) **矩阵次序问题**。Blender 的 matrix_world = T @ R @ S，
       即"先缩放后平移"。所以若想让世界包围盒居中，
       平移量必须是 T = -S * center，而不是 -center
       （后者在缩放后会残留偏移，实测把物体甩到 Y∈[18,19]）。

    注意：这里只影响渲染取景，**不改动提交模型本身的毫米尺寸**。
    """
    import math

    # 1) 复位 + 修正朝向
    #    关键坑：glTF 导入器给对象的 rotation_mode 是 QUATERNION，
    #    直接给 rotation_euler 赋值会被静默忽略（世界包围盒前后完全不变），
    #    必须先切到 'XYZ' 再赋值。
    for o in objs:
        o.location = (0.0, 0.0, 0.0)
        o.rotation_mode = "XYZ"
        # glTF 导入后 trimesh 轴到 Blender 轴的映射是 (x, -z, y)：
        # 再绕 X 转 -90° 才能把 trimesh 的 +Z（打印朝向）转回 Blender 的 +Z。
        # 用 +90° 会让所有模型上下颠倒（实测 hid_con_007 的挂钩跑到了下半部）。
        o.rotation_euler = (-math.pi / 2, 0.0, 0.0)  # Y-up -> Z-up
        o.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

    # 2) 在"已修正朝向、T=0、S=1"状态下量包围盒，此时世界坐标 = 旋转后的局部坐标
    mn, mx = _world_bbox(objs)
    center = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
    ext = [mx[i] - mn[i] for i in range(3)]
    max_ext = max(ext) or 1.0
    s = target_size / max_ext

    # 3) 先缩放，再按 T = -S * center 平移
    for o in objs:
        o.scale = (s, s, s)
        o.location = (-s * center[0], -s * center[1], -s * center[2])
    bpy.context.view_layer.update()

    # 4) 底面抬到 Z=0，便于 top/front 取景
    mn2, _ = _world_bbox(objs)
    for o in objs:
        o.location.z -= mn2[2]
    bpy.context.view_layer.update()

    return max_ext


def setup_camera_and_lights(radius: float = 3.4):
    """三点布光 + 相机，突出轮廓与细节层次。

    能量比早期版本低很多：GLB 导入后默认是近乎纯白的材质，
    再叠加 0.92 的白色世界背景，画面会整体过曝、所有细节被洗掉，
    多视图语义评分（大模型看图）会直接失效。
    """
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.type = "PERSP"
    cam.data.lens = 50

    def add_light(name, typ, loc, energy, size=2.0):
        d = bpy.data.lights.new(name, type=typ)
        d.energy = energy
        if typ == "AREA":
            d.size = size
        o = bpy.data.objects.new(name, d)
        bpy.context.scene.collection.objects.link(o)
        o.location = loc
        return o

    add_light("Key", "AREA", (2.6, -2.4, 3.2), 110, 4.0)
    add_light("Fill", "AREA", (-3.0, -1.6, 2.0), 45, 5.0)
    add_light("Rim", "AREA", (-1.0, 2.8, 2.6), 60, 3.0)

    return cam


def assign_material(objs):
    """给导入的网格统一赋中灰蓝材质。

    纯白默认材质 + 强光会把四视图洗成一片白，大模型完全无法辨认结构；
    中等明度 + 中等粗糙度能同时保住轮廓、棱线与凹槽层次。
    """
    mat = bpy.data.materials.new("PrintMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.36, 0.41, 0.47, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.58
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    for o in objs:
        if o.data.materials:
            o.data.materials.clear()
        o.data.materials.append(mat)


def _imported_base_color(objs):
    """读出 GLB 自带的 PBR 底色（pipeline.py 已按类别写入）。"""
    for o in objs:
        for m in (o.data.materials or []):
            if not (m and m.use_nodes):
                continue
            b = m.node_tree.nodes.get("Principled BSDF")
            if b:
                c = b.inputs["Base Color"].default_value
                return (c[0], c[1], c[2], 1.0)
    return None


def _variant_material(objs, index: int = 0):
    """
    给导入的网格赋材质。

    优先沿用 GLB 自带的类别底色；同一题的多个独立体（打印就位机构）
    用明度区分开，让装配关系在四视图里一眼可辨。
    """
    base = _imported_base_color(objs) or (0.40, 0.45, 0.52, 1.0)

    def shade(rgb, k):
        return (min(1.0, rgb[0] * k), min(1.0, rgb[1] * k),
                min(1.0, rgb[2] * k), 1.0)

    variants = [base, shade(base, 1.45), shade(base, 0.62)]
    mats = []
    for i, col in enumerate(variants):
        m = bpy.data.materials.new(f"PrintMat{i}")
        m.use_nodes = True
        b = m.node_tree.nodes.get("Principled BSDF")
        if b:
            b.inputs["Base Color"].default_value = col
            b.inputs["Roughness"].default_value = 0.58
            b.inputs["Metallic"].default_value = 0.0
        mats.append(m)

    seen: dict[str, int] = {}
    for o in objs:
        key = o.name.split(".")[0]
        if key not in seen:
            seen[key] = len(seen) % len(mats)
        o.data.materials.clear()
        o.data.materials.append(mats[seen[key]])


def setup_world():
    """中性浅灰背景（略降亮度，避免与浅色物体糊在一起）。"""
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.80, 0.82, 0.85, 1.0)
        bg.inputs[1].default_value = 1.0


def setup_render(size: int):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 96
    sc.cycles.use_denoising = True
    sc.render.resolution_x = size
    sc.render.resolution_y = size
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.render.image_settings.color_depth = "8"
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = -0.35
    sc.render.use_persistent_data = True


# --------------------------------------------------------------------------
# 视角定义：统一四视图
# --------------------------------------------------------------------------
def look_at(cam, target, dist, theta, phi):
    """球坐标定位相机并朝向目标。theta=方位角, phi=俯仰角(度)。"""
    import math
    t = math.radians(theta)
    p = math.radians(phi)
    x = target[0] + dist * math.cos(p) * math.cos(t)
    y = target[1] + dist * math.cos(p) * math.sin(t)
    z = target[2] + dist * math.sin(p)
    cam.location = (x, y, z)

    import mathutils
    direction = mathutils.Vector(target) - mathutils.Vector(cam.location)
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()


VIEWS = {
    # 名称: (方位角theta, 俯仰角phi)
    "front": (-90.0, 0.0),         # 正前
    "side": (0.0, 0.0),            # 正侧
    "top": (-90.0, 89.9),          # 正上（略偏避免万向锁）
    # 俯仰角从 28° 提到 40°：赛题里大量模型的关键特征（走线槽、抽屉凹位、
    # 轨道、栅格、凹槽阵列）都开在顶面，俯仰太低会被侧壁挡住看不见，
    # 多视图语义评分就认不出题干描述的结构。
    "perspective": (-55.0, 40.0),  # 三分之四视角（偏俯视）
}


def main():
    model, outdir, size = parse_args()
    if not model:
        print("[x] 用法: blender -b -P render_views.py -- <model> <outdir> [size]")
        return

    clear_scene()
    objs = import_model(model)
    max_ext = normalize_object(objs)
    try:
        _variant_material(objs)
    except Exception:
        assign_material(objs)
    cam = setup_camera_and_lights()
    setup_world()
    setup_render(size)

    os.makedirs(outdir, exist_ok=True)

    # 取景中心：物体归一化后高度中心
    import mathutils
    min_z, max_z = 1e9, -1e9
    for o in objs:
        for c in o.bound_box:
            wc = o.matrix_world @ mathutils.Vector(c)
            min_z = min(min_z, wc[2])
            max_z = max(max_z, wc[2])
    target = (0.0, 0.0, (min_z + max_z) / 2.0)

    for name, (theta, phi) in VIEWS.items():
        look_at(cam, target, dist=3.2, theta=theta, phi=phi)
        out = os.path.join(outdir, f"{name}.png")
        bpy.context.scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print(f"[ok] {name} -> {out}")

    print("DONE")


if __name__ == "__main__":
    main()
