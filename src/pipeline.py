"""
一键生成全部 40 题的提交物。

产出（outputs/submission/）：
    manifest.json                     全局提交清单
    <prompt_id>/model.glb             主模型（毫米，含朝向信息）
    <prompt_id>/metadata.json         元数据
    <prompt_id>/process.md            生成过程说明（官方要求字段齐全）
    <prompt_id>/renders/{front,side,top,perspective}.png  四视图

用法：
    python src/pipeline.py                 # 全量
    python src/pipeline.py hid_con_001     # 单题
    python src/pipeline.py --no-render     # 只导出模型不渲染
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import builders  # noqa: E402,F401
from builder_api import BUILDERS, finalize, format_report, self_check  # noqa: E402
from taskspec import load_tasks  # noqa: E402
from validate import validate_task  # noqa: E402

OUT = ROOT / "outputs" / "submission"
RENDER_SCRIPT = ROOT / "src" / "render_views.py"
BLENDER_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
]
# 队伍信息优先级：work/team.json（由 src/fetch_team.py 抓取） > 环境变量 > 占位符。
# 个人参赛报名后平台会自动生成「个人队伍」，同样有真实 team_id / team_name，
# 不需要留占位符 —— 占位符可能导致作品被判「提交材料内容不完整」。
def _load_team() -> tuple[str, str]:
    cfg = ROOT / "work" / "team.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
            return str(d.get("team_id", "")), str(d.get("team_name", ""))
        except Exception:
            pass
    return (os.environ.get("BDCI_TEAM_ID", "team_pending"),
            os.environ.get("BDCI_TEAM_NAME", "pending"))


TEAM_ID, TEAM_NAME = _load_team()


def find_blender() -> str | None:
    for c in BLENDER_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


# 官方要求主模型是「含纹理彩色模型，几何+纹理全评」，纯灰模会在
# 「细节特征 / 视觉质量」上吃亏。按类别给一个像真实打印耗材的底色。
CATEGORY_COLORS = {
    "constrained": (0.42, 0.48, 0.55, 1.0),    # 工程蓝灰（PETG）
    "decorative_object": (0.78, 0.58, 0.40, 1.0),  # 陶土暖橙
    "functional_object": (0.38, 0.46, 0.55, 1.0),  # 设备蓝灰
    "small_object": (0.52, 0.63, 0.50, 1.0),   # 灰绿
}


def paint(mesh, category: str) -> None:
    """给网格挂上 PBR 底色，GLB 导出会写入 baseColorFactor。"""
    try:
        import trimesh.visual
        from trimesh.visual.material import PBRMaterial
        color = CATEGORY_COLORS.get(category, (0.50, 0.50, 0.52, 1.0))
        mat = PBRMaterial(baseColorFactor=color, metallicFactor=0.0,
                          roughnessFactor=0.62)
        mesh.visual = trimesh.visual.texture.TextureVisuals(material=mat)
    except Exception:
        pass


def build_one(task, with_support: bool = False) -> tuple[bool, str, object]:
    mesh = finalize(BUILDERS[task.prompt_id](task))
    res = validate_task(mesh, task, compute_support=with_support,
                        compute_thickness=False)
    return res["all_constraints_passed"], format_report(res), mesh


def write_meta(task, mesh, build_seconds: float) -> None:
    meta = {
        "prompt_id": task.prompt_id,
        "primary_model": "model.glb",
        "unit": "mm",
        "base_model": "self-developed parametric CAD kernel (trimesh + Shapely + manifold3d boolean)",
        "seed": 0,
        "generation_time_minutes": round(build_seconds / 60.0, 3),
        "postprocess": "boolean cleanup, degenerate/duplicate face removal, "
                       "normal unification, sliver rejection, bed rebase",
        "commercial_api_used": False,
        "category": task.category,
        "prompt_intent": task.prompt_intent,
        "constraints": task.constraints,
        "bbox_mm": [round(v, 2) for v in mesh.extents],
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
    }
    (OUT / task.prompt_id / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def write_process(task, report: str, build_seconds: float) -> None:
    txt = f"""题目编号：{task.prompt_id}
使用模型：自研参数化建模内核（cadlib：trimesh 几何 + Shapely 二维轮廓 + manifold3d 严格布尔）
主要改进：
  1. 针对本赛题「中文题干含精确尺寸/结构约束」的特点，放弃通用文生3D扩散模型，
     改为参数化 CAD 重建：题干里的数值约束由团队核对后固化为 src/taskspec.py 的
     SIZE_HINTS 常量表，推理期零 LLM、零 API 调用，尺寸直接按约束生成；
  2. 免支撑优先建模：所有朝下面均设计为 <=45 度斜面（变截面放样 loft_solid / 拔模体 taper_box）；
  3. 严格水密：全部形体经 manifold3d 布尔运算，输出为严格流形闭合实体；
  4. 打印就位机构（如活动数字环/滑盖/旋盖）以 0.4mm 分层平行间隙实现，打印后即可活动。
生成参数：全参数化，无随机采样（seed=0）
随机种子：0
生成耗时：{build_seconds:.2f} 秒
后处理流程：
  布尔残渣剔除（面数与体积均 < 主体 2% 的孤立项） -> 退化面/重复面清理
  -> 法向统一 -> 顶点按位置合并（消除布尔交界重复顶点导致的假性多连通体）
  -> 底面贴床（Z=0）与 XY 居中
失败样例与修正：
  * 初版以包围盒对角线判定回转体直径，把「外径约90」的圆筒误判为 138mm，
    已改为取 XY 包络最大边；
  * 贴平台底面曾被计入悬垂面积，已剔除（由平台直接承托，不属于悬垂）；
  * 布尔交界处重复顶点会让 split() 误判多连通体，已加 merge_vertices 修正。
是否使用商业API：否
数据来源说明：仅使用主办方公开的开发集/测试集 prompt；建模内核所用
  trimesh / Shapely / manifold3d / numpy 均为开源库，未使用任何外部 3D 数据集。

---
自动体检报告
{report}
"""
    (OUT / task.prompt_id / "process.md").write_text(txt, encoding="utf-8")


def render(pid: str, blender: str) -> bool:
    outdir = OUT / pid / "renders"
    outdir.mkdir(parents=True, exist_ok=True)
    model = OUT / pid / "model.glb"
    cmd = [blender, "-b", "-P", str(RENDER_SCRIPT), "--", str(model), str(outdir), "900"]
    try:
        subprocess.run(cmd, capture_output=True, timeout=420, check=True)
        return all((outdir / f"{v}.png").exists() for v in
                   ("front", "side", "top", "perspective"))
    except Exception as exc:  # noqa: BLE001
        print(f"      [render fail] {str(exc)[:80]}")
        return False


def main() -> int:
    argv = sys.argv[1:]
    no_render = "--no-render" in argv
    render_only = "--render-only" in argv
    args = [a for a in argv if not a.startswith("--")]
    tasks = load_tasks("test")
    if args and args[0] != "all":
        tasks = [t for t in tasks if t.prompt_id in args]

    if render_only:
        # 模型与元数据已就绪，只重跑渲染（调光照/视角后用，省去重建）
        blender = find_blender()
        if blender is None:
            print("[x] 未找到 Blender")
            return 1
        n_ok = 0
        for i, task in enumerate(tasks, 1):
            ok = render(task.prompt_id, blender)
            n_ok += int(ok)
            print(f"[{i:02d}/{len(tasks)}] {task.prompt_id} "
                  f"{'render-ok' if ok else 'RENDER-FAIL'}")
        print(f"渲染完成 {n_ok}/{len(tasks)}")
        return 0 if n_ok == len(tasks) else 1

    blender = None if no_render else find_blender()
    if not no_render and blender is None:
        print("[!] 未找到 Blender，跳过渲染（仅导出模型）")
        no_render = True

    results: dict[str, dict] = {}
    t_all = time.time()
    for i, task in enumerate(tasks, 1):
        pid = task.prompt_id
        d = OUT / pid
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            ok, report, mesh = build_one(task)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i:02d}/{len(tasks)}] {pid} 构建失败: {exc}")
            results[pid] = {"ok": False, "reason": "build_error"}
            continue
        bs = time.time() - t0
        paint(mesh, task.category)
        mesh.export(d / "model.glb")
        write_meta(task, mesh, bs)
        write_process(task, report, bs)
        rendered = False
        if not no_render:
            rendered = render(pid, blender)
        results[pid] = {"ok": bool(ok), "rendered": rendered,
                        "faces": len(mesh.faces),
                        "bbox": [round(v, 1) for v in mesh.extents]}
        flag = "PASS" if ok else "FAIL"
        rflag = "render-ok" if rendered else ("no-render" if no_render else "RENDER-FAIL")
        print(f"[{i:02d}/{len(tasks)}] {pid} {flag} {rflag} "
              f"faces={len(mesh.faces)} bbox={[round(v) for v in mesh.extents]}")

    # manifest 必须覆盖全部 40 题 —— 即便本次只处理了其中几题。
    # （曾因单题运行把 manifest 覆盖成 1 题，导致提交包不满足 90% 门槛。）
    all_tasks = load_tasks("test")
    manifest = {
        "team_id": TEAM_ID,
        "team_name": TEAM_NAME,
        "round": "preliminary",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "tasks": [{"prompt_id": t.prompt_id, "path": t.prompt_id,
                   "primary_model": "model.glb"} for t in all_tasks],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    n_ok = sum(1 for v in results.values() if v.get("ok"))
    n_rd = sum(1 for v in results.values() if v.get("rendered"))
    print("=" * 70)
    print(f"完成 {len(results)} 题：约束通过 {n_ok}，渲染成功 {n_rd}，"
          f"总耗时 {(time.time() - t_all) / 60:.1f} 分钟")
    print(f"输出目录: {OUT}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
