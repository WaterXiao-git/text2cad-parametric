"""
建模器注册表：导入全部类别模块后，把 builder_api.REGISTRY 暴露为 BUILDERS。

用法：
    import sys; sys.path.insert(0, 'src')
    from builders import BUILDERS
    mesh = BUILDERS['hid_con_001'](get_task('hid_con_001', 'test'))
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from builder_api import REGISTRY  # noqa: E402

_MODULES = [
    "builders_constrained",
    "builders_decorative",
    "builders_functional",
    "builders_small",
]

_loaded: list[str] = []
_failed: dict[str, str] = {}
for _m in _MODULES:
    try:
        importlib.import_module(_m)
        _loaded.append(_m)
    except Exception as _e:  # noqa: BLE001
        _failed[_m] = str(_e)

BUILDERS = dict(REGISTRY)


def status() -> str:
    """返回注册情况摘要，便于发现漏实现或语法错误的模块。"""
    lines = [f"已加载模块: {len(_loaded)}/{len(_MODULES)}"]
    for m, e in _failed.items():
        lines.append(f"  [!!] {m}: {e.splitlines()[-1] if e else '?'}")
    lines.append(f"已注册建模器: {len(BUILDERS)}")
    missing = []
    from taskspec import load_tasks
    for t in load_tasks("test"):
        if t.prompt_id not in BUILDERS:
            missing.append(t.prompt_id)
    if missing:
        lines.append(f"未实现 {len(missing)} 个: {', '.join(missing)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status())
