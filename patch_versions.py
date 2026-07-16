"""
patch_versions.py
=================
Patches mmyolo (and mmdet) __init__.py files to relax the mmcv version
ceiling so that MMCV 2.2.x is accepted.

Usage inside a Kaggle/Colab notebook:
    %run patch_versions.py

Run this cell BEFORE importing mmcv, mmdet, or mmyolo.
After running, the patched modules will be freshly imported.
"""

import sys
import importlib
import importlib.util


def _patch_package(pkg_name: str, old_ceiling: str, new_ceiling: str = "2.3.0") -> bool:
    """Return True if the file was patched."""
    spec = importlib.util.find_spec(pkg_name)
    if spec is None or not spec.origin:
        print(f"[patch_versions] {pkg_name} not found – skipping.")
        return False

    init_file = spec.origin
    with open(init_file, "r", encoding="utf-8") as f:
        src = f.read()

    old_str = f"mmcv_maximum_version = '{old_ceiling}'"
    new_str = f"mmcv_maximum_version = '{new_ceiling}'"

    if old_str not in src:
        print(f"[patch_versions] {pkg_name}: ceiling '{old_ceiling}' not found – already patched or different version.")
        return False

    src = src.replace(old_str, new_str)
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"[patch_versions] {pkg_name}: patched ceiling {old_ceiling!r} -> {new_ceiling!r}  ({init_file})")
    return True


# ── Patch both mmyolo and mmdet ──────────────────────────────────────────────
_patch_package("mmyolo", old_ceiling="2.1.0")   # mmyolo default ceiling
_patch_package("mmyolo", old_ceiling="2.2.0")   # in case a newer mmyolo wheel ships 2.2.0
_patch_package("mmdet",  old_ceiling="2.2.0")   # mmdet sometimes has this ceiling
_patch_package("mmdet",  old_ceiling="2.1.0")

# ── Evict stale cached modules so fresh import picks up the patch ─────────────
_stale = [m for m in sys.modules if m.split(".")[0] in ("mmcv", "mmdet", "mmyolo", "mmengine")]
for _mod in _stale:
    del sys.modules[_mod]

print("[patch_versions] Evicted stale modules:", _stale or "(none)")
print("[patch_versions] Done. You can now safely import mmcv / mmyolo / mmdet.")
