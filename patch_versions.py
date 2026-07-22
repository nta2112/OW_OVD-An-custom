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
import re


def _patch_package(pkg_name: str, new_ceiling: str = "2.3.0") -> bool:
    """Return True if the file was patched."""
    spec = importlib.util.find_spec(pkg_name)
    if spec is None or not spec.origin:
        print(f"[patch_versions] {pkg_name} not found – skipping.")
        return False

    init_file = spec.origin
    print(f"[patch_versions] Found {pkg_name} at: {init_file}")

    with open(init_file, "r", encoding="utf-8") as f:
        src = f.read()

    print(f"[patch_versions] {pkg_name} lines containing 'mmcv_maximum_version':")
    for i, line in enumerate(src.splitlines()):
        if 'mmcv_maximum_version' in line:
            print(f"  Line {i+1}: {line}")

    # Regex pattern to match mmcv_maximum_version = '...' or "..."
    # Allowing spaces around =
    pattern = r"(mmcv_maximum_version\s*=\s*['\"])([^'\"]+)(['\"])"

    # Perform regex replacement on all matches
    new_src, count = re.subn(pattern, rf"\g<1>{new_ceiling}\g<3>", src)
    if count == 0:
        print(f"[patch_versions] {pkg_name}: 'mmcv_maximum_version' not found in file.")
        return False

    # Evict pycache on disk
    import shutil
    pycache_dir = os.path.join(os.path.dirname(init_file), "__pycache__")
    if os.path.exists(pycache_dir):
        try:
            shutil.rmtree(pycache_dir)
            print(f"[patch_versions] {pkg_name}: Evicted stale pycache directory.")
        except Exception as e:
            print(f"[patch_versions] {pkg_name}: Failed to evict pycache directory: {e}")

    if new_src == src:
        print(f"[patch_versions] {pkg_name}: ceiling is already '{new_ceiling}'. No patch needed.")
        return True

    with open(init_file, "w", encoding="utf-8") as f:
        f.write(new_src)

    print(f"[patch_versions] {pkg_name}: successfully patched {count} occurrence(s) to '{new_ceiling}'")
    return True


print("[patch_versions] Starting robust regex patch...")
_patch_package("mmyolo")
_patch_package("mmdet")

# Evict cached modules
_stale = [m for m in sys.modules if m.split(".")[0] in ("mmcv", "mmdet", "mmyolo", "mmengine")]
for _mod in _stale:
    del sys.modules[_mod]

print("[patch_versions] Done. You can now safely import mmcv / mmyolo / mmdet.")
