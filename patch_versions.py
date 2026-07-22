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

    # Regex pattern to match mmcv_maximum_version = '...' or "..."
    # Allowing spaces around =
    pattern = r"(mmcv_maximum_version\s*=\s*['\"])([^'\"]+)(['\"])"
    match = re.search(pattern, src)
    if not match:
        print(f"[patch_versions] {pkg_name}: 'mmcv_maximum_version' not found in file.")
        return False

    current_ceiling = match.group(2)
    if current_ceiling == new_ceiling:
        print(f"[patch_versions] {pkg_name}: ceiling is already '{new_ceiling}'. No patch needed.")
        return True

    # Perform regex replacement
    new_src = re.sub(pattern, rf"\g<1>{new_ceiling}\g<3>", src)
    with open(init_file, "w", encoding="utf-8") as f:
        f.write(new_src)

    print(f"[patch_versions] {pkg_name}: successfully patched ceiling '{current_ceiling}' -> '{new_ceiling}'")
    return True


print("[patch_versions] Starting robust regex patch...")
_patch_package("mmyolo")
_patch_package("mmdet")

# Evict cached modules
_stale = [m for m in sys.modules if m.split(".")[0] in ("mmcv", "mmdet", "mmyolo", "mmengine")]
for _mod in _stale:
    del sys.modules[_mod]

print("[patch_versions] Done. You can now safely import mmcv / mmyolo / mmdet.")
