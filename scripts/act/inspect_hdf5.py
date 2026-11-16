"""
运行方式：
cd ~/act_manipulation
conda activate env_act
python scripts/act/inspect_hdf5.py
检查hpf5数据的结构
"""

from pathlib import Path
import h5py
import numpy as np

DATA_DIR = Path("datasets/raw")   #小心自己的路径问题

files = sorted(DATA_DIR.glob("episode_*.hdf5"))

print(f"Found {len(files)} episodes")

if not files:
    raise RuntimeError("No HDF5 files found!")

# 检查第一条 Episode 的完整结构
path = files[0]

with h5py.File(path, "r") as f:
    print(f"\n========== {path.name} ==========")

    print("\nRoot attributes:")
    for key, value in f.attrs.items():
        print(key, value)

    print("\nDatasets:")

    def inspect(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(
                f"{name}: "
                f"shape={obj.shape}, "
                f"dtype={obj.dtype}"
            )

            if obj.size <= 30:
                print("  sample:", obj[()])

    f.visititems(inspect)

# 检查所有 Episode 的帧数
print("\n========== Episode Statistics ==========")

for path in files:
    with h5py.File(path, "r") as f:
        print(f"\n{path.name}")

        for key in ["action", "observations/qpos"]:
            if key in f:
                print(key, f[key].shape)

print("\nInspection completed.")


