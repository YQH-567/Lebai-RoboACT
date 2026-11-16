"""
运行方法：
conda activate env_isaaclab
cd /home/l/isaac_space_01/act_manipulation
python scripts/record/initial_position_test.py
这个文件是用来检查数据集里面的初始位置的，主要是检查小方块的初始位置是否正确
"""


from pathlib import Path
import h5py
import numpy as np

dataset_dir = Path("datasets/raw")

for path in sorted(dataset_dir.glob("episode_*.hdf5")):
    with h5py.File(path, "r") as f:
        pos = f["object_position"][0]
        print(
            path.name,
            "initial position:",
            np.round(pos, 4)
        )