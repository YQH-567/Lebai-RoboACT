"""
运行方式：
conda activate env_isaaclab
cd /home/l/isaac_space_01/act_manipulation
python scripts/record/inspect_dataset.py
文件说明：主要用来检查两个相机开始到中间到结束的分别三张图片的
"""



import h5py
import matplotlib.pyplot as plt

path = "datasets/raw/episode_000000.hdf5"

with h5py.File(path, "r") as f:
    wrist = f["observations/images/wrist"]
    front = f["observations/images/front"]
    actions = f["action"]

    print("Wrist:", wrist.shape, wrist.dtype)
    print("Front:", front.shape, front.dtype)
    print("Actions:", actions.shape)
    print("FSM Done:", f.attrs["fsm_done"])
    print("Success:", f.attrs["success"])

    # 查看开始、中间、结束三个时刻
    indices = [0, len(actions) // 2, len(actions) - 1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 5))

    for i, idx in enumerate(indices):
        axes[0, i].imshow(wrist[idx])
        axes[0, i].set_title(f"Wrist: frame {idx}")
        axes[0, i].axis("off")

        axes[1, i].imshow(front[idx])
        axes[1, i].set_title(f"Front: frame {idx}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("datasets/episode_preview.png", dpi=150)
    plt.show()
