"""
Script to run an environment with a pick and lift state machine.

The original Warp pick/place state machine is preserved. RGB, metric depth,
robot state, actions, and success diagnostics are sampled synchronously at 25Hz.

.. code-block:: bash

   python scripts/record/scripted_act_data_colletion_25hz.py --target-episodes 200
"""

"""Launch Omniverse Toolkit first."""

import argparse
import math
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sample-fps", type=int, default=25, help="Synchronized camera/state/action recording frequency.")
parser.add_argument("--target-episodes", type=int, default=200, help="Number of FSM-completed episodes to collect.")
parser.add_argument("--save-dir", type=str, default="datasets/raw_25hz", help="Output directory for new HDF5 episodes.")
parser.add_argument("--max-ee-speed", type=float, default=0.50, help="Cartesian target speed limit in m/s.")
parser.add_argument("--success-stable-seconds", type=float, default=0.5, help="Required stable placement duration; default matches 5 frames at the previous 10Hz evaluation rate.")
parser.add_argument("--tray-center-x", type=float, default=0.0)
parser.add_argument("--tray-center-y", type=float, default=-0.3)
parser.add_argument("--tray-half-x", type=float, default=0.12)
parser.add_argument("--tray-half-y", type=float, default=0.12)
parser.add_argument("--object-z-min", type=float, default=0.025)
parser.add_argument("--object-z-max", type=float, default=0.20)
parser.add_argument("--object-speed-threshold", type=float, default=0.04, help="Maximum object displacement speed in m/s for a settled placement.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

#开启相机
# 启用相机渲染扩展
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

#import IsaacLab.source.isaaclab_tasks  # noqa: F401
from act_manipulation.tasks.manager_based.lift.lift_env_cfg import LiftEnvCfg
#from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# 25Hz HDF5 recorder with RGB + metric depth.
from demo_recorder_25hz import DemoRecorder

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    PLACE_OBJECT = wp.constant(5)   # 放置后抬起
    RELEASE_OBJECT = wp.constant(6)  #末端保持在放置点，夹爪打开；单独计时，确保物体有足够时间释放。
    RETREAT = wp.constant(7) #夹爪保持打开，末端移动到放置点上方的安全位置。
    DONE = wp.constant(8)    #增加的完成状态


class PickSmWaitTime:
    """切换前状态的额外等待时间（单位为秒）."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.8)  # 原本是0.5
    APPROACH_OBJECT = wp.constant(0.6)    # 原本是0.6
    GRASP_OBJECT = wp.constant(1.0)   # 👈 这里就是闭合夹爪的等待时间（秒）原本是0.3秒
    LIFT_OBJECT = wp.constant(1.0)   # 原本是1.0秒
    PLACE_OBJECT = wp.constant(1.2)
    RELEASE_OBJECT = wp.constant(1.0)
    RETREAT = wp.constant(0.5)
    DONE = wp.constant(0.2)   #怎么都没进入这个DONE的状态，奇了怪了



@wp.func  #@wp.func：内核内部使用的辅助函数
# distance_below_threshold() 负责判断末端和目标的距离是否小于阈值。
# 你要能解释输入、输出和判断逻辑，不需要自己实现 Warp 的向量运算库
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel    #并行执行的入口
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),

    place_ee_pose: wp.array(dtype=wp.transform),  #加的放置点，11个数里面的倒数第三个哈，别对应错了

    offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    # 获取线程ID
    tid = wp.tid()
    # 获取状态机状态
    state = sm_state[tid]
    # 定义每个状态
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # 等待一会儿
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # 进入下一个状态并重置等待时间
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_ABOVE_OBJECT:   #这里修正一下应该等待的是自己APPROACH_ABOVE_OBJECT设置的时间
                # 进入下一个状态并重置等待时间
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]   #APPROACH_OBJECT 当前直接将末端目标设为物体的根位姿，相当于末端目标与物体中心重合，这里刚好是因为物体是正方体，不同物体就会需要不同的位姿调整
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            # wait for a while
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                # move to next state and reset wait time
                sm_state[tid] = PickSmState.PLACE_OBJECT
                sm_wait_time[tid] = 0.0  

    # 自己增加的部分  
    elif state == PickSmState.PLACE_OBJECT:
        # 1. 目标位置设为固定放置点
        des_ee_pose[tid] = place_ee_pose[tid]  
        # 2. 过程中保持夹紧 
        gripper_state[tid] = GripperState.CLOSE
        # 3. 判断：当前实际末端(ee_pose) 距离 放置点(place_ee_pose) 是否小于阈值
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,        #这里的threshold用的position_threshold=0.01，想改可以在main那里改
        ):
            # 4. 等待一会，进入下一个阶段
            if sm_wait_time[tid] >= PickSmWaitTime.PLACE_OBJECT:
                sm_state[tid] = PickSmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0 

    elif state == PickSmState.RELEASE_OBJECT:
        # 松开夹爪
        des_ee_pose[tid] = place_ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.RELEASE_OBJECT:
            sm_state[tid] = PickSmState.RETREAT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.RETREAT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):                     #这里的threshold用的position_threshold=0.01，想改可以在main那里改
            if sm_wait_time[tid] >= PickSmWaitTime.RETREAT:
                sm_state[tid] = PickSmState.DONE
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.DONE:
        gripper_state[tid] = GripperState.OPEN   #到达之后，算了还是保持张开吧

    # 增加等待时间
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift and place an object.

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the 我设置放置位置的上方. 
    6. PLACE_OBJECT: 机械臂将物体方法到我设置的放置位置
    7. RELEASE_OBJECT: 释放物体
    8. RETREAT: 让机械臂回到放置位置的上方,这里我确保就是放置点的上方，我写为了固定点，在act_manipulation/source/act_manipulation/act_manipulation/tasks/manager_based/lift/lift_env_cfg.py这个文件里
    7. DONE: 合并加爪
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """初始化状态机

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
        
        # 🟢 在这里直接写入固定的放置点
        # 暂时让末端在 (0.3, 0, 0.3)，姿态保持默认(单位四元数)
        # Warp: (x, y, z, qx, qy, qz, qw)
        # 初始化放置位姿
        self.des_place_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_place_ee_pose[:, 6] = 1.0  # 确保四元数合法 (w=1)   这里只初始化这个des_place_ee_pose，在下面计算那里再设置具体数值再传到warp

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """重置状态机"""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        # 清除旧的期望末端位姿
        self.des_ee_pose[env_ids] = 0.0
        self.des_ee_pose[env_ids, 6] = 1.0  # Warp 内部 XYZW

        # 默认夹爪张开
        self.des_gripper_state[env_ids] = 1.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor,) -> torch.Tensor:
        """计算机器人末端执行器和夹具的期望状态"""
        # 将所有变换从 (w, x, y, z) 转换为 (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        object_quat = object_pose[:, 3:] 
        fixed_xyz = torch.tensor([0.0, -0.3, 0.08], device=self.device)
        self.des_place_ee_pose[:, 0:3] = fixed_xyz
        self.des_place_ee_pose[:, 3:] = object_quat
        
        # 转换到warp上
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # 4. 更新 warp
        self.des_place_ee_pose_wp = wp.from_torch(self.des_place_ee_pose.contiguous(), wp.transform)
        
        # 运行状态机
        wp.launch(      #wp.launch(dim=self.num_envs, ...) 会为每个环境启动对应的线程。wp.tid() 返回当前线程索引；在你的状态机中，它对应当前要处理的环境编号
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.des_place_ee_pose_wp,   #新加的放置点，11个数里面的倒数第三个哈，别对应错了
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )

        # 将变换转换回 (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]   #小心下标别搞错了

        
        # 转换为torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)



def main():
    if args_cli.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive.")
    if args_cli.target_episodes <= 0:
        raise ValueError("--target-episodes must be positive.")
    if args_cli.success_stable_seconds <= 0:
        raise ValueError("--success-stable-seconds must be positive.")
    if args_cli.num_envs != 1:
        raise ValueError("This data collector currently supports exactly one environment.")

    # Parse the same Absolute-IK environment used by the original collector.
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Lebai-IK-Abs-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    control_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    sample_dt = 1.0 / float(args_cli.sample_fps)
    steps_per_sample = int(round(sample_dt / control_dt))

    if steps_per_sample < 1 or not math.isclose(
        steps_per_sample * control_dt, sample_dt, rel_tol=0.0, abs_tol=1e-7
    ):
        raise ValueError(
            f"Cannot achieve {args_cli.sample_fps}Hz exactly with env step "
            f"{control_dt:.6f}s. Choose an FPS whose period is an integer "
            "multiple of sim.dt * decimation."
        )

    # Force both camera sensors to the same period as the data stream.
    # Set the sensor period here as well as in the task config, so the HDF5
    # metadata cannot claim 25Hz while the camera is still running at 10Hz.
    for camera_name in ("wrist_camera", "front_camera"):
        camera_cfg = getattr(env_cfg.scene, camera_name)
        camera_cfg.update_period = sample_dt
        camera_cfg.data_types = ["rgb", "distance_to_image_plane"]

    env_cfg.rerender_on_reset = True
    env = gym.make("Isaac-Lift-Cube-Lebai-IK-Abs-v0", cfg=env_cfg)
    env.reset()

    if env.unwrapped.num_envs != 1:
        env.close()
        raise RuntimeError("Only num_envs=1 is supported by this recorder.")

    recorder = DemoRecorder(
        save_dir=args_cli.save_dir,
        fps=args_cli.sample_fps,
    )

    # Placement region mirrors the geometric criterion used by the current
    # ACT evaluation script. Tray center (0, -0.3, 0.03) comes from the scene
    # config; XY half extents and Z limits are an explicit approximation.
    success_criterion = {
        "type": "geometric_settled_in_tray",
        "tray_center_xy_m": [args_cli.tray_center_x, args_cli.tray_center_y],
        "tray_half_extents_xy_m": [args_cli.tray_half_x, args_cli.tray_half_y],
        "object_center_z_range_m": [args_cli.object_z_min, args_cli.object_z_max],
        "max_object_speed_m_s": args_cli.object_speed_threshold,
        "gripper_must_be_open": True,
        "required_consecutive_frames": int(math.ceil(args_cli.success_stable_seconds * args_cli.sample_fps)),
        "required_stable_seconds": args_cli.success_stable_seconds,
        "sampling_hz": args_cli.sample_fps,
        "note": (
            "Approximate geometric label, not a contact-force or human-verified "
            "label. Match the tray bounds to the actual USD inner geometry if needed."
        ),
    }

    ee_frame_sensor = env.unwrapped.scene["ee_frame"]
    tcp_rest_position = (
        ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
        - env.unwrapped.scene.env_origins
    )
    tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()

    actions = torch.cat(
        [
            tcp_rest_position,
            tcp_rest_orientation,
            torch.ones(
                (env.unwrapped.num_envs, 1),
                device=env.unwrapped.device,
            ),
        ],
        dim=-1,
    )
    planned_pos = tcp_rest_position.clone()

    desired_orientation = torch.zeros(
        (env.unwrapped.num_envs, 4),
        device=env.unwrapped.device,
    )
    desired_orientation[:, 1] = 1.0

    # The FSM and Cartesian rate limiter are updated once per recorded frame
    # (25Hz), while each command is held for two 50Hz environment steps.
    pick_sm = PickAndLiftSm(
        sample_dt,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        position_threshold=0.01,
    )

    def get_hold_action():
        ee_sensor = env.unwrapped.scene["ee_frame"]
        pos = (
            ee_sensor.data.target_pos_w[..., 0, :].clone()
            - env.unwrapped.scene.env_origins
        )
        quat = ee_sensor.data.target_quat_w[..., 0, :].clone()
        gripper = torch.ones(
            (env.unwrapped.num_envs, 1),
            device=env.unwrapped.device,
        )
        return pos, quat, torch.cat([pos, quat, gripper], dim=-1)

    episode_sample_index = 0
    completed_episodes = 0
    previous_object_position = None
    stable_count = 0
    success_latched = False
    max_ee_speed = float(args_cli.max_ee_speed)

    print("=" * 72)
    print("[COLLECTOR] Synchronized recording frequency:", args_cli.sample_fps, "Hz")
    print("[COLLECTOR] Environment step:", f"{control_dt:.4f}s")
    print("[COLLECTOR] Environment steps per recorded frame:", steps_per_sample)
    print("[COLLECTOR] Camera update period:", f"{sample_dt:.4f}s")
    print("[COLLECTOR] Output directory:", args_cli.save_dir)
    print("[COLLECTOR] Depth keys: distance_to_image_plane (float32, meters)")
    print("[COLLECTOR] Success criterion:", json.dumps(success_criterion, ensure_ascii=False))
    print("=" * 72)

    try:
        while simulation_app.is_running() and completed_episodes < args_cli.target_episodes:
            with torch.inference_mode():
                # Advance exactly one sample interval while holding the same
                # absolute IK target. This makes the action stream genuinely
                # 25Hz instead of logging every second command from a 50Hz FSM.
                abnormal_reason = None
                for _ in range(steps_per_sample):
                    _, _, terminated, time_outs, _ = env.step(actions)
                    reset_buf = terminated | time_outs
                    reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    if reset_env_ids.numel() > 0:
                        if bool(terminated[0].item()):
                            abnormal_reason = "terminated_object_drop_or_other"
                        elif bool(time_outs[0].item()):
                            abnormal_reason = "time_out"
                        else:
                            abnormal_reason = "environment_reset"
                        break

                if abnormal_reason is not None:
                    path = recorder.save(
                        success=int(success_latched),
                        fsm_done=False,
                        termination_reason=abnormal_reason,
                        success_criterion=success_criterion,
                    )
                    print(f"[RESET] Saved incomplete failed episode: {path}")

                    # Isaac Lab has already auto-reset the environment.
                    pick_sm.reset_idx()
                    tcp_rest_position, tcp_rest_orientation, actions = get_hold_action()
                    planned_pos = tcp_rest_position.clone()
                    episode_sample_index = 0
                    previous_object_position = None
                    stable_count = 0
                    success_latched = False
                    continue

                # ------------------------------------------------------
                # Read the synchronized robot/object state after 40ms.
                # ------------------------------------------------------
                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_rest_position = (
                    ee_frame_sensor.data.target_pos_w[..., 0, :].clone()
                    - env.unwrapped.scene.env_origins
                )
                tcp_rest_orientation = (
                    ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
                )

                object_data = env.unwrapped.scene["object"].data
                object_position = (
                    object_data.root_pos_w
                    - env.unwrapped.scene.env_origins
                )

                desired_position = (
                    env.unwrapped.command_manager.get_command("object_pose")[..., :3]
                )

                # ------------------------------------------------------
                # FSM: update at 25Hz using dt=0.04s.
                # ------------------------------------------------------
                target_actions = pick_sm.compute(
                    torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                    torch.cat([object_position, desired_orientation], dim=-1),
                    torch.cat([desired_position, desired_orientation], dim=-1),
                )

                # Cartesian rate limiter is expressed in m/s, so scale the
                # per-update increment by the 40ms sample period.
                delta = target_actions[:, :3] - planned_pos
                distance = torch.linalg.vector_norm(
                    delta, dim=-1, keepdim=True
                ).clamp_min(1e-8)
                max_step = max_ee_speed * sample_dt
                ratio = torch.clamp(max_step / distance, max=1.0)
                planned_pos = planned_pos + delta * ratio

                actions = target_actions.clone()
                actions[:, :3] = planned_pos

                # ------------------------------------------------------
                # Geometric placement success label.
                # A success requires the object to remain in the tray,
                # gripper open, and object speed low for >= 0.2s.
                # ------------------------------------------------------
                object_pos_np = (
                    object_position[0].detach().cpu().numpy().astype("float32")
                )
                if previous_object_position is None:
                    object_velocity_np = [0.0, 0.0, 0.0]
                    object_speed = 0.0
                else:
                    displacement = object_pos_np - previous_object_position
                    object_velocity_np = (
                        displacement / sample_dt
                    ).astype("float32").tolist()
                    object_speed = math.sqrt(
                        float(displacement[0]) ** 2
                        + float(displacement[1]) ** 2
                        + float(displacement[2]) ** 2
                    ) / sample_dt
                previous_object_position = object_pos_np.copy()

                inside_tray = (
                    abs(float(object_pos_np[0]) - args_cli.tray_center_x)
                    <= args_cli.tray_half_x
                    and abs(float(object_pos_np[1]) - args_cli.tray_center_y)
                    <= args_cli.tray_half_y
                    and args_cli.object_z_min
                    <= float(object_pos_np[2])
                    <= args_cli.object_z_max
                )

                fsm_state = int(pick_sm.sm_state[0].item())
                gripper_open = float(actions[0, 7].item()) > 0.0
                success_condition = (
                    fsm_state >= 6
                    and inside_tray
                    and gripper_open
                    and object_speed < args_cli.object_speed_threshold
                )

                if success_condition:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= int(math.ceil(args_cli.success_stable_seconds * args_cli.sample_fps)):
                    success_latched = True

                episode_sample_index += 1
                current_ee_pose = torch.cat(
                    [tcp_rest_position, tcp_rest_orientation],
                    dim=-1,
                )

                recorder.add(
                    scene=env.unwrapped.scene,
                    action=actions,
                    ee_pose=current_ee_pose,
                    fsm_state=fsm_state,
                    timestamp=episode_sample_index * sample_dt,
                    env_origins=env.unwrapped.scene.env_origins,
                    object_position=object_position,
                    object_velocity=object_velocity_np,
                    object_speed=object_speed,
                    inside_tray=inside_tray,
                    success_condition=success_condition,
                    success_stable_count=stable_count,
                )

                if episode_sample_index % (args_cli.sample_fps * 2) == 0:
                    print(
                        f"[COLLECT] t={episode_sample_index * sample_dt:5.2f}s "
                        f"fsm={fsm_state} object={object_pos_np.round(3).tolist()} "
                        f"inside_tray={inside_tray} object_speed={object_speed:.3f} "
                        f"stable={stable_count}/{int(math.ceil(args_cli.success_stable_seconds * args_cli.sample_fps))}"
                    )

                # ------------------------------------------------------
                # FSM completed. Label success from measured placement,
                # not merely from reaching DONE.
                # ------------------------------------------------------
                done_ready = (
                    fsm_state == 8
                    and pick_sm.sm_wait_time[0].item() >= 0.2
                )

                if done_ready:
                    episode_success = int(success_latched)
                    reason = (
                        "object_settled_in_tray"
                        if episode_success
                        else "fsm_completed_but_placement_not_verified"
                    )

                    path = recorder.save(
                        success=episode_success,
                        fsm_done=True,
                        termination_reason=reason,
                        success_criterion=success_criterion,
                    )

                    # if path is not None:
                    #     completed_episodes += 1
                    # 上面是收到两百条数据，下面是收集两百条成功数据：
                    if path is not None and episode_success == 1:
                        completed_episodes += 1

                    print(
                        f"[EPISODE SAVED] {path} "
                        f"success={episode_success} "
                        f"completed={completed_episodes}/{args_cli.target_episodes}"
                    )

                    if completed_episodes >= args_cli.target_episodes:
                        print("[DONE] Target number of FSM-completed episodes collected.")
                        break

                    env.reset()
                    pick_sm.reset_idx()
                    tcp_rest_position, tcp_rest_orientation, actions = get_hold_action()
                    planned_pos = tcp_rest_position.clone()
                    episode_sample_index = 0
                    previous_object_position = None
                    stable_count = 0
                    success_latched = False

    finally:
        # If the user closes Isaac Sim mid-episode, preserve the partial data
        # but mark its outcome unknown instead of fabricating a failure/success.
        if recorder.has_data:
            recorder.save(
                success=(1 if success_latched else -1),
                fsm_done=False,
                termination_reason="interrupted_or_simulation_closed",
                success_criterion=success_criterion,
            )
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
