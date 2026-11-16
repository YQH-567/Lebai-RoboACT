# LM3 + LMG90 参考图外观修改报告

目标：`lebai_lm3_lmg90_color_new1114.usd`，已覆盖保存。

六个 Link 使用银灰白壳体、局部灰蓝色接缝带；Link 2 和 Link 3 的两面添加黑红 LEBAI 标识；肩部顶部加入细红/青色状态环。夹爪主体为银白，所有左右运动连杆和末端夹指统一红色。

使用物体局部坐标的 MDL 投影，无需 UV，不增加装饰 Mesh，不修改点、法线、面索引、层级或实例化设置。色带位置依据参考图和现有网格范围近似放置。

品牌标识使用[乐白官网 SVG](https://www.lebai.ltd/logo.svg)，保留官方形状，将网站较浅的配色调整为参考图中的黑红印刷色。[MDL 局部坐标方法](https://github.com/NVIDIA/MDL-SDK/blob/master/src/shaders/mdl/base/base.mdl)用于随部件运动的投影。

## 材质参数（线性 RGB）

| 区域 | RGB | Roughness | Metallic |
|---|---|---|---|
| 银灰白外壳 | 0.58, 0.60, 0.63 | 0.47 | 0.18 |
| 灰蓝色带 | 0.28, 0.34, 0.38 | 0.60 | 0.05 |
| 红色夹爪及末端 | 0.55, 0.018, 0.025 | 0.43 | 0.28 |

## 验证

- 12 个原有转动关节和 2 个活动固定关节保留。
- 原有 Prim 路径、类型、实例化结构保留。
- 对比修改前备份：根层移除允许的材质与绑定字段后，文本完全相同；因此 Joint、Drive、限位、PD、PhysX、相机、灯光和其他非材质配置未变化。
- 所有被引用 USD 层的文件 SHA-256 保持一致。
- 14 个 Visual Mesh 的新绑定生效，14 个 Collision Mesh 的材质绑定保持不变。
- Isaac Sim 5.1 RTX 成功编译自定义 MDL，正面、反面和夹爪近景均输出 RGB，日志无 MDL/渲染错误。
- 预览使用运行时临时相机和 450 强度补光；没有保存相机或灯光变更。预览结束后目标 USD 的 SHA-256 与保存时一致。

## 部件映射

|部件 Prim|Visual Mesh|Collision Mesh|材质 Prim|
|---|---|---|---|
|`/World/lebai_lm3/base_link`|`/World/lebai_lm3/base_link/visuals/base_link/node_STL_BINARY_/mesh`|`/World/lebai_lm3/base_link/collisions/base_link/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Base`|
|`/World/lebai_lm3/link_1`|`/World/lebai_lm3/link_1/visuals/link1/node_STL_BINARY_/mesh`|`/World/lebai_lm3/link_1/collisions/link1/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Link1`|
|`/World/lebai_lm3/link_2`|`/World/lebai_lm3/link_2/visuals/link2/node_STL_BINARY_/mesh`|`/World/lebai_lm3/link_2/collisions/link2/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Link2`|
|`/World/lebai_lm3/link_3`|`/World/lebai_lm3/link_3/visuals/link3/node_STL_BINARY_/mesh`|`/World/lebai_lm3/link_3/collisions/link3/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Link3`|
|`/World/lebai_lm3/link_4`|`/World/lebai_lm3/link_4/visuals/link4/node_STL_BINARY_/mesh`|`/World/lebai_lm3/link_4/collisions/link4/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Link4`|
|`/World/lebai_lm3/link_5`|`/World/lebai_lm3/link_5/visuals/link5/node_STL_BINARY_/mesh`|`/World/lebai_lm3/link_5/collisions/link5/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Link5`|
|`/World/lebai_lm3/link_6`|`/World/lebai_lm3/link_6/visuals/link6/node_STL_BINARY_/mesh`|`/World/lebai_lm3/link_6/collisions/link6/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_LM3_Reference_Link6`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_base_link`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_base_link/visuals/gripper_base/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_base_link/collisions/gripper_base/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_SatinLightGray`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link1`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link1/visuals/gripper_link1/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link1/collisions/gripper_link1/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_GraphiteGray`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link2`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link2/visuals/gripper_link2/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link2/collisions/gripper_link2/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_GraphiteGray`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link_finger`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link_finger/visuals/gripper_link_finger/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_l_link_finger/collisions/gripper_link_finger/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_GraphiteGray`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link1`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link1/visuals/gripper_link1/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link1/collisions/gripper_link1/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_GraphiteGray`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link2`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link2/visuals/gripper_link2/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link2/collisions/gripper_link2/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_GraphiteGray`|
|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link_finger`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link_finger/visuals/gripper_link_finger/node_STL_BINARY_/mesh`|`/World/lmg90_gripper_inertial_stable/gripper/gripper_r_link_finger/collisions/gripper_link_finger/node_STL_BINARY_/mesh`|`/World/Looks/Industrial_GraphiteGray`|

## 文件依赖与使用

运行时需要与 USD 一同保留 `materials/lm3_reference/lm3_reference.mdl` 和 `materials/lm3_reference/lebai_marking.png`。SVG 为标识源文件；材质相对路径支持整体移动模型目录。RTX 使用完整 MDL 外观，普通非 MDL USD 查看器只能显示基础颜色。

已有材质路径 `Industrial_GraphiteGray` 被复用为红色，以保留先前路径；显示名称已注明红色。`Industrial_MutedSteelBlue` 保留但不再绑定到任何夹指。

备份：`lebai_lm3_lmg90_color_new1114.before_projection.usd`。可将其内容恢复到目标文件撤销这次投影外观修改。

当前 Isaac Lab 资产配置仍指向 `lebai_lm3_lmg90_new1112.usd`。本次只修改指定 USD 和它的外观依赖，未切换任务模型来源。采集使用新外观时需指向 `lebai_lm3_lmg90_color_new1114.usd`。

本次按产品参考图近似外观，尚未用实机采集相机的 RGB 标定色差。光照、曝光和白平衡仍会影响仿真与实机最终像素；原有 STL 网格的棱面也保持原样。未运行用户的 IK 抓取或 ACT 训练。
