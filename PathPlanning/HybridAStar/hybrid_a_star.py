"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from dynamic_programming_heuristic import calc_distance_heuristic
from ReedsSheppPath import reeds_shepp_path_planning as rs
from car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R

XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
N_STEER = 20  # number of steer command

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

show_animation = True


class Node:
    """
    混合A*搜索树中的一个节点：代表车的一个配置状态和到达该状态的轨迹
    
    存储三维关键信息：
    1. 离散3D网格位置 (x_index, y_index, yaw_index)
    2. 完整的运动轨迹（30个采样点的xyz坐标和方向）
    3. 搜索信息（成本、转向角、父节点指针）
    
    特点：每个Node包含一段连续的3米运动轨迹（不只是离散点），保证路径平滑性
    """

    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        """
        初始化Node对象
        
        参数解释：
          x_ind, y_ind, yaw_ind: 3D网格索引（整数坐标）
          direction: 运动方向（True=前进，False=倒车）
          x_list, y_list, yaw_list: 轨迹采样点序列（通常30个点，代表3米运动）
          directions: 轨迹上每点的运动方向标志列表
          steer: 此段运动的转向角（-0.5到0.5弧度之间，影响运动学）
          parent_index: 父节点的网格索引（1D整数），用于路径回溯
          cost: 从起点到此节点的累积成本
        
        数值示例：
          Node(8, 6, 6, True,
               [15.4, 15.5, ..., 18.4],  # 30个X坐标
               [12.2, 12.3, ..., 15.2],  # 30个Y坐标
               [1.57, 1.575, ...],       # 30个方向角
               [True, True, ...],        # 30个方向标志
               steer=0.3,                # 转向角
               parent_index=12345,       # 前驱节点ID
               cost=45.2)                # 成本
        """
        self.x_index = x_ind  # X网格索引
        self.y_index = y_ind  # Y网格索引
        self.yaw_index = yaw_ind  # YAW网格索引（方向）
        self.direction = direction  # 整体运动方向（True=前进）
        self.x_list = x_list  # 轨迹X坐标序列（米）
        self.y_list = y_list  # 轨迹Y坐标序列（米）
        self.yaw_list = yaw_list  # 轨迹方向角序列（弧度）
        self.directions = directions  # 轨迹方向标志序列
        self.steer = steer  # 此段运动的方向盘转角（弧度）
        self.parent_index = parent_index  # 前驱节点的网格1D索引
        self.cost = cost  # 从起点的累积成本


class Path:
    """
    最终规划路径：包含完整的轨迹坐标和运动成本
    
    由get_final_path()函数生成，通过连接所有visited Node的轨迹得到
    可直接用于车的控制和可视化
    """

    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        """
        初始化Path对象
        
        参数：
          x_list: 路径上所有采样点的X坐标列表（米）
          y_list: 路径上所有采样点的Y坐标列表（米）
          yaw_list: 路径上所有采样点的方向角列表（弧度）
          direction_list: 路径上每个采样点的运动方向标志（True=前进）
          cost: 路径的总成本（从起点到终点的累积成本）
        
        数值示例：
          Path(
            x_list=[10.0, 10.1, 10.2, ..., 50.0],  # 数百个点
            y_list=[10.0, 10.0, 10.1, ..., 50.0],
            yaw_list=[1.57, 1.571, ...],           # 对应的方向
            direction_list=[True, True, ..., True],
            cost=134.2                             # 总成本
          )
        
        用途：直接输入给车的轨迹跟踪控制器
        """
        self.x_list = x_list  # 路径X坐标序列
        self.y_list = y_list  # 路径Y坐标序列
        self.yaw_list = yaw_list  # 路径方向序列
        self.direction_list = direction_list  # 路径方向标志序列
        self.cost = cost  # 总成本


class Config:
    """
    网格配置类：将连续坐标和角度离散化为网格索引
    
    核心作用：建立从连续世界坐标 → 离散网格索引的映射
    存储A*搜索的3D网格的边界和宽度信息
    """

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        """
        初始化网格配置参数
        
        输入：障碍物坐标列表、分辨率参数
        输出：9个网格参数（min_x/y/yaw, max_x/y/yaw, x_w/y_w/yaw_w）
        """
        
        # ════════════════════════════════════ 第1-4行：找障碍物的外包围框 ════════════════════════════════════
        min_x_m = min(ox)  # 障碍物最小X坐标（米）
        min_y_m = min(oy)  # 障碍物最小Y坐标（米）
        max_x_m = max(ox)  # 障碍物最大X坐标（米）
        max_y_m = max(oy)  # 障碍物最大Y坐标（米）
        # 示例：地图60×60米，则min=0, max=60

        # ════════════════════════════════════ 第5-8行：添加边界点确保有效 ════════════════════════════════════
        ox.append(min_x_m)  # 向障碍物列表添加四个角点
        oy.append(min_y_m)  # 确保边界被认为是网格范围的有效部分
        ox.append(max_x_m)  # 修改原始列表（Python中列表是引用传递）
        oy.append(max_y_m)

        # ════════════════════════════════════ 第10-13行：XY平面网格离散化 ════════════════════════════════════
        self.min_x = round(min_x_m / xy_resolution)  # 最小X网格索引
        # = round(0 / 2.0) = 0
        self.min_y = round(min_y_m / xy_resolution)  # 最小Y网格索引
        self.max_x = round(max_x_m / xy_resolution)  # 最大X网格索引
        # = round(60 / 2.0) = 30 (当分辨率=2.0米时)
        self.max_y = round(max_y_m / xy_resolution)  # 最大Y网格索引

        # ════════════════════════════════════ 第15-16行：网格宽度（用于索引计算） ════════════════════════════════════
        self.x_w = round(self.max_x - self.min_x)
        # = 30 - 0 = 30 (X方向共30个网格单元，加上起点共31个点)
        self.y_w = round(self.max_y - self.min_y)
        # = 30 - 0 = 30 (Y方向网格数)
        # 这两个值在calc_index()中用于3D→1D索引转换
        # 公式：ind = (yaw_idx - min_yaw) × x_w × y_w + (y_idx - min_y) × x_w + (x_idx - min_x)

        # ════════════════════════════════════ 第18-20行：方向(YAW)网格离散化 ════════════════════════════════════
        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        # = round(-3.14159 / 0.262) - 1
        # = round(-12.0) - 1 = -13
        # -π弧度 ≈ -180° (车方向向左)
        
        self.max_yaw = round(math.pi / yaw_resolution)
        # = round(3.14159 / 0.262)
        # = round(12.0) = 12
        # +π弧度 ≈ +180° (车方向向右)
        # YAW_GRID_RESOLUTION = 0.262弧度 = 15°
        # 方向索引范围：[-13, 12]，共26个方向等级
        
        self.yaw_w = round(self.max_yaw - self.min_yaw)
        # = 12 - (-13) = 25 (方向网格数)
        # 在calc_index()中用于YAW维度的偏移计算
        
        # 总结：这3个Config参数定义了一个31×31×26的3D网格
        # 约有25,000个搜索状态（较大的搜索空间）


def calc_motion_inputs():
    """
    生成所有可能的转向角和方向组合
    返回一个生成器，每次产生 [转向角, 方向] 对
    
    总共产生：21个转向角 × 2个方向 = 42种动作
    """
    
    # ════════════════════════════════════ 第1-3行：生成转向角数组 ════════════════════════════════════
    # np.linspace(-MAX_STEER, MAX_STEER, N_STEER)
    # 在[-MAX_STEER, MAX_STEER]范围内均匀生成N_STEER个点
    #
    # 参数含义：
    # -MAX_STEER = -0.5 rad  (最大左转角)
    # MAX_STEER = 0.5 rad    (最大右转角)
    # N_STEER = 20           (分割数)
    #
    # 结果：20个点均匀分布
    # [-0.5, -0.45, -0.4, ..., 0.4, 0.45, 0.5]
    #
    # np.concatenate((..., [0.0]))
    # 将上述数组与[0.0]合并，确保0.0（直线）一定被包含
    #
    # 最终结果：21个转向角
    # [-0.5, -0.45, ..., 0.45, 0.5, 0.0]
    #          ↑                    ↑
    #       np.linspace的20个   额外加入的0.0
    
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER,
                                             N_STEER), [0.0])):
        # 遍历所有21个转向角
        # 每个steer是一个浮点数：-0.5, -0.45, ..., 0.5, 0.0
        
        # ────────────────────── 第4-5行：生成前进和倒车两个方向 ──────────────────────
        for d in [1, -1]:
            # 对每个转向角，生成两个方向
            # d = 1 : 前进
            # d = -1: 倒车
            #
            # 这样形成 21 × 2 = 42 种动作
            
            yield [steer, d]
            # yield是生成器关键字，每次返回一对[转向角, 方向]
            # 不是return，而是逐个产生，节省内存
            #
            # 使用示例：
            # for action in calc_motion_inputs():
            #     print(action)  # [0.5, 1], [0.5, -1], [0.45, 1], ... (共42对)
            #
            # 为什么用生成器？
            # 如果一次性生成42个动作放在列表里会占用内存
            # 用生成器可以按需产生，每次需要新动作时才计算
            # 特别适合在循环中多次使用（如get_neighbors中）


def get_neighbors(current, config, ox, oy, kd_tree):
    """
    获取当前节点的所有有效邻域节点
    
    流程：
    1. 尝试所有42种动作
    2. 对每种动作，用calc_next_node()模拟3米运动
    3. 碰撞检测（在calc_next_node内部）
    4. 边界验证（verify_index）
    5. 仅返回有效的邻域节点
    """
    
    # ════════════════════════════════════ 遍历所有可能的动作 ════════════════════════════════════
    for steer, d in calc_motion_inputs():
        # calc_motion_inputs()生成器产生42对[转向角, 方向]
        # 对每一对执行以下步骤
        #
        # steer示例: 0.5, 0.45, 0.4, ..., -0.5, 0.0
        # d示例:     1(前进) 或 -1(倒车)
        
        # ─────────────────────── 第2-3行：模拟动作并碰撞检测 ──────────────────────
        node = calc_next_node(current, steer, d, config, ox, oy, kd_tree)
        # calc_next_node()执行：
        # 1️⃣ 从当前节点出发
        # 2️⃣ 以转向角steer、方向d执行运动
        # 3️⃣ 模拟3米距离的运动（30步，每步0.1米）
        # 4️⃣ 检查轨迹上所有30个点是否与障碍物碰撞
        # 5️⃣ 计算这个动作的成本
        #
        # 返回值：
        # node = Node对象    : 动作有效（无碰撞，成本计算完毕）
        # node = None        : 动作无效（有碰撞，此动作被拒绝）
        
        # ─────────────────────── 第4行：边界检查和最终验证 ──────────────────────
        if node and verify_index(node, config):
            # 两个条件都必须满足：
        
            # 条件1: node（非None）
            # 已通过碰撞检测，点数据有效
        
            # 条件2: verify_index(node, config)
            # 验证节点的网格索引在地图范围内
            # verify_index检查：config.min_x <= x_index <= config.max_x
            #                  config.min_y <= y_index <= config.max_y
            # （不检查yaw_index，因为yaw是圆周的）
            #
            # 为什么需要这个检查？
            # calc_next_node会模拟3米运动
            # 可能会跑出地图边界
            # verify_index确保新节点在有效范围内
            
            yield node
            # 只有同时通过碰撞检测和边界检查的节点才会被返回
            #
            # 使用生成器yield而不是return：
            # 一次调用get_neighbors()可能产生0-42个有效邻域
            # yield逐个返回，上层函数通过for循环接收
            #
            # 使用示例（在hybrid_a_star_planning中）：
            # for neighbor in get_neighbors(current, ...):
            #     print(neighbor)  # 依次处理每个有效邻域


def calc_next_node(current, steer, direction, config, ox, oy, kd_tree):
    """
    A*的核心：从当前节点执行动作，发生3米运动，返回新节点或None
    
    十大步骤：
    1. 提取当前位置  2. 计算运动弧长  3. 模拟30步运动  4. 碰撞检测
    5. 离散化终点    6. 换向惩罚      7. 转向惩罚      8. 转向变化惩罚
    9. 累积总成本    10. 创建新Node对象
    """
    
    # ════════════════════════════════════ 步骤1：提取当前位置 ════════════════════════════════════
    x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]
    # [-1] 取当前Node轨迹的末尾点（最近的位置）
    # 返回值都是浮点数，单位：x,y(米)，yaw(弧度)
    # 例：x=15.4m, y=12.2m, yaw=1.57rad(≈90°)

    # ════════════════════════════════════ 步骤2-3：模拟3米高精度运动 ════════════════════════════════════
    arc_l = XY_GRID_RESOLUTION * 1.5
    # = 2.0 × 1.5 = 3.0米
    # 这是混合A*中每个离散步骤的连续运动距离
    
    x_list, y_list, yaw_list, direction_list = [], [], [], []
    # 初始化4个列表，记录运动轨迹的30个采样点
    
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
    # np.arange(0, 3.0, 0.1) = [0, 0.1, 0.2, ..., 2.9] 共30个值
    # 每次迭代代表0.1米的微小步骤
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        # move()函数实现汽车运动学模型（自行车模型）
        # 输入：当前位置(x,y,yaw)、移动距离(0.1 × direction)、转向角steer
        # direction: 1=前进，-1=倒车
        # 输出：更新后的位置和方向
        # 数值示例：
        #   输入：x=15.4, y=12.2, yaw=1.57, dist=0.1, steer=0.5
        #   输出：x≈15.5, y≈12.3, yaw≈1.575（通过运动学方程计算）
        
        x_list.append(x)  # 记录此步骤后的位置
        y_list.append(y)
        yaw_list.append(yaw)
        direction_list.append(direction == 1)  # True=前进，False=倒车
    # 循环后：三个list各有30个元素，完整记录了3米运动的轨迹

    # ════════════════════════════════════ 步骤4：碰撞检测（关键！） ════════════════════════════════════
    if not check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
    # check_car_collision()逐点检查轨迹上30个位置是否与任何障碍物碰撞
    # 使用KDTree数据结构加速最近邻查询
    # 返回值：True=无碰撞✓，False=有碰撞✗
    #
    # 如果发生碰撞（返回False），则NOT False = True，进入if块
        return None  # 此动作无效，直接返回None，不加入openList

    # ════════════════════════════════════ 步骤5：离散化终点网格位置 ════════════════════════════════════
    d = direction == 1  # d是布尔值，用于Node.direction属性
    # d=True：前进，d=False：倒车
    # 注意：这里的d是逻辑离散化，不同于运动学中的direction
    
    x_ind = round(x / XY_GRID_RESOLUTION)
    # 将最终的米数坐标转换为网格索引
    # round(15.4 / 2.0) = round(7.7) = 8
    # 网格索引用整数表示，便于字典查询
    
    y_ind = round(y / XY_GRID_RESOLUTION)
    # = round(12.2 / 2.0) = round(6.1) = 6
    
    yaw_ind = round(yaw / YAW_GRID_RESOLUTION)
    # = round(1.57 / 0.262) = round(6.0) = 6
    # 三个整数索引(8,6,6)定义了搜索树中的新网格状态

    # ════════════════════════════════════ 步骤6-8：计算额外成本三部分 ════════════════════════════════════
    added_cost = 0.0  # 初始化本步骤额外成本

    # 部分1：换向惩罚（前进↔倒车切换）
    if d != current.direction:
    # d是本步骤的方向，current.direction是前驱Node的方向
    # 如果两者不同，说明发生了前进↔倒车的切换
        added_cost += SB_COST
        # SB_COST = 100.0（非常大！）
        # 强烈禁止频繁倒车和换向，激励连续前进

    # 部分2：转向角惩罚（鼓励直线少转向）
    added_cost += STEER_COST * abs(steer)
    # STEER_COST = 1.0
    # abs(steer)是转向角的绝对值
    # 数值示例：
    #   steer=0.0   → 惩罚=1.0×0=0     （直线，无惩罚）
    #   steer=0.5   → 惩罚=1.0×0.5=0.5 （最大转向，惩罚0.5）
    #   steer=-0.3  → 惩罚=1.0×0.3=0.3 （中等转向，惩罚0.3）
    # 目的：激励路径规划器选择转向角较小的动作（接近直线）

    # 部分3：转向角变化惩罚（激励平滑转向）
    added_cost += STEER_CHANGE_COST * abs(current.steer - steer)
    # STEER_CHANGE_COST = 5.0
    # 计算前驱Node的转向角与本步骤转向角的差值
    # 数值示例：
    #   前驱steer=0.3，本步骤steer=-0.3
    #   |(-0.3) - 0.3| = 0.6
    #   惩罚 = 5.0 × 0.6 = 3.0（很大！）
    # 这激励路径选择转向平滑过渡的动作
    # 避免从左满舵突然切到右满舵的急剧转向

    # ════════════════════════════════════ 步骤9：累积总成本 ════════════════════════════════════
    cost = current.cost + added_cost + arc_l
    # current.cost：从起点到当前Node的累积成本
    #   示例：100.0（已走过的路径总成本）
    # added_cost：此步骤的三部分额外成本
    #   示例：SB_COST(100) + 0.3(转向) + 2.0(转向变化) = 102.3
    # arc_l：3.0米距离的基础成本
    # 总成本 = 100.0 + 102.3 + 3.0 = 205.3
    # 这个cost用于路径优化决策和最终成本计算

    # ════════════════════════════════════ 步骤10：创建并返回新Node对象 ════════════════════════════════════
    node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                y_list, yaw_list, direction_list,
                parent_index=calc_index(current, config),
                cost=cost, steer=steer)
    # Node参数详解：
    #   x_ind, y_ind, yaw_ind：新节点的3D网格位置
    #   d：运动方向（True=前进，False=倒车）
    #   x_list, y_list, yaw_list：完整轨迹的30个采样点
    #   direction_list：轨迹上每点的运动方向标志
    #   parent_index：当前Node的1维索引（用于后续路径回溯）
    #     calc_index()将(x_ind,y_ind,yaw_ind)转换为唯一的整数ID
    #   cost：从起点的累积成本
    #   steer：此步骤的转向角（供下一步骤计算转向变化）
    # 
    # 新Node代表了从current出发，执行"转向角=steer，方向=direction"的动作后
    # 运动3米得到的结果。Node包含这3米运动的完整轨迹和成本信息。

    return node  # 返回新创建的有效Node，将被加入openList进行后续搜索


def is_same_grid(n1, n2):
    """
    比较两个节点是否占据相同的网格位置
    
    在3D网格空间中去重：检查(x_index, y_index, yaw_index)是否完全相同
    短路求值：第一个False条件就直接返回
    """
    
    if n1.x_index == n2.x_index \
            and n1.y_index == n2.y_index \
            and n1.yaw_index == n2.yaw_index:
    # 三个条件必须同时满足（AND逻辑）
    # 短路求值：如果第一个条件False，后续条件不被计算
    # 性能优化：大多数情况下只需计算第一个条件（不匹配）
        return True  # 三个网格索引都相同
    return False     # 至少有一个不相同


def analytic_expansion(current, goal, ox, oy, kd_tree):
    """
    混合A*的优化：尝试用Reeds-Shepp曲线直接连接当前节点到目标
    
    如果找到无碰撞的最优路径，可跳过数百步A*搜索！
    返回最优Reeds-Shepp路径对象或None
    """
    
    # ════════════════════════════════════ 第1-6行：提取起点和终点坐标 ════════════════════════════════════
    start_x = current.x_list[-1]  # 当前节点末尾位置X
    start_y = current.y_list[-1]  # 当前节点末尾位置Y
    start_yaw = current.yaw_list[-1]  # 当前节点末尾方向
    # 例：(15.4, 12.2, 1.57) 当前车位置

    goal_x = goal.x_list[-1]  # 目标节点X
    goal_y = goal.y_list[-1]  # 目标节点Y
    goal_yaw = goal.yaw_list[-1]  # 目标节点方向
    # 例：(50.0, 50.0, -1.57) 目标位置和应到达的方向

    # ════════════════════════════════════ 第8行：计算最大曲率（车辆约束） ════════════════════════════════════
    max_curvature = math.tan(MAX_STEER) / WB
    # MAX_STEER = 0.5 rad（最大转向角）
    # WB = 2.9 m（轴距）
    # 公式：curvature = tan(steering_angle) / wheelbase
    # = tan(0.5) / 2.9 ≈ 0.188
    # 表示汽车的最大转弯能力约束

    # ════════════════════════════════════ 第9-12行：生成所有可能的Reeds-Shepp路径 ════════════════════════════════════
    paths = rs.calc_paths(start_x, start_y, start_yaw,
                          goal_x, goal_y, goal_yaw,
                          max_curvature, step_size=MOTION_RESOLUTION)
    # rs.calc_paths()是Reeds-Shepp路径规划器
    # 从任意起点/方向连接到任意目标/方向的所有可能方案（约48种）
    #
    # 方案包括：
    #   - 左弧+直线+右弧 (LSR)
    #   - 右弧+直线+左弧 (RSL)
    #   - 左弧+倒车+左弧 (LBL)
    #   - ... 其他48种组合
    #
    # 返回值：paths列表，每个元素是一条路径对象
    # 路径对象包含：x, y, yaw, lengths, ctypes, directions
    # step_size=0.1 表示沿路径每0.1米取一个采样点（用于碰撞检测）

    # ════════════════════════════════════ 第14-15行：检查是否有可行方案 ════════════════════════════════════
    if not paths:
    # 如果paths为空列表
    # 原因：
    #   1. 起点和目标点在同一网格（太近）
    #   2. 几何上无可行方案（极端情况）
    #   3. 规划器内部错误
        return None  # 无法直接连接，继续用常规A*搜索

    # ════════════════════════════════════ 第17-24行：评估所有路径并找最优 ════════════════════════════════════
    best_path, best = None, None  # 初始化最优路径追踪变量

    for path in paths:  # 遍历所有候选路径
        # ────────────────── 碰撞检测 ─────────────────
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
        # 检查路径上所有采样点是否与障碍物碰撞
        # 返回True=无碰撞✓，False=有碰撞✗
            # ────────────────── 成本计算 ─────────────────
            cost = calc_rs_path_cost(path)
            # calc_rs_path_cost()计算该Reeds-Shepp路径的总成本
            # 成本 = 距离 + 倒车惩罚 + 换向惩罚 + 转向惩罚 + 转向变化惩罚
            # 例：路径成本=34.2
            
            # ────────────────── 最优化更新 ─────────────────
            if not best or best > cost:
            # 两个条件之一满足就更新：
            # 1. not best：第一条有效路径（best=None）
            # 2. best > cost：当前路径成本更低
                best = cost  # 更新最低成本
                best_path = path  # 记录最优路径对象

    return best_path
    # 返回值三种情况：
    # 1️⃣ 返回路径对象：找到无碰撞的最优Reeds-Shepp路径
    # 2️⃣ 返回None：所有Reeds-Shepp路径都与障碍物碰撞
    # 3️⃣ 返回None：根本没有生成任何路径（if not paths返回）


def update_node_with_analytic_expansion(current, goal,
                                        c, ox, oy, kd_tree):
    """
    尝试用Reeds-Shepp曲线直接连接到goal，成功则返回最终路径
    混合A*关键优化：避免搜索数百个中间节点！
    """
    path = analytic_expansion(current, goal, ox, oy, kd_tree)
    # analytic_expansion()穷举48种RS路径，返回最优无碰撞方案或None

    if path:  # RS路径存在且无碰撞
        if show_animation:
            plt.plot(path.x, path.y)  # 绘制此RS弧线
        
        # 提取RS轨迹（去除起点，起点已在current中）
        f_x = path.x[1:]   # x坐标：29个点（30个点减去起点）
        f_y = path.y[1:]   # y坐标
        f_yaw = path.yaw[1:]  # 方向

        # RS路径成本累积
        f_cost = current.cost + calc_rs_path_cost(path)
        # = 起点到current的成本 + RS路径自身的成本
        
        f_parent_index = calc_index(current, c)
        # current在网格中的1维索引，用于路径回溯

        # RS轨迹方向标志
        fd = []  # boolean list: True=前进, False=倒车
        for d in path.directions[1:]:
            fd.append(d >= 0)  # path.directions[i] > 0→True

        # 创建最终路径节点（目标节点）
        f_steer = 0.0  # RS轨迹无需额外转向惩罚
        f_path = Node(current.x_index, current.y_index, current.yaw_index,
                      current.direction, f_x, f_y, f_yaw, fd,
                      cost=f_cost, parent_index=f_parent_index, steer=f_steer)
        # Node参数：起点网格位置（保持连接信息）
        # + RS轨迹坐标 + RS方向标志 + 成本和父指针
        
        return True, f_path  # 成功找到！返回最终节点

    return False, None  # 所有RS方案都碰撞，继续A*常规搜索


def calc_rs_path_cost(reed_shepp_path):
    """
    计算Reeds-Shepp路径的总成本
    
    成本构成（4大部分）：
    1️⃣ 距离成本  : 路径长度（倒车长度×5倍惩罚）
    2️⃣ 换向成本  : 每次前进↔倒车切换 +100
    3️⃣ 转向成本  : 曲线段（L弧或R弧）的转向惩罚
    4️⃣ 转向变化  : 相邻转向角的变化量（从左满舵到右满舵）
    """
    
    # ======================== 第1行：初始化成本累计器 ========================
    cost = 0.0
    # 从0开始累计所有惩罚，最后返回总成本
    
    
    # ======================== 第2-7行：距离成本（前进和倒车） ========================
    # Reeds-Shepp路径由多个"段"组成，每段是一个弧线或直线
    # reed_shepp_path.lengths = [1.0, -1.5, 2.0, -0.5, ...]
    # 正数 = 前进，负数 = 倒车
    
    for length in reed_shepp_path.lengths:
        # 遍历路径的每一段
        # length示例：1.0（前进1米）、-1.5（倒车1.5米）、2.0（前进2米）等
        
        if length >= 0:                    # 第3-4行：前进段的成本
            # 前进距离（正数）
            cost += length                 # 直接加上距离作为成本
                                           # 前进1米 → 成本+1
                                           # 前进2.5米 → 成本+2.5
                                           # 含义：正常前进，每米成本1
        
        else:                              # 第5-7行：倒车段的成本
            # 倒车距离（负数）
            cost += abs(length) * BACK_COST
            #                      ^^^^^^^^
            #                      BACK_COST = 5.0（倒车惩罚系数）
            #
            # abs(length)取绝对值，去掉负号
            # 倒车1米 → 成本 += abs(-1.0) × 5.0 = 5.0
            # 倒车1.5米 → 成本 += abs(-1.5) × 5.0 = 7.5
            #
            # 为什么要乘以5倍？
            # 倒车很不优雅，自动驾驶应该避免倒车
            # 倒车1米的"实际成本"视为前进5米的代价
            # 这激励路径规划器选择少倒车或不倒车的方案
    
    # 距离成本累计示例：
    # lengths = [2.0, -1.5, 3.0, -0.5]
    # 成本 = 2.0 + (1.5×5.0) + 3.0 + (0.5×5.0)
    #      = 2.0 + 7.5 + 3.0 + 2.5
    #      = 15.0
    
    
    # ======================== 第9-13行：换向惩罚（前进↔倒车切换） ========================
    # 从前进突然改为倒车，或从倒车突然改为前进，都很不优雅
    # 这个循环检测所有"相邻段方向不同"的位置
    
    for i in range(len(reed_shepp_path.lengths) - 1):
        # 遍历相邻的段对：(0,1), (1,2), (2,3), ...
        # range(3-1) = range(2) = [0, 1]
        # 所以检查 lengths[0]和lengths[1]、lengths[1]和lengths[2]
        #
        # 为什么是 len(...) - 1？
        # 如果有3段，索引为[0, 1, 2]
        # 相邻对有2对：(0,1)和(1,2)
        # 需要i从0到1，所以range(3-1)=range(2)=[0,1]
        
        # ─────────────────────── 第11行：判断是否换向 ──────────────────────
        if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
        # 相邻两段的长度相乘
        # 
        # 情况1：都是正数（都前进）
        #   1.0 × 2.0 = 2.0 > 0 ✓ 不换向
        # 
        # 情况2：一正一负（换向）
        #   1.0 × (-1.5) = -1.5 < 0 ✓ 检测到换向！
        #   (-1.5) × 2.0 = -3.0 < 0 ✓ 检测到换向！
        # 
        # 情况3：都是负数（都倒车）
        #   (-1.0) × (-2.0) = 2.0 > 0 ✓ 不换向
        #
        # 妙处：乘积的符号就能判断方向是否改变
        #      无需额外的符号检查
        
            cost += SB_COST                # 加上换向惩罚（SB=Switch Back）
            #        ^^^^^^^
            #        SB_COST = 100.0（非常大的惩罚！）
            #
            # 每次换向 → 成本+100
            # 这确保路径规划器只在必要时才倒车和换向
            #
            # 数值示例：
            # 如果路径中有2次换向（1次前进→倒车，1次倒车→前进）
            # SB_COST贡献 = 100.0 × 2 = 200.0（巨大的惩罚！）
    
    # 换向惩罚示例：
    # lengths = [2.0, -1.5, 3.0, -0.5]
    #                ^换向    ^换向  ^换向
    # 换向次数 = 3次
    # 换向惩罚 = 100.0 × 3 = 300.0
    
    
    # ======================== 第15-18行：转向成本（曲线段）========================
    # Reeds-Shepp路径由"直线段"(S)和"曲线段"(L/R)组成
    # L = 左弧（左转），R = 右弧（右转），S = 直线
    # 直线很便宜，曲线很贵（需要转向）
    
    for course_type in reed_shepp_path.ctypes:
        # 遍历路径的每一段的类型
        # ctypes示例：['L', 'S', 'R', 'L', 'S', ...]
        # L表示左弧段，S表示直线段，R表示右弧段
        #
        # 每个course_type是一个字符串：'L'、'S'或'R'
        
        if course_type != "S":             # 如果是曲线段（不是直线）
            #
            # course_type = 'L' 或 'R' → 需要转向 → 加惩罚
            # course_type = 'S' → 直线 → 不加惩罚
            
            cost += STEER_COST * abs(MAX_STEER)
            #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #       STEER_COST = 1.0
            #       MAX_STEER = 0.5 rad (最大转向角)
            #       abs(MAX_STEER) = 0.5
            # 
            # 每条曲线段 → 成本 += 1.0 × 0.5 = 0.5
            #
            # 为什么按 MAX_STEER 计算？
            # 因为L弧和R弧通常是在最大转向角下执行的
            # MAX_STEER是汽车的最大转向能力
            # 一次最大转向 = 成本0.5
            #
            # 数值示例：
            # 路径有5段：['L', 'S', 'R', 'L', 'S']
            # 曲线段有3个（'L', 'R', 'L'）
            # ctypes成本 = 0.5 × 3 = 1.5
    
    # 转向成本起什么作用？
    # 促进路径规划器选择包含更多直线(S)的路径
    # 因为直线不需要转向，成本更低
    # 有助于生成光滑的、转向较少的路径
    
    
    # ======================== 第20-24行：转向角变化成本（准备工作）========================
    # 这部分计算每一段对应的转向角，为下一部分准备数据
    
    n_ctypes = len(reed_shepp_path.ctypes)
    # 路径段的总数
    # ctypes = ['L', 'S', 'R', 'L', 'S']
    # n_ctypes = 5
    
    u_list = [0.0] * n_ctypes
    # 创建一个列表，长度=n_ctypes，所有元素初始为0.0
    # u_list = [0.0, 0.0, 0.0, 0.0, 0.0]（如果n_ctypes=5）
    #
    # u_list 的目的：存储每一段的转向角度
    # u = steering angle（转向角）
    # L弧 → MAX_STEER（左满舵）
    # R弧 → -MAX_STEER（右满舵）
    # S直线 → 0.0（不转向）
    
    for i in range(n_ctypes):
        # 遍历所有段，给每段赋予对应的转向角
        # i = 0, 1, 2, 3, 4
        
        if reed_shepp_path.ctypes[i] == "R":
            # 如果第i段是右弧(R)
            u_list[i] = - MAX_STEER         # 右转 → 转向角为负（-0.5 rad）
            #           ^^^^^^^^
            #           MAX_STEER = 0.5 rad
            #           右转的转向角是负数
            #
            # 为什么是负数？
            # 在汽车坐标系中：
            #   正的转向角 = 左转
            #   负的转向角 = 右转
        
        elif reed_shepp_path.ctypes[i] == "L":
            # 如果第i段是左弧(L)
            u_list[i] = MAX_STEER           # 左转 → 转向角为正（+0.5 rad）
            #           ^^^^^^^^
            #           正的转向角表示左转
        
        # 如果是 'S'（直线），u_list[i] 保持初始值 0.0
    
    # u_list 转换后的示例：
    # ctypes = ['L', 'S', 'R', 'L', 'S']
    # u_list = [0.5, 0.0, -0.5, 0.5, 0.0]
    
    
    # ======================== 第26-28行：转向角变化成本 ========================
    # 相邻两段的转向角变化越大，转向越不平滑，成本越高
    # 目的：生成光滑的转向，避免突然的大角度转向变化
    
    for i in range(len(reed_shepp_path.ctypes) - 1):
        # 遍历相邻的段对
        # 如果有5段，检查(0,1), (1,2), (2,3), (3,4)
        # i从0到3，共4次迭代
        
        cost += STEER_CHANGE_COST * abs(u_list[i + 1] - u_list[i])
        #       ^^^^^^^^^^^^^^^^^^
        #       STEER_CHANGE_COST = 5.0
        #
        # abs(u_list[i + 1] - u_list[i])
        # = 相邻两段转向角的差值的绝对值
        #
        # 数值示例1（平滑转向）：
        #   u_list[0] = 0.5（左转）
        #   u_list[1] = 0.0（直线）
        #   |u_list[1] - u_list[0]| = |0.0 - 0.5| = 0.5
        #   成本 += 5.0 × 0.5 = 2.5
        #
        # 数值示例2（急剧转向）：
        #   u_list[1] = 0.5（左转）
        #   u_list[2] = -0.5（右转）
        #   |u_list[2] - u_list[1]| = |-0.5 - 0.5| = 1.0（变化量很大！）
        #   成本 += 5.0 × 1.0 = 5.0（高惩罚）
        #
        # 含义：
        # - 从0.5→0.0的平滑转向变化 成本低
        # - 从0.5→-0.5的急剧转向变化 成本高
        # 
        # 这促使路径规划器选择更平滑的转向序列
        # 而不是在弧形之间急速切换方向
    
    # 转向变化成本示例：
    # u_list = [0.5, 0.0, -0.5, 0.5, 0.0]
    # 相邻差值 = |0.0-0.5| + |-0.5-0.0| + |0.5-(-0.5)| + |0.0-0.5|
    #          = 0.5 + 0.5 + 1.0 + 0.5 = 2.5
    # 成本 = 5.0 × 2.5 = 12.5
    
    
    # ======================== 第30行：返回总成本 ========================
    return cost
    #
    # 返回值：四部分成本的总和
    # total_cost = 距离成本 + 换向成本 + 转向成本 + 转向变化成本
    #
    # 完整数值示例：
    # 路径：lengths=[2.0, -1.5, 3.0]，ctypes=['L', 'S', 'L']
    # 
    # 1️⃣ 距离成本 = 2.0 + (1.5×5.0) + 3.0 = 2.0 + 7.5 + 3.0 = 12.5
    # 2️⃣ 换向成本 = 100.0 × 1 = 100.0（1次换向：正→负）
    # 3️⃣ 转向成本 = 0.5 × 2 = 1.0（2个曲线段L和L）
    # 4️⃣ u_list = [0.5, 0.0, 0.5]
    #    转向变化 = 5.0 × (|0.0-0.5| + |0.5-0.0|) = 5.0 × 1.0 = 5.0
    #
    # 总成本 = 12.5 + 100.0 + 1.0 + 5.0 = 118.5


def hybrid_a_star_planning(start, goal, ox, oy, xy_resolution, yaw_resolution):
    """
    混合A*路径规划：离散A*+ 连续Reeds-Shepp优化
    输入：起点、目标、障碍物、网格分辨率
    输出：最优路径(x,y,yaw轨迹+总成本)
    """
    
    # 角度归一化到[-pi, pi]范围，避免360度冗余
    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    
    # 障碍物副本（Config会修改，需保护原始输入）
    tox, toy = ox[:], oy[:]

    # KDTree加速碰撞检测：最近邻查询O(log N)
    obstacle_kd_tree = cKDTree(np.vstack((tox, toy)).T)

    # 初始化3D网格配置(31x31x26≈25000状态)
    config = Config(tox, toy, xy_resolution, yaw_resolution)

    # 起点Node：网格索引+轨迹+成本=0
    start_node = Node(round(start[0] / xy_resolution),
                      round(start[1] / xy_resolution),
                      round(start[2] / yaw_resolution), True,
                      [start[0]], [start[1]], [start[2]], [True], cost=0)
    
    # 目标Node：定义搜索的终止条件
    goal_node = Node(round(goal[0] / xy_resolution),
                     round(goal[1] / xy_resolution),
                     round(goal[2] / yaw_resolution), True,
                     [goal[0]], [goal[1]], [goal[2]], [True])

    # openList: 待探索{索引→Node}，closedList: 已探索{索引→Node}
    openList, closedList = {}, {}

    # 启发函数：DP距离表(2D网格到目标的下界距离) ***
    h_dp = calc_distance_heuristic(
        goal_node.x_list[-1], goal_node.y_list[-1],
        ox, oy, xy_resolution, BUBBLE_R)

    # 初始化堆(最小堆)
    pq = []
    openList[calc_index(start_node, config)] = start_node
    heapq.heappush(pq, (calc_cost(start_node, h_dp, config),
                        calc_index(start_node, config)))
    final_path = None

    while True:
        # Main A* search loop
        if not openList:
            print("Error: Cannot find path, No open set")
            return Path([], [], [], [], 0)

        cost, c_id = heapq.heappop(pq)
        # Pop minimum f(n) node from heap
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            # Expired heap entry (already replaced by better path)
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.x_list[-1], current.y_list[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        # Key optimization: Try Reeds-Shepp direct connection (can save hundreds of steps)
        is_updated, final_path = update_node_with_analytic_expansion(
            current, goal_node, config, ox, oy, obstacle_kd_tree)

        if is_updated:
            # RS path found! Search complete
            print("path found")
            break

        # Regular A*: Expand 42 neighbors (21 steering x 2 direction)
        for neighbor in get_neighbors(current, config, ox, oy,
                                      obstacle_kd_tree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor_index not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, final_path)
    return path


def calc_cost(n, h_dp, c):
    """
    计算A*搜索的含段估计成本 f(n) = g(n) + h(n)
    
    g(n) = 实际成本 = n.cost
    h(n) = 含段估计 = H_COST × 到目标的下界距离
    """
    
    # ════════════════════════════════════════════════════════════ 第1行：计算索引是什么是启发 ════════════════════════════════════════════════════════════
    ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
    # 计算在网格上的左右位置（忽略YAW维度）
    # 这是2D网格纯需法不消耗时间的启发
    # h_dp是动态规划下界表（每个XY位置到目标的最短距离）
    # 例如：
    # 节点布局：(5, 10), (6, 10), (7, 11), ...
    # h_dp[ind] = 从(5,10)到目标点的下界距离(Dijkstra预计)
    
    # ════════════════════════════════════════════════════════════ 第2-4行：离线网格灾祸检查 ════════════════════════════════════════════════════════════
    if ind not in h_dp:
    # h_dp是一个字典，包含所有有效XY位置的启发值
    # 如果ind不在其中，说明什么？
    # 原因有两种：
    # 1. 节点搜索超出了地图范围
    # 2. 节点网格索引计算错误
        
        return n.cost + 999999999  # 极府大的成本
        # 返回极大成本，便优先队列抛弃此节点
        # 999999999是年几下界值消耗的时间（很归的设计）
        # 寄武器：二一般旁虑了，是难达a*方法的网格可达性错武
    
    # ════════════════════════════════════════════════════════════ 第5行：格牌来泵虫算及包括启段 ════════════════════════════════════════════════════════════
    return n.cost + H_COST * h_dp[ind].cost
    #      ^^^^^^^   ^^^^^^
    #       g(n)     h(n)
    #
    # A*评估函数：f(n) = g(n) + h(n)
    #
    # g(n) = n.cost
    #   从起点到当前节点的实际费用
    #   已经确明的成本，每步马不会算错
    #
    # h(n) = H_COST × h_dp[ind].cost
    #   从当前节点到目标的支止估计（可能小五）
    #   h_dp[ind].cost = 动态规划下界距离
    #   H_COST = 5.0 (扩大计数一方所以优先搜索靠近目标的节点)
    #
    # 数值比较：
    # 节点5: n.cost=50  (到目标模拟距离=10)
    #   f(n5) = 50 + 5.0 × 10 = 100
    #
    # 节点6: n.cost=45  (到目标模拟距离=15)
    #   f(n6) = 45 + 5.0 × 15 = 120
    #
    # 鼠日f(n5) < f(n6)，n5先取出（趋惯同目标）


def get_final_path(closed, goal_node):
    """从目标节点反向回溯重构完整路径"""
    
    # 第1-4行：从目标节点提取轨迹并反转
    reversed_x, reversed_y, reversed_yaw = \
        list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
        list(reversed(goal_node.yaw_list))
    # A*从起点逐步前进，每个Node轨迹是一段连续路径
    # 从目标开始反向回溯，所以首次反转轨迹
    
    direction = list(reversed(goal_node.directions))
    nid = goal_node.parent_index
    final_cost = goal_node.cost

    # 第7-12行：反向遍历搜索树，串联所有轨迹
    while nid:
        # nid是整数索引，>0为True
        n = closed[nid] 


        # 从closed字典中取出前驱Nod
        reversed_x.extend(list(reversed(n.x_list)))
        reversed_y.extend(list(reversed(n.y_list)))
        reversed_yaw.extend(list(reversed(n.yaw_list)))
        # 反转该Node的轨迹并向前拼接
        direction.extend(list(reversed(n.directions)))

        nid = n.parent_index
        # 继续指向上一级Node

    # 第14-17行：最终正向恢复
    reversed_x = list(reversed(reversed_x))
    reversed_y = list(reversed(reversed_y))
    reversed_yaw = list(reversed(reversed_yaw))
    direction = list(reversed(direction))
    # 所有轨迹都已反转累积，最后再反转一次恢复正向
    # 第19行：调整起点端的方向 
    direction[0] = direction[1]
    # 起点的direction[0]可能不准确，用direction[1]覆盖 ？

    path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

    return path


def verify_index(node, c):
    """
    验证节点的网格索引是否在地图范围内
    
    只检查XY平面的范围，不检查YAW（因为YAW是圆周的）
    """
    
    # 第1-2行：提取XY网格索引
    x_ind, y_ind = node.x_index, node.y_index
    # 从节点对象中提取X和Y网格坐标
    # 示例：x_ind=5, y_ind=12（表示网格位置）
    
    # 第3行：范围检查
    if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
    # c是Config对象，包含网格的边界信息
    # c.min_x / c.max_x : X方向的网格索引范围（如 0 到 30）
    # c.min_y / c.max_y : Y方向的网格索引范围（如 0 到 30）
    #
    # 检查：是否 0 <= x_ind <= 30 且 0 <= y_ind <= 30
    #
    # 为什么只检查XY？
    # YAW是方向角，范围是 [-π, π]
    # 使用模运算处理循环，不需要范围检查
    # 例如：YAW = 3.5 rad → 相当于 3.5 - 2π ≈ -2.78 rad
        
        return True
        # 在范围内，返回True（有效节点）

    return False
    # 超出范围，返回False（无效节点）
    #
    # 使用场景（在get_neighbors和calc_next_node中）：
    # node = calc_next_node(...)
    # if node and verify_index(node, config):
    #     yield node  # 只返回有效的节点


def calc_index(node, c):
    """
    将3D网格索引(x_index, y_index, yaw_index)转换为1D线性索引 ？
    
    目的：用作字典键，快速查找节点
    公式：3D坐标 → 1D线性地址（类似于多维数组的行优先展开）
    """
    
    # ======================== 关键公式：3D → 1D转换 ========================
    ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
          (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)
    #
    # 这个公式的妙处：将3D网格展开成1D数组
    # 类比：照片扫描，从左到右，从上到下，一行一行
    #
    # 公式详解：
    # ─────────────────────────────────────────────────────────
    # (node.yaw_index - c.min_yaw) * c.x_w * c.y_w
    #   ↑ YAW维度贡献
    #   按YAW层数计算偏移
    #   每一YAW层包含 x_w × y_w 个网格单元
    #   示例：如果c.x_w=31, c.y_w=31，则每层有961个单元
    #
    # (node.y_index - c.min_y) * c.x_w
    #   ↑ Y维度贡献
    #   在当前YAW层内，按Y行计算
    #   每一行有 c.x_w 个单元
    #
    # (node.x_index - c.min_x)
    #   ↑ X维度贡献
    #   在当前Y行内的具体位置
    #
    # ─────────────────────────────────────────────────────────
    # 为什么要减去min值？
    # 因为Config中存储的是绝对网格坐标（如0-30）
    # 需要转换成相对索引（如0-30变成0-960等）
    #
    # 数值示例：
    # Config参数：
    #   min_x=0, max_x=30, x_w=31
    #   min_y=0, max_y=30, y_w=31
    #   min_yaw=-13, max_yaw=12, yaw_w=26
    #
    # Node位置：x_index=5, y_index=10, yaw_index=3
    #
    # 计算：
    #   yaw贡献 = (3 - (-13)) * 31 * 31 = 16 * 961 = 15,376
    #   y贡献 = (10 - 0) * 31 = 310
    #   x贡献 = (5 - 0) = 5
    #   总索引 = 15,376 + 310 + 5 = 15,691
    #
    # 验证：这个索引唯一对应网格位置(5, 10, 3)
    
    if ind <= 0:
        # 错误检查：索引不应该≤0
        print("Error(calc_index):", ind)
        #
        # 何时出现此错误？
        # 1. 节点网格索引超出范围（负数索引）
        # 2. 公式计算错误
        # 3. Config初始化有问题
        #
        # 调试建议：
        # print(f"node.yaw_index={node.yaw_index}, c.min_yaw={c.min_yaw}")
        # print(f"node.y_index={node.y_index}, c.min_y={c.min_y}")
        # print(f"node.x_index={node.x_index}, c.min_x={c.min_x}")

    return ind
    # 返回转换后的1D索引
    #
    # 使用场景：
    # 1. 在openList和closedList中作为字典键
    # 2. 在Node的parent_index中存储父节点的索引
    # 3. 实现O(1)的节点查询效率
    #
    # 在hybrid_a_star_planning中的使用：
    # openList[calc_index(start_node, config)] = start_node
    # neighbor_index = calc_index(neighbor, config)
    # if neighbor_index in closedList:  # 快速查询


def main():
    """
    混合A*算法的演示程序
    
    场景：60×60米的矩形环境，中间有两条竖墙障碍
    任务：从左下(10,10)规划到右上(50,50)，同时改变车指向
    """
    print("Start Hybrid A* planning")

    # ════════════════════════════════════ 第1部分：构建环境（60×60矩形+内部障碍） ════════════════════════════════════
    ox, oy = [], []
    # ox, oy: 障碍物坐标列表

    # 第1段：下边界 (0,0) → (60,0)
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    # 第2段：右边界 (60,0) → (60,60)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    # 第3段：上边界 (60,60) → (0,60)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    # 第4段：左边界 (0,60) → (0,0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    # 构成了一个封闭的矩形边界，共约240个障碍物点

    # ────────────────────────────────── 内部障碍物：两条竖墙 ──────────────────────────────
    # 竖墙1: x=20, 从y=0到y=40
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    # 竖墙2: x=40, 从y=60向下到y=20 (倒序)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)
    # 这两堵墙强制车辆的规划路径必须在特定的通道中行驶
    # 总障碍物数：约240(边界) + 80(内部墙) ≈ 320点

    # ════════════════════════════════════ 第2部分：设定起点和目标 ════════════════════════════════════
    start = [10.0, 10.0, np.deg2rad(90.0)]
    # 起点：(10, 10)，方向90° = π/2 rad（指向北方+Y）
    
    goal = [50.0, 50.0, np.deg2rad(-90.0)]
    # 目标：(50, 50)，方向-90° = -π/2 rad（指向南方-Y）
    # 注意：车不仅要到达(50,50)，还要改变方向从北指向南！

    print("start : ", start)
    print("goal : ", goal)

    # ════════════════════════════════════ 第3部分：可视化初始场景（可选） ════════════════════════════════════
    if show_animation:
        plt.plot(ox, oy, ".k")  # 黑点绘制障碍物
        rs.plot_arrow(start[0], start[1], start[2], fc='g')  # 绿箭头：起点和方向
        rs.plot_arrow(goal[0], goal[1], goal[2])  # 红箭头：目标和目标方向

        plt.grid(True)  # 网格
        plt.axis("equal")  # 等长坐标轴

    # ════════════════════════════════════ 第4部分：调用混合A*算法 ════════════════════════════════════
    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    # 调用混合A*规划函数
    # 输入：起点[x,y,yaw]、目标[x,y,yaw]、障碍物列表、网格分辨率
    # 输出：Path对象 (包含x_list, y_list, yaw_list, directions, cost)

    # 提取路径的三个坐标序列
    x = path.x_list  # x坐标序列（米）
    y = path.y_list  # y坐标序列（米）
    yaw = path.yaw_list  # 方向序列（弧度）
    # 每个列表都有相同的长度（路径上的采样点个数）

    # ════════════════════════════════════ 第5部分：可视化规划结果与动画播放 ════════════════════════════════════
    if show_animation:
        # 动画：逐帧绘制车的运动轨迹
        for i_x, i_y, i_yaw in zip(x, y, yaw):
            # 每次迭代代表车在路径上的一个位置
            
            plt.cla()  # 清除前一帧
            plt.plot(ox, oy, ".k")  # 重新绘制障碍物
            plt.plot(x, y, "-r", label="Hybrid A* path")  # 红线显示完整路径
            plt.grid(True)
            plt.axis("equal")
            plot_car(i_x, i_y, i_yaw)  # 绘制该位置的车矩形和方向箭头
            plt.pause(0.0001)  # 延迟，实现逐帧动画（1帧≈0.0001秒）

    print(__file__ + " done!!")


if __name__ == '__main__':
    main()
