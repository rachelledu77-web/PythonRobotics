"""

Reeds Shepp path planner sample code

author Atsushi Sakai(@Atsushi_twi)
co-author Videh Patel(@videh25) : Added the missing RS paths

"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math

import matplotlib.pyplot as plt
import numpy as np
from utils.angle import angle_mod

show_animation = True


class Path:
    """
    Reeds-Shepp路径容器：存储最优曲线路径的所有信息
    
    Reeds-Shepp曲线由最多5个段组成，每段是圆弧(L/R)或直线(S)
    例如：LSR表示左转→直线→右转的三段路径
    """

    def __init__(self):
        # ════════════════════════════════════ 路径段信息 ════════════════════════════════════
        self.lengths = []  # 每段的长度列表（负值表示倒车段）
        # 示例：[1.5, 2.0, 0.8] 表示三段，长度分别为1.5, 2.0, 0.8米
        # 负值示例：[-1.5, 0.8] 表示第一段倒车，长度1.5米
        
        self.ctypes = []  # 路径类型：'S'直线，'L'左转(圆弧)，'R'右转(圆弧)
        # 示例：['L', 'S', 'R'] 表示LSR模式（最常见）
        # 最多5个元素，对应最复杂的CCSCC型(5段)  
        
        self.L = 0.0  # 总路径长度（所有段长度之和的绝对值）
        # 用于选择最优路径：L值最小的路径为最优
        
        # ════════════════════════════════════ 轨迹采样点 ════════════════════════════════════
        self.x = []  # X坐标序列（全局坐标系，米）
        self.y = []  # Y坐标序列（全局坐标系，米）
        self.yaw = []  # 方向角序列（全局坐标系，弧度）
        self.directions = []  # 方向标志序列（1=前进，-1=倒车）


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    绘制方向箭头
    
    可以绘制单个点或多个点的方向向量
    用于可视化起点、终点和路径采样点的方向
    """
    if isinstance(x, list):  # x是列表，批量绘制
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)  # 递归调用绘制每个点
    else:  # x是单个值，绘制单个箭头
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), 
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


def pi_2_pi(x):
    """
    将角度归一化到[-π, π]范围
    
    目的：避免角度重复表示，例如360°=0°、-180°=180°
    """
    return angle_mod(x)

def mod2pi(x):
    """
    角度模运算到2π范围，与C++的fmod一致
    
    先进行2π模运算，再调整到[-π, π]区间
    用于保证RS路径几何计算的角度有效性
    """
    # 第1步：模运算，结果范围约为[-2π, 2π]
    v = np.mod(x, np.copysign(2.0 * math.pi, x))
    # copysign保持x的符号，处理负角度
    
    # 第2步：调整到[-π, π]范围
    if v < -math.pi:
        v += 2.0 * math.pi  # 负角度补偿
    else:
        if v > math.pi:
            v -= 2.0 * math.pi  # 正角度补偿
    return v

def set_path(paths, lengths, ctypes, step_size):
    """
    将新计算的RS路径添加到列表中（去重并检查有效性）
    
    验证：
    1. 相同类型+相同长度的路径不重复添加
    2. 路径长度必须大于最小步长（至少能走一步）
    """
    path = Path()  # 创建新Path对象
    path.ctypes = ctypes  # 设置路径段类型：['L', 'S', 'R']
    path.lengths = lengths  # 设置路径段长度
    path.L = sum(np.abs(lengths))  # 计算总长度（所有段的绝对值之和）
    
    # 检查是否已存在完全相同的路径
    for i_path in paths:
        type_is_same = (i_path.ctypes == path.ctypes)  # 路径类型相同（例都是LSR）
        length_is_close = (sum(np.abs(i_path.lengths)) - path.L) <= step_size
        # 总长度接近（差距≤最小步长）
        if type_is_same and length_is_close:
            return paths  # 找到相同路径，不添加

    # 检查路径长度是否足够大（至少能走一步）
    if path.L <= step_size:
        return paths  # 路径太短，舍弃

    # 通过所有检查，添加到列表
    paths.append(path)
    return paths


def polar(x, y):
    """
    笛卡尔坐标转极坐标
    
    输入：(x, y) 笛卡尔坐标
    输出：(r, θ) 极坐标（半径和角度）
    
    用途：RS路径几何计算中大量使用极坐标
    """
    r = math.hypot(x, y)  # 半径 = sqrt(x² + y²)
    theta = math.atan2(y, x)  # 角度 = atan2(y, x) ∈ [-π, π]
    return r, theta


# ════════════════════════════════════════════════════════════════════════════════════════════════
# 14种Reeds-Shepp（RS）路径类型的求解函数
# ════════════════════════════════════════════════════════════════════════════════════════════════
# 这些函数根据给定的目标位置和方向角，计算对应RS曲线类型的段长度和方向。
# 
# 14种分类（按几何段数）：
#   CSC (2种):  LSL, LSR - 两端圆弧中间直线
#   CCC (3种):  LRL, RLR等 - 三个相连圆弧，S形或Z形
#   CCCC (2种): LRLR, RLRL - 四段圆弧
#   CCSC (2种): LRSL, LRSR等 - 圆→圆→直→圆
#   CSCC (2种): LSLR, LSRL等 - 圆→直→圆→圆
#   CCSCC (1种): LRLSR等 - 最复杂，5段路径
# 
# 参数：x, y 标准化坐标；phi 方向变化角
# 返回：(可行性, [段长], [段类型])\n# ════════════════════════════════════════════════════════════════════════════════════════════════\n\ndef left_straight_left(x, y, phi):
    u, t = polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return True, [t, u, v], ['L', 'S', 'L']

    return False, [], []


def left_straight_right(x, y, phi):
    u1, t1 = polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 ** 2
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)

        if (t >= 0.0) and (v >= 0.0):
            return True, [t, u, v], ['L', 'S', 'R']

    return False, [], []


def left_x_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2 * A)
        v = mod2pi(phi - t - u)
        return True, [t, -u, v], ['L', 'R', 'L']

    return False, [], []


def left_x_right_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        A = math.acos(0.25 * u1)
        t = mod2pi(A + theta + math.pi/2)
        u = mod2pi(math.pi - 2*A)
        v = mod2pi(-phi + t + u)
        return True, [t, -u, -v], ['L', 'R', 'L']

    return False, [], []


def left_right_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 <= 4.0:
        u = math.acos(1 - u1**2 * 0.125)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(-A + theta + math.pi/2)
        v = mod2pi(t - u - phi)
        return True, [t, u, -v], ['L', 'R', 'L']

    return False, [], []


def left_right_x_left_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    # Solutions refering to (2 < u1 <= 4) are considered sub-optimal in paper
    # Solutions do not exist for u1 > 4
    if u1 <= 2:
        A = math.acos((u1 + 2) * 0.25)
        t = mod2pi(theta + A + math.pi/2)
        u = mod2pi(A)
        v = mod2pi(phi - t + 2*u)
        if ((t >= 0) and (u >= 0) and (v >= 0)):
            return True, [t, u, -u, -v], ['L', 'R', 'L', 'R']

    return False, [], []


def left_x_right_left_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)
    u2 = (20 - u1**2) / 16

    if (0 <= u2 <= 1):
        u = math.acos(u2)
        A = math.asin(2 * math.sin(u) / u1)
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -u, -u, v], ['L', 'R', 'L', 'R']

    return False, [], []


def left_x_right90_straight_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi + math.pi/2)
        if (t >= 0) and (v >= 0):
           return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'L']

    return False, [], []


def left_straight_right90_x_left(x, y, phi):
    zeta = x - math.sin(phi)
    eeta = y - 1 + math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        u = math.sqrt(u1**2 - 4) - 2
        A = math.atan2(math.sqrt(u1**2 - 4), 2)
        t = mod2pi(theta - A + math.pi/2)
        v = mod2pi(t - phi - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'R', 'L']

    return False, [], []


def left_x_right90_straight_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta + math.pi/2)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -v], ['L', 'R', 'S', 'R']

    return False, [], []


def left_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 2.0:
        t = mod2pi(theta)
        u = u1 - 2
        v = mod2pi(phi - t - math.pi/2)
        if (t >= 0) and (v >= 0):
            return True, [t, u, math.pi/2, -v], ['L', 'S', 'L', 'R']

    return False, [], []


def left_x_right90_straight_left90_x_right(x, y, phi):
    zeta = x + math.sin(phi)
    eeta = y - 1 - math.cos(phi)
    u1, theta = polar(zeta, eeta)

    if u1 >= 4.0:
        u = math.sqrt(u1**2 - 4) - 4
        A = math.atan2(2, math.sqrt(u1**2 - 4))
        t = mod2pi(theta + A + math.pi/2)
        v = mod2pi(t - phi)
        if (t >= 0) and (v >= 0):
            return True, [t, -math.pi/2, -u, -math.pi/2, v], ['L', 'R', 'S', 'L', 'R']

    return False, [], []


def timeflip(travel_distances):
    \"\"\"\n    时间反演变换：将路径的所有段长度反向\n    \n    目的：实现从目标点看回起点的路径（对应\"倒车\"运动）\n    \n    输入：\n      travel_distances: 原路径的各段长度 [t1, t2, t3, ...]\n    \n    输出：\n      反向后的各段长度 [-t1, -t2, -t3, ...]\n    \n    原理：\n      负长度表示反向运动（倒车）\n      例如：[1.5, 2.0, -0.8] → [-1.5, -2.0, 0.8]（倒序运动）\n    \n    使用场景（generate_path中）：\n      从终点反向规划到起点，然后timeflip得到真实路径\n      这样可以同时覆盖\"前进→前进\"和\"前进→倒车\"的所有组合\n    \"\"\"\n    return [-x for x in travel_distances]


def reflect(steering_directions):\n    \"\"\"\n    镜像变换：交换左右方向（L ↔ R）\n    \n    目的：实现横向镜像路径（对称地覆盖另一侧的方案）\n    \n    输入：\n      steering_directions: 原路径各段的转向方向 ['L', 'S', 'R', ...]\n    \n    输出：\n      镜像后的转向方向 ['R', 'S', 'L', ...]\n      其中 L和R互换，S保持不变（直线无方向）\n    \n    原理：\n      由于对称性，左转的RS曲线可以通过镜像得到右转的曲线\n      例如：LSL(L左转→S直行→L左转) 镜像后为 RSR(R右转→S直行→R右转)\n    \n    使用场景（generate_path中）：\n      坐标镜像（正Y→负Y）后的路径需要反向所有转向指令\n      这样可以覆盖4个象限的所有情况\n    \"\"\"\n    def switch_dir(dirn):\n        \"\"\"单个方向交换器\"\"\"\n        if dirn == 'L':\n            return 'R'  # 左→右\n        elif dirn == 'R':\n            return 'L'  # 右→左\n        else:\n            return 'S'  # 直线保持\n    return[switch_dir(dirn) for dirn in steering_directions]


def generate_path(q0, q1, max_curvature, step_size):
    """
    生成从起点到终点的所有可能的Reeds-Shepp路径（最多14种结构）
    
    输入：
      q0: 起点[x, y, yaw] (米、米、弧度)
      q1: 终点[x, y, yaw] (米、米、弧度)
      max_curvature: 最大曲率 = 1/轴距 (1/米)
      step_size: 采样步长 (米)
    
    输出：Path对象列表（包含所有可行的RS路径）
    
    原理：
    1. 坐标系变换：将问题转换到以起点为原点的局部坐标系
    2. 曲率标准化：乘以max_curvature使轴距=1（简化计算）
    3. 穷举14种RS路径类型，包括对称变换（镜像、反向）
    
    RS路径类型（14种）：
    - CSC: 3段（圆-直-圆）  → LSL, LSR等 (2种)
    - CCC: 3段（圆-圆-圆）  → LRL等 (3种)
    - CCCC: 4段 (2种)
    - CCSC: 4段 (2种)
    - CSCC: 4段 (2种)
    - CCSCC: 5段 (1种，最复杂)
    """
    # 计算起点到终点的相对位置和方向
    dx = q1[0] - q0[0]  # X方向距离
    dy = q1[1] - q0[1]  # Y方向距离 
    dth = q1[2] - q0[2]  # 方向角变化
    
    # 起点方向的三角函数值
    c = math.cos(q0[2])  # cos(yaw₀)
    s = math.sin(q0[2])  # sin(yaw₀)
    
    # 坐标变换到以起点为原点、起点方向为x轴的局部坐标系
    # 旋转矩阵[cos(yaw) sin(yaw); -sin(yaw) cos(yaw)]作用后，再乘以max_curvature标准化
    x = (c * dx + s * dy) * max_curvature  # 标准化X坐标
    y = (-s * dx + c * dy) * max_curvature  # 标准化Y坐标
    step_size *= max_curvature  # 标准化步长

    paths = []  # 存储所有可行的RS路径
    
    # 14种RS路径对应的计算函数
    path_functions = [
        left_straight_left, left_straight_right,  # CSC类型 (2种)
        left_x_right_x_left, left_x_right_left, left_right_x_left,  # CCC类型 (3种)
        left_right_x_left_right, left_x_right_left_x_right,  # CCCC类型 (2种)
        left_x_right90_straight_left, left_x_right90_straight_right,  # CCSC类型 (2种)
        left_straight_right90_x_left, left_straight_left90_x_right,  # CSCC类型 (2种)
        left_x_right90_straight_left90_x_right  # CCSCC类型 (1种)
    ]

    # 对每种路径类型，尝试4种对称变换
    # 原因：通过对称可以覆盖目标在各个方向的情况
    for path_func in path_functions:
        # ────────────────────────────── 变换1：原始方向 ──────────────────────────────
        flag, travel_distances, steering_dirns = path_func(x, y, dth)
        if flag:  # 该变换下路径存在
            # 检查步长合理性（避免采样点过疏导致碰撞检测不充分）
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        # ────────────────────────────── 变换2：X反向，方向反向 ──────────────────────────────
        # 对应：从目标点看回起点的路径
        flag, travel_distances, steering_dirns = path_func(-x, y, -dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)  # 反向时间轴（倒序运动）
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        # ────────────────────────────── 变换3：Y反向（镜像） ──────────────────────────────
        # 对应：关于X轴的镜像路径（左右互换）
        flag, travel_distances, steering_dirns = path_func(x, -y, -dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            steering_dirns = reflect(steering_dirns)  # L↔R互换（镜像变换）
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

        # ────────────────────────────── 变换4：X和Y都反向 ──────────────────────────────
        # 综合变换1和3
        flag, travel_distances, steering_dirns = path_func(-x, -y, dth)
        if flag:
            for distance in travel_distances:
                if (0.1*sum([abs(d) for d in travel_distances]) < abs(distance) < step_size):
                    print("Step size too large for Reeds-Shepp paths.")
                    return []
            travel_distances = timeflip(travel_distances)  # 反向
            steering_dirns = reflect(steering_dirns)  # 镜像
            paths = set_path(paths, travel_distances, steering_dirns, step_size)

    return paths  # 返回所有可行路径列表


def calc_interpolate_dists_list(lengths, step_size):
    """
    计算每段路径的采样点列表
    
    输入：
      lengths: 路径段长度列表（可能含负值=倒车）
      step_size: 采样步长（米）
    
    输出：采样距离列表的列表
    
    例：lengths=[1.5, 2.0, 0.8], step_size=0.5
    → [[0, 0.5, 1.0, 1.5], [0, 0.5, 1.0, 1.5, 2.0], [0, 0.5, 0.8]]
    """
    interpolate_dists_list = []  # 所有采样距离
    for length in lengths:
        # 根据方向（正/负）设定采样方向
        d_dist = step_size if length >= 0.0 else -step_size
        # 从0到length，每step_size采样一次
        interp_dists = np.arange(0.0, length, d_dist)
        # 确保末端点包含在内（可能不是整数倍）
        interp_dists = np.append(interp_dists, length)
        interpolate_dists_list.append(interp_dists)

    return interpolate_dists_list


def generate_local_course(lengths, modes, max_curvature, step_size):
    """
    生成局部（起点为原点）的完整路径轨迹
    
    输入：
      lengths: 路径段长度列表
      modes: 路径段类型列表（'S', 'L', 'R'）
      max_curvature: 最大曲率
      step_size: 采样步长
    
    输出：
      xs, ys, yaws, directions: 轨迹上每个采样点的坐标
    
    流程：
    1. 为每段计算采样点
    2. 依次插值得到全局轨迹
    3. 更新起点为当前段终点（继续下一段）
    """
    # 计算每段的采样距离列表
    interpolate_dists_list = calc_interpolate_dists_list(lengths, step_size * max_curvature)

    origin_x, origin_y, origin_yaw = 0.0, 0.0, 0.0  # 起点为原点

    xs, ys, yaws, directions = [], [], [], []  # 采样点列表
    
    # 遍历所有路径段
    for (interp_dists, mode, length) in zip(interpolate_dists_list, modes,
                                            lengths):
        # 该段内的所有采样距离
        for dist in interp_dists:
            # 在该段上插值得到采样点
            x, y, yaw, direction = interpolate(dist, length, mode,
                                               max_curvature, origin_x,
                                               origin_y, origin_yaw)
            xs.append(x)
            ys.append(y)
            yaws.append(yaw)
            directions.append(direction)
        
        # 更新起点为该段的终点（为下一段做准备）
        origin_x = xs[-1]  # 最后一个点的X
        origin_y = ys[-1]  # 最后一个点的Y
        origin_yaw = yaws[-1]  # 最后一个点的方向

    return xs, ys, yaws, directions


def interpolate(dist, length, mode, max_curvature, origin_x, origin_y,
                origin_yaw):
    """
    在单个路径段上进行点的插值
    
    输入：
      dist: 沿路径的距离（0到length）
      length: 该段总长度
      mode: 段类型（'S'直线、'L'左转、'R'右转）
      max_curvature: 最大曲率
      origin_x, origin_y, origin_yaw: 该段起点
    
    输出：
      x, y, yaw: 该位置的坐标和方向
      direction: 运动方向（1=前进，-1=倒车）
    
    原理：使用运动学方程逐步计算轨迹
    """
    if mode == "S":  # 直线段
        # 沿当前方向走dist距离
        x = origin_x + dist / max_curvature * math.cos(origin_yaw)
        y = origin_y + dist / max_curvature * math.sin(origin_yaw)
        yaw = origin_yaw  # 方向不变
    
    else:  # 圆弧段（L或R）
        # Ackermann车模型的圆弧运动
        # 在局部坐标系中计算
        ldx = math.sin(dist) / max_curvature
        ldy = 0.0
        yaw = None  # 待计算
        
        if mode == "L":  # 左转（反时针）
            ldy = (1.0 - math.cos(dist)) / max_curvature  # 圆心在左侧
            yaw = origin_yaw + dist  # 方向角增加dist
        
        elif mode == "R":  # 右转（顺时针）
            ldy = (1.0 - math.cos(dist)) / -max_curvature  # 圆心在右侧
            yaw = origin_yaw - dist  # 方向角减少dist
        
        # 坐标系变换：从局部坐标系转换回全局坐标系
        # 旋转矩阵作用于(ldx, ldy)
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        
        x = origin_x + gdx
        y = origin_y + gdy

    # 运动方向：length > 0为前进，否则为倒车
    return x, y, yaw, 1 if length > 0.0 else -1


def calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size):
    """
    计算所有可行的Reeds-Shepp路径，包括坐标变换和轨迹采样
    
    输入：
      sx, sy, syaw: 起点[x, y, yaw]
      gx, gy, gyaw: 终点[x, y, yaw]  
      maxc: 最大曲率(1/轴距)
      step_size: 采样步长(米)
    
    输出：Path对象列表（每个Path包含完整的全局坐标轨迹）
    
    流程：
    1. generate_path(): 生成所有可能的RS路径结构和长度
    2. 对每条路径，generate_local_course(): 采样轨迹点
    3. 坐标变换: 从局部坐标系转换到全局坐标系
    """
    q0 = [sx, sy, syaw]  # 起点
    q1 = [gx, gy, gyaw]  # 终点

    # 生成所有可能的RS路径（最多14种结构）
    paths = generate_path(q0, q1, maxc, step_size)
    
    # 对每条路径生成完整的轨迹采样点
    for path in paths:
        # 在局部坐标系中生成轨迹
        xs, ys, yaws, directions = generate_local_course(
            path.lengths,  # 路径段长度
            path.ctypes,   # 路径段类型（S/L/R）
            maxc,          # 最大曲率
            step_size      # 采样步长
        )

        # ════════════════════════════════ 坐标系变换：局部→全局 ════════════════════════════════
        # 旋转：局部坐标系以起点方向为x轴，需要旋转回全局坐标系
        # 公式：[cos(θ) sin(θ); -sin(θ) cos(θ)] × [x; y] + [x₀; y₀]
        path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] 
                  for (ix, iy) in zip(xs, ys)]
        path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] 
                  for (ix, iy) in zip(xs, ys)]
        # 方向角也需要加上起点的方向
        path.yaw = [pi_2_pi(yaw + q0[2]) for yaw in yaws]
        # 保存方向标志
        path.directions = directions
        
        # ════════════════════════════════ 反标准化：长度还原回实际值 ════════════════════════════════
        # 之前乘以max_curvature进行标准化，现在需要除以恢复
        path.lengths = [length / maxc for length in path.lengths]
        path.L = path.L / maxc  # 总长度也要反标准化

    return paths  # 返回所有轨迹已采样的路径


def reeds_shepp_path_planning(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=0.2):
    """
    Reeds-Shepp路径规划的对外接口
    
    最优曲线计算：从所有可行RS路径中选择长度最短的
    
    输入：
      sx, sy, syaw: 起点[x, y, yaw] (米、米、弧度)
      gx, gy, gyaw: 目标[x, y, yaw] (米、米、弧度)
      maxc: 最大曲率 = 1/L (L是轴距，单位米)
      step_size: [可选] 采样步长，默认0.2米
    
    输出：
      返回5元组 (xs, ys, yaws, modes, lengths) 或 (None, None, None, None, None)
      xs, ys, yaws: 路径上所有采样点的坐标和方向
      modes: 路径段类型列表 ['L', 'S', 'R', ...]
      lengths: 路径段长度列表 [1.2, 2.3, 0.8, ...]
    
    使用示例：
      xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(
          0, 0, 0.0,      # 起点 (0,0)米，0°方向
          5, 5, np.pi/2,  # 终点 (5,5)米，90°方向
          0.1,            # 轴距1米 (maxc=1/10=0.1)
          0.05            # 采样步长0.05米
      )
    """
    # 计算所有可行路径
    paths = calc_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size)
    
    if not paths:
        # 没有可行路径（通常表示无解或参数错误）
        return None, None, None, None, None

    # 选择长度最短的路径（最优路径）
    # RS路径长度定义为所有段长度的绝对值之和
    best_path_index = paths.index(min(paths, key=lambda p: abs(p.L)))
    b_path = paths[best_path_index]

    # 返回最优路径的坐标和参数
    return b_path.x, b_path.y, b_path.yaw, b_path.ctypes, b_path.lengths


def main():
    """
    Reeds-Shepp路径规划演示程序
    
    任务：从(-1, -4)规划到(5, 5)，同时改变方向从-20°到+25°
    
    演示内容：
    1. 调用规划函数
    2. 绘制最优路径
    3. 显示路径的段类型和长度
    """
    print("Reeds Shepp path planner sample start!!")

    # ════════════════════════════════ 设定起点和终点 ════════════════════════════════
    start_x = -1.0  # 起点X坐标 [m]
    start_y = -4.0  # 起点Y坐标 [m]
    start_yaw = np.deg2rad(-20.0)  # 起点方向 -20° [rad]

    end_x = 5.0  # 终点X坐标 [m]
    end_y = 5.0  # 终点Y坐标 [m]
    end_yaw = np.deg2rad(25.0)  # 终点方向 +25° [rad]

    # ════════════════════════════════ 规划参数 ════════════════════════════════
    curvature = 0.1  # 最大曲率 = 1/轴距 (轴距10米)
    step_size = 0.05  # 采样步长 [m]

    # ════════════════════════════════ 调用规划函数 ════════════════════════════════
    xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(
        start_x, start_y, start_yaw,  # 起点
        end_x, end_y, end_yaw,        # 终点
        curvature,                    # 最大曲率
        step_size                     # 采样步长
    )

    # ════════════════════════════════ 检查规划结果 ════════════════════════════════
    if not xs:
        assert False, "No path"  # 规划失败

    # ════════════════════════════════ 可视化 ════════════════════════════════
    if show_animation:  # pragma: no cover
        plt.cla()  # 清除前一个图
        plt.plot(xs, ys, label="final course " + str(modes))  # 绘制路径
        print(f"{lengths=}")  # 打印各段长度

        # 绘制起点和终点的方向
        plot_arrow(start_x, start_y, start_yaw, fc='g')
        plot_arrow(end_x, end_y, end_yaw, fc='r')

        plt.legend()
        plt.grid(True)
        plt.axis("equal")  # 等长坐标轴
        plt.show()


if __name__ == '__main__':
    main()
