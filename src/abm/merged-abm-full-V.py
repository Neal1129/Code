"""
用于微流控芯片中 run-and-tumble 细菌的合并 ABM（基于智能体模型），包含：
  - 基于区域的流场与界面穿越要求（来自 ABM.py）
  - 可选的黏液黏度剖面，对运动参数进行重标定（来自 abm_light_mucus_v2）
  - 可选趋化模块（记忆/适应模型）（来自 abm_light_mucus_v2）
  - 光控：开环（连续/脉冲）与闭环门控控制器（来自 abm_light_mucus_v2）
  - 双通道光遗传直接映射（蓝光降低 tumble，绿光提高 tumble）（来自 ABM.py）
  - CheY 风格内部状态映射（light -> c -> tumble rate）（来自 abm_light_mucus_v2）

依赖：numpy, matplotlib；可选 tqdm（进度条）、imageio（视频合成）、ffmpeg（备选）
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from abm_metrics import summarize_transport
from light_response import effective_light_input

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *, total=None, desc=None, unit=None, **kwargs):
        if iterable is None:
            return _SimpleProgress(total=total, desc=desc, unit=unit)
        return iterable


class _SimpleProgress:
    """tqdm 不可用时的简易进度条（按帧更新）。"""

    def __init__(self, *, total: int, desc: str | None = None, unit: str | None = None) -> None:
        self.total = int(total)
        self.desc = desc or "进度"
        self.unit = unit or "it"
        self.n = 0
        print(f"{self.desc}: 0/{self.total} {self.unit}")

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.n == self.total or self.n % max(1, self.total // 20) == 0:
            print(f"{self.desc}: {self.n}/{self.total} {self.unit}")

    def close(self) -> None:
        if self.n < self.total:
            print(f"{self.desc}: {self.total}/{self.total} {self.unit} (完成)")


# =============================================================================
# 0) 全部超参数（集中放在文件前部）
# =============================================================================

@dataclass
class HyperParams:
    # -----------------------------
    # 复现实验 / 规模
    # -----------------------------
    SEED: int = 7
    N: int = 500               #细菌数量
    T_S: float = 300.0          # 模拟总时长 (s)
    FRAME_STEP_S: float = 1.0   # 输出时间步：每隔多少秒保存一帧结果
    DT_S: float = 0.05          # 积分时间步 (s)

    # -----------------------------
    # 输出
    # -----------------------------
    OUTPUT_DIR: str = str(Path.home() / "Desktop" / "Merged_ABM_outputs")
    SAVE_FRAMES: bool = True              # 按 FRAME_STEP_S 保存逐秒帧
    FRAMES_SUBDIR: str = "frames"         # 帧结果子目录名
    FRAME_PNG_DPI: int = 150              # 单帧 PNG 分辨率（视频用，不必与终图相同）
    MAKE_VIDEO: bool = True               # 仿真结束后将各分析图帧自动合成为视频
    VIDEO_FPS: float = 30.0               # 输出视频帧率（30 帧/秒）
    VIDEOS_SUBDIR: str = "videos"           # 9 个 MP4 的输出目录
    PLOT_FRAMES_SUBDIR: str = "frames/plots"  # 各分析图逐秒 PNG 子目录
    SHOW_PLOTS: bool = False  # False: 只保存图像，不弹出显示窗口
    N_PLOT: int = 120
    WINDOW_S: float = 0.5  # 绘图平滑窗口
    X_BINS_HEATMAP: int = 80
    Y_BINS_HEATMAP: int = 80
    XT_BINS_X: int = 80
    STATE_SAMPLE_STRIDE: int = 5
    THETA_HIST_BINS: int = 36

    # -----------------------------
    # 几何范围（芯片窗口）
    # -----------------------------
    X_MIN: float = 0.0
    X_MAX: float = 600.0
    Y_MIN: float = -8000.0   # um（底部，8 mm 高度）
    Y_MAX: float = 0.0       # um（顶部设为 0）

    # 两层黏膜厚度调节（放在最上面便于快速调参）
    # B 层（mucus surface）厚度 + C 层（mucus deep）厚度共同决定右侧边界位置。
    # 若开启自动同步，会自动更新：
    #   REG_B_END = REG_A_END + MUCUS_B_THICKNESS_UM
    #   X_MAX     = REG_B_END + MUCUS_C_THICKNESS_UM
    USE_MUCUS_LAYER_THICKNESS_CONTROL: bool = True
    MUCUS_B_THICKNESS_UM: float = 200.0
    MUCUS_C_THICKNESS_UM: float = 200.0

    # -----------------------------
    # y 方向边界条件
    # 版本B：主流沿 y，如果不处理边界，y 会被平流带走到很远，导致轨迹图看起来像“竖直截断”。
    # 这里提供周期边界（推荐）：把 y 包裹回观测窗口 [Y_MIN, Y_MAX]，等效于在显微镜视野内循环观测。
    # 若你更想模拟“离开视野就算离开”，可以把 USE_Y_PERIODIC=False，并自行定义离开条件。
    # -----------------------------
    USE_Y_PERIODIC: bool = True

    # 区域划分（A/B/C）
    REG_A_END: float = 200.0
    REG_B_END: float = 400.0

    # 屏障/连续运行要求（需要连续直跑足够久才可穿越）
    # 界面“连续直跑”门槛：先调低以观察穿透（后续可再回标）
    RUN_REQ_AB_S: float = 0.40
    RUN_REQ_BC_S: float = 0.35
    X_INT_EPS: float = 1e-6

    # ---- 界面层厚度（um）：有限厚度的屏障带 ----
    DELTA_INT_AB: float = 12.0   # 以 x=REG_A_END 为中心的半宽
    DELTA_INT_BC: float = 12.0   # 以 x=REG_B_END 为中心的半宽

    # ---- 概率穿越（剪切惩罚 + 蓝光增强）----
    USE_PROB_PASS: bool = True
    # 概率穿越：提高基线、减弱剪切惩罚、增强蓝光加成（先跑通穿透）
    PASS_P0_AB: float = 0.55
    PASS_P0_BC: float = 0.8
    PASS_ALPHA_GAMMA: float = 0.12
    PASS_BETA_BLUE: float = 0.32
    PASS_BLUE_I_HALF: float = 0.45
    PASS_BLUE_HILL_N: float = 2.0
    PASS_CLIP_MIN: float = 0.02
    PASS_CLIP_MAX: float = 0.98
    # BC 失败时的小回退距离（um）：避免“钳死在 400-eps”的界面卡死
    BC_FAIL_BACKSTEP_MIN_UM: float = 1.0
    BC_FAIL_BACKSTEP_MAX_UM: float = 5.0

    # -----------------------------
    # 黏膜物理化学场（B=表面黏膜，C=深层黏膜）+ 剪切变稀（Carreau–Yasuda）
    # 目的：把“网孔/致密度（空间位阻）+ 非牛顿流变（剪切变稀）”映射到 v、Dr、困陷率
    # -----------------------------
    USE_MUCUS_PHYS: bool = True

    # 黏膜致密度/网孔强度场 M(x)：B 层较低，C 层较高（无量纲）
    M_B: float = 0.6
    M_C: float = 1.0
    M_X0: float = 400.0      # um，B->C 过渡中心（建议与 REG_B_END 对齐）
    M_SCALE: float = 25.0    # um，过渡宽度（越小过渡越“硬”）

    # 困陷随致密度增强：p_trap *= (1 + TRAP_M_GAIN*M)
    TRAP_M_GAIN: float = 0.8

    # 速度随致密度衰减：v *= exp(-V_M_ALPHA*M)
    V_M_ALPHA: float = 0.12

    # 旋转扩散随致密度变化：Dr *= exp(-DR_M_ALPHA*M)
    # 你也可以把符号改成“+”表示更致密更容易被撞乱（Dr 增大），这要靠实验轨迹拟合决定
    DR_M_ALPHA: float = 0.15

    # 剪切变稀黏度模型（Carreau–Yasuda，广义牛顿流体）
    # mu(gamma) = MU_INF + (MU0_CY - MU_INF) * (1 + (LAM_CY*gamma)^A_CY)^((N_CY-1)/A_CY)
    MU0_CY: float = 3.0      # 零剪切黏度（相对量纲/系数）
    MU_INF: float = 0.6      # 无限剪切黏度（相对量纲/系数）
    LAM_CY: float = 0.8      # s，时间常数
    N_CY: float = 0.35       # 幂律指数（<1 表示剪切变稀）
    A_CY: float = 2.0        # Yasuda 参数

    # 黏度对运动的作用：v *= mu(gamma)^(-V_MU_POWER)，Dr *= mu(gamma)^(DR_MU_POWER)
    V_MU_POWER: float = 0.7
    DR_MU_POWER: float = 0.25

    # -----------------------------
    # 流场（平流）—— 在 A->B 平滑过渡
    # -----------------------------
    DELTA_U: float = 8.0
    U_FAST: float = 80.0
    U_SLOW: float = 30.0
    GAMMA_WALL: float = 0.0  # 可选线性剪切：uy += GAMMA_WALL*y（版本B：主流沿 y）

    # -----------------------------
    # 剪切扫描（用于复现 Yeo 的“单峰黏附窗口”并对比穿透率）
    # 说明：gamma_eff 是一个工程代理量；这里通过一个全局倍率 GAMMA_MULT
    #      来扫描“等效剪切强度”，从而画出 trap/穿透 vs 剪切曲线。
    # -----------------------------
    GAMMA_MULT: float = 1.0

    ENABLE_SHEAR_SWEEP: bool = False
    # 扫描用的剪切倍率列表（对 gamma_eff 线性缩放）
    SWEEP_GAMMA_MULT_LIST: Tuple[float, ...] = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    # 扫描时要对比的光照模式（只比较“蓝光 vs 无光”）
    SWEEP_LIGHT_MODES: Tuple[str, ...] = ("off", "blue")
    # 扫描时每个点重复次数（用于误差条）；1 表示不做重复
    SWEEP_REPEATS: int = 5
    # 扫描时是否保存每个点的单独 run 文件夹（默认不保存，避免输出爆炸）
    SWEEP_SAVE_EACH_RUN: bool = False
    # 扫描图输出文件名
    SWEEP_FIG_TRAP: str = "06_sweep_trap_vs_shear.png"
    SWEEP_FIG_PEN: str = "07_sweep_penetration_vs_shear.png"

    # -----------------------------
    # 运动性：按区域设置基线（参考 ABM.py）
    # -----------------------------
    V0_A: float = 20.0
    DR_A: float = 0.35  # rad^2/s（旋转扩散）
    V0_B: float = 14.0
    DR_B: float = 0.30
    V0_C: float = 9.0
    DR_C: float = 0.18

    # -----------------------------
    # 另一种运动模型：黏液剖面（参考 abm_light_mucus_v2）
    # 若 USE_MUCUS_PROFILE=True，则计算 mu(x) 并重标定 v、Dr、Dt。
    # -----------------------------
    USE_MUCUS_PROFILE: bool = False
    MU0: float = 1.0                 # 基线黏度因子
    MU1: float = 4.0                 # 高黏度因子（深层黏液）
    MU_X0: float = 320.0             # 过渡中心（um）
    MU_SCALE: float = 40.0           # 过渡陡峭度（um）
    MU_SPEED_POWER: float = 1.0      # v ~ mu^(-power)
    MU_DR_POWER: float = 1.0         # Dr ~ mu^(power)
    MU_DT_POWER: float = 1.0         # Dt ~ mu^(power)

    # 平移扩散（仅在启用时使用）
    USE_TRANSLATIONAL_DIFFUSION: bool = False
    DT_A: float = 0.2    # um^2/s
    DT_B: float = 0.12
    DT_C: float = 0.08

    # -----------------------------
    # 跑动-翻转（run-and-tumble）基础速率
    # -----------------------------
    LAMBDA_BASE: float = 0.45
    LAMBDA_FLOOR: float = 0.05
    LAMBDA_CEIL: float = 2.00
    # 无光惩罚：当 LIGHT_MODE == "off" 时，提高 lambda（更易翻滚）
    LAMBDA_DARK_BOOST: float = 0.35

    # 无光惩罚：当 LIGHT_MODE == "off" 时（无光）
    V0_DARK_FACTOR: float = 0.78      # v0 乘以该系数（游动更慢）
    TRAP_DARK_MULT: float = 2.1       # 困陷概率乘以该系数（更容易被困陷）

    # -----------------------------
    # 光照模式与映射
    # LIGHT_MODE 可选值："dual" | "blue" | "green" | "off"
    # LIGHT_MAPPING 可选映射：
    #   - "direct_dualcolor"：公式为 lambda = base - d_blue*Hill(blue) + d_green*Hill(green)
    #   - "chey"：使用内部状态 c(t)：c <- lowpass(c_inf(u))，再由 lambda <- Hill(c)
    # -----------------------------
    LIGHT_MODE: str = "off"
    LIGHT_MAPPING: str = "chey"

    # 光照空间分布（默认蓝光/绿光共用）
    GRAD_CENTER: float = 30.0
    GRAD_SCALE: float = 60.0

    # 开环时间策略（参考 ABM.py）
    USE_TIME_POLICY: bool = True
    BLUE_ON: float = 1.0
    GREEN_ON: float = 1.0
    GREEN_PULSE_PERIOD: float = 2.0
    GREEN_PULSE_WIDTH: float = 0.25

    # -----------------------------
    # 双通道直接映射超参数（参考 ABM.py）
    # -----------------------------
    DLAMBDA_BLUE: float = 0.35
    BLUE_HILL_N: float = 2.0
    BLUE_I_HALF: float = 0.45

    DLAMBDA_GREEN: float = 0.80
    GREEN_HILL_N: float = 2.0
    GREEN_I_HALF: float = 0.45

    # -----------------------------
    # “蓝光趋向 +x”的光趋化（phototaxis）模块（新增）
    # 目的：在主流沿 y 的情况下，仅靠“降低 tumble(λ)”很难产生净 +x 迁移。
    #      因此引入更接近经典 E. coli biased random walk 的做法：
    #      - tumble 后重采样方向时，对 +x 方向给更高权重（von Mises 分布）
    #      - run 过程中加入很弱的“朝 +x 拉回”的转向漂移（可选）
    #      - 蓝光下速度稍增、困陷稍减（可选）
    #
    # NOTE：这些参数都集中在这里，便于你后续快速调节看到“蓝光下穿透率更高”。
    # -----------------------------
    USE_PHOTOTAXIS_BIASED_TUMBLE: bool = True
    # 更真实的做法：尽量不要在 run 期间强行“转向漂移”（那会把方向性推到接近 1，导致不真实的 100% 穿透）
    # 默认关闭 steering，只保留“tumble 后的偏置重定向”，更接近经典 E. coli biased random walk
    USE_PHOTOTAXIS_STEERING: bool = False

    # -----------------------------
    # 光趋向的“记忆/适应”（更接近 E. coli 趋化的时间导数机制）
    # 思路：不是对光强 I 的绝对值做偏置，而是对 I(t) 的变化（I - memory）做偏置。
    # r_photo > 0 表示“刚刚走向更亮处”，此时延长 run / 或 tumble 后更愿意朝 +x。
    # -----------------------------
    USE_PHOTO_MEMORY: bool = True
    TAU_PHOTO_MEM_S: float = 1.2
    PHOTO_R_CLIP: float = 0.6
    # 现实里“只对时间导数敏感”会在强平流+弱横向迁移时导致偏置长期很弱。
    # 因此加入一个“绝对光强的弱权重”，让细胞在亮区更倾向于把新方向采样到 +x。
    # 取值建议 0~0.6，0 表示完全只用时间导数机制。
    PHOTO_ABS_WEIGHT: float = 0.13

    # 蓝光强度 -> 偏置强度（kappa）的 Hill 映射：phi = Hill01(Ib)
    PHOTO_BIAS_I_HALF: float = 0.35
    PHOTO_BIAS_HILL_N: float = 2.0

    # tumble 后 von Mises 集中度：kappa = base + gain*phi，越大越“朝 +x”
    PHOTO_BIAS_KAPPA_BASE: float = 0.0
    # 重要：把增益调小（原来 80/250 太强，容易直接“锁死朝 +x”从而不真实）
    PHOTO_BIAS_KAPPA_GAIN: float = 4.5
    PHOTO_BIAS_KAPPA_MAX: float = 40.0

    # run 过程的弱转向漂移：dtheta += TURN_GAIN * phi * sin(theta_target - theta) dt
    # 若你后续要打开 steering，可先用很小的值（否则会把方向性推到接近 1）
    PHOTO_TURN_GAIN: float = 0.6  # 1/s

    # 蓝光对速度/困陷的额外调节（经验项，可关掉）
    # 经验项建议先弱化：速度/困陷的改变通常不应比“tumble 行为偏置”更主导
    V_BLUE_GAIN: float = 0.09        # v *= (1 + V_BLUE_GAIN*phi)
    TRAP_BLUE_REDUCE: float = 0.055  # p_trap *= (1 - TRAP_BLUE_REDUCE*phi)

    # -----------------------------
    # CheY 风格映射超参数（参考 abm_light_mucus_v2）
    # -----------------------------
    # c 动力学：dc/dt = (c_inf(u) - c)/TAU
    USE_GATE_CONTROL: bool = False   # 若为 True，u(t) 由基于方向性的门控控制器给出
    TAU_C_S: float = 0.6
    C_INF_DARK: float = 0.75
    C_INF_LIT: float = 0.25
    # 序列特异实验响应幅度与归一化物理光强（均为无量纲 [0,1]）。
    # D_tilde 仅进入 c_inf 上游，不进入 photo_mem/kappa 支路。
    SEQUENCE_RESPONSE_D_TILDE: float = 1.0
    LIGHT_INTENSITY: float = 1.0

    # 通过 Hill 将 c 映射到 lambda（CW-bias 风格）
    C_HALF: float = 0.5
    C_HILL_N: float = 3.0
    LAMBDA_MIN: float = 0.08
    LAMBDA_MAX: float = 1.20

    # 门控控制器（闭环）：A 高则开灯，否则关灯
    GATE_A_ON: float = 0.25
    GATE_A_OFF: float = 0.10
    GATE_MIN_HOLD_S: float = 0.25

    # 若不使用门控，CheY 映射采用开环脉冲
    CHEY_PULSE_FREQ_HZ: float = 1.0
    CHEY_PULSE_DUTY: float = 0.50
    CHEY_PULSE_PHASE_S: float = 0.0
    CHEY_MODE: str = "continuous"  # 可选值："continuous" | "pulse" | "off"

    # -----------------------------
    # 剪切相关的困陷 + 朝向漂移（参考 ABM.py）
    # -----------------------------
    USE_TRAP: bool = True
    P_TRAP_OUT: float = 0.0002
    P_TRAP_INT: float = 0.0060
    P_TRAP_MUCUS: float = 0.0020
    # 界面层 trap：先降到较弱，避免 run_streak 被频繁打断导致卡界面
    P_TRAP_AB_LAYER: float = 0.0030
    P_TRAP_BC_LAYER: float = 0.0015
    TRAP_RELEASE_RATE: float = 1.0  # 1/s

    USE_SHEAR_TRAP: bool = True
    GAMMA_CRIT: float = 2.0
    SHEAR_TRAP_GAIN: float = 2.0

    # 先关闭剪切导致的朝向漂移：它会把细菌强行对齐到流向，掩盖光/趋化本身的效果
    USE_SHEAR_DRIFT: bool = False
    BETA_SHAPE: float = 0.7

    # -----------------------------
    # 趋化（参考 abm_light_mucus_v2）
    # -----------------------------
    # 这里保留适度的 +x 化学趋化基线，用来抬高无光组最终到达率；
    # 蓝光组则在此基础上叠加额外的 phototaxis 增益。
    USE_CHEMOTAXIS: bool = True
    # 营养场：C(x,y)=C0 + gx*x + gy*y，并在 C_FLOOR 处截断
    C0: float = 1.0
    GX: float = 0.032
    GY: float = 0.0
    C_FLOOR: float = 1e-3
    CHEM_SIGNAL: str = "linear"  # 可选值："log" | "linear"

    # 记忆/适应：m <- lowpass(S)，响应 r = S - m
    TAU_MEM_S: float = 1.25
    CHEM_GAIN: float = 1.9  # r 对 lambda 乘法调制的强度
    CHEM_CLIP: float = 1.5  # 对 exp() 中调制项做安全截断

    # 额外：趋化“转向”项（让效果更明显）
    # 思路：当细菌经历到 S 的上升（r>0）时，不仅延长 run（降低 lambda），还会在 run 期间把朝向轻微“拉向”梯度方向。
    # 对于本模型默认 GX>0, GY=0，因此梯度方向 theta_grad=0（沿 +x）。
    USE_CHEM_STEERING: bool = True
    CHEM_TURN_GAIN: float = 5.2  # 1/s，越大越强地把朝向拉向梯度方向

    # 额外：趋化“偏置翻滚后重定向”（更接近经典 E. coli 的 biased random walk）
    # 思路：不是只靠降低 lambda 延长 run，而是在 tumble 发生后重新采样方向时，
    #       让新朝向更偏向营养梯度方向（默认 GX>0 时就是 +x 方向）。
    # 实现：用 von Mises 分布重采样 theta，集中度 kappa 越大越“朝向梯度”。
    #       这里让 kappa 随 r=S-m 的正值增大（r>0 表示沿梯度变好）。
    USE_CHEM_BIASED_TUMBLE: bool = True
    CHEM_BIAS_KAPPA_BASE: float = 0.0   # 基础集中度（0 表示均匀随机）
    CHEM_BIAS_KAPPA_GAIN: float = 32.0   # r>0 时集中度增长系数（越大越强趋化）
    CHEM_BIAS_KAPPA_MAX: float = 200.0   # 集中度上限（防止数值过强）

    # -----------------------------
    # 指标目标位置
    # -----------------------------
    X_INTERFACE: float = 200.0
    X_TARGET: float = 500.0

    # 是否将 tumble 视为纯重定向（不平移）？（ABM.py 默认值为 True）
    TUMBLE_NO_TRANSLATION: bool = True

    # -----------------------------
    # 碰撞/反射开关（用于调试）
    # 关闭后：界面阻挡/边界越界不再“镜面反射”角度（theta 不翻转）。
    # 说明：你当前想先关闭全部碰撞反射，便于观察在流场 + 趋化/光趋向下的净迁移。
    # -----------------------------
    USE_REFLECT_INTERFACE: bool = False   # AB/BC 界面阻挡时是否反射
    USE_REFLECT_LEFT_WALL: bool = False  # x<X_MIN 时是否反射


def apply_mucus_layer_thickness(hp: HyperParams) -> None:
    """
    根据两层黏膜厚度自动同步几何边界与关键位置。
    这样只需改 MUCUS_B_THICKNESS_UM / MUCUS_C_THICKNESS_UM 即可整体生效。
    """
    if not hp.USE_MUCUS_LAYER_THICKNESS_CONTROL:
        return
    hp.REG_B_END = hp.REG_A_END + hp.MUCUS_B_THICKNESS_UM
    hp.X_MAX = hp.REG_B_END + hp.MUCUS_C_THICKNESS_UM
    hp.M_X0 = hp.REG_B_END
    hp.X_INTERFACE = hp.REG_A_END
    hp.X_TARGET = hp.REG_B_END + 0.5 * hp.MUCUS_C_THICKNESS_UM


HP = HyperParams()
apply_mucus_layer_thickness(HP)
# 安全检查
if HP.LIGHT_MODE not in {"dual", "blue", "green", "off"}:
    raise ValueError("LIGHT_MODE must be one of: dual, blue, green, off")
if HP.LIGHT_MAPPING not in {"direct_dualcolor", "chey"}:
    raise ValueError("LIGHT_MAPPING must be one of: direct_dualcolor, chey")


# =============================================================================
# 1) 基础数学工具
# =============================================================================

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def hill01(I: np.ndarray, k_half: float, n: float) -> np.ndarray:
    """Hill 饱和函数：输入 [0,1]，输出 [0,1]。"""
    I = np.clip(I, 0.0, 1.0)
    num = I ** n
    den = (k_half ** n + num)
    return num / den


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Return a same-length moving average, including for short test runs."""
    if w <= 1 or x.size <= 1:
        return x.copy()
    w = min(int(w), int(x.size))
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[w:] - c[:-w]) / w
    pad_left = w // 2
    pad_right = len(x) - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode="edge")


# =============================================================================
# 2) 几何与区域
# =============================================================================

def is_region_a(x: np.ndarray) -> np.ndarray:
    return x < HP.REG_A_END


def is_region_b(x: np.ndarray) -> np.ndarray:
    return (x >= HP.REG_A_END) & (x < HP.REG_B_END)


def is_region_c(x: np.ndarray) -> np.ndarray:
    return x >= HP.REG_B_END


def is_int_ab(x: np.ndarray) -> np.ndarray:
    return (x >= (HP.REG_A_END - HP.DELTA_INT_AB)) & (x <= (HP.REG_A_END + HP.DELTA_INT_AB))


def is_int_bc(x: np.ndarray) -> np.ndarray:
    return (x >= (HP.REG_B_END - HP.DELTA_INT_BC)) & (x <= (HP.REG_B_END + HP.DELTA_INT_BC))


# =============================================================================
# 3) 流场与剪切代理量（用于困陷/漂移）
# =============================================================================

def ux_base(x: np.ndarray) -> np.ndarray:
    """版本B：主流沿 y 方向，x 方向不施加平流（ux≈0）。"""
    return np.zeros_like(x, dtype=float)


def uy_base(x: np.ndarray) -> np.ndarray:
    """版本B：外部液体 A 区沿 y 方向快流，黏膜区 B/C 沿 y 方向慢流（平滑过渡）。"""
    s = sigmoid((HP.REG_A_END - x) / HP.DELTA_U)  # A区 s≈1，B/C区 s≈0
    # 约定：y 轴顶部为 0、向下为负值，因此“从上往下流”对应 uy<0
    return -(HP.U_SLOW + (HP.U_FAST - HP.U_SLOW) * s)


def ux_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """版本B：x 方向平流为 0（保留接口以兼容后续扩展）。"""
    return ux_base(x)


def uy_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """版本B：塞流 uy(x) + 可选 y 方向线性剪切（uy += GAMMA_WALL*y）。"""
    return uy_base(x) + HP.GAMMA_WALL * y


def shear_rate_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """
    简化的有效剪切强度（版本B）：
      - 来自 A->B 的过渡梯度 |duy/dx|
      - 可选壁面剪切 |duy/dy| = |GAMMA_WALL|

    说明：这里的“剪切强度”是工程化代理量，用来驱动 trap/概率穿越/漂移模块。
    """
    s = sigmoid((HP.REG_A_END - x) / HP.DELTA_U)
    ds_dz = s * (1.0 - s)
    duy_dx = -(HP.U_FAST - HP.U_SLOW) * ds_dz / HP.DELTA_U
    duy_dy = HP.GAMMA_WALL
    return HP.GAMMA_MULT * (np.abs(duy_dx) + np.abs(duy_dy))


def shear_peak_factor(gamma: np.ndarray, gamma_crit: float) -> np.ndarray:
    """单峰因子，在 gamma=gamma_crit 达峰：f=x*exp(1-x)。"""
    x = np.maximum(gamma, 0.0) / max(gamma_crit, 1e-12)
    return x * np.exp(1.0 - x)


def interface_gamma_reference(hp: HyperParams) -> float:
    """
    用固定空间采样定义界面参考剪切，避免因为某次仿真里细菌没有进入界面，
    导致 gamma_interface_mean 为 NaN，从而 sweep 图上只剩极少几个点。
    """
    n_x = 33
    n_y = 65

    x_ab = np.linspace(hp.REG_A_END - hp.DELTA_INT_AB, hp.REG_A_END + hp.DELTA_INT_AB, n_x)
    x_bc = np.linspace(hp.REG_B_END - hp.DELTA_INT_BC, hp.REG_B_END + hp.DELTA_INT_BC, n_x)
    y_line = np.linspace(hp.Y_MIN, hp.Y_MAX, n_y)

    Xab, Yab = np.meshgrid(x_ab, y_line)
    Xbc, Ybc = np.meshgrid(x_bc, y_line)

    # 使用与 shear_rate_field 相同的形式，但基于当前 hp 显式计算，避免 sweep 时出现全局参数错配。
    s_ab = sigmoid((hp.REG_A_END - Xab.ravel()) / hp.DELTA_U)
    s_bc = sigmoid((hp.REG_A_END - Xbc.ravel()) / hp.DELTA_U)
    duy_dx_ab = -(hp.U_FAST - hp.U_SLOW) * s_ab * (1.0 - s_ab) / hp.DELTA_U
    duy_dx_bc = -(hp.U_FAST - hp.U_SLOW) * s_bc * (1.0 - s_bc) / hp.DELTA_U
    duy_dy = np.abs(hp.GAMMA_WALL)
    gab = hp.GAMMA_MULT * (np.abs(duy_dx_ab) + duy_dy)
    gbc = hp.GAMMA_MULT * (np.abs(duy_dx_bc) + duy_dy)

    return float(np.mean(np.concatenate([gab, gbc])))


def interface_contact_reference_rate(frac_in_interface: np.ndarray, hp: HyperParams) -> float:
    """界面平均驻留比例，作为更稳定的界面接触参考量。"""
    _ = hp
    return float(np.mean(frac_in_interface))


# =============================================================================
# 4) 运动参数场：按区域或按黏液剖面重标定
# =============================================================================

def v0_by_region(x: np.ndarray) -> np.ndarray:
    v = np.full_like(x, HP.V0_C, dtype=float)
    v[is_region_a(x)] = HP.V0_A
    v[is_region_b(x)] = HP.V0_B
    v[is_region_c(x)] = HP.V0_C
    return v


def dr_by_region(x: np.ndarray) -> np.ndarray:
    dr = np.full_like(x, HP.DR_C, dtype=float)
    dr[is_region_a(x)] = HP.DR_A
    dr[is_region_b(x)] = HP.DR_B
    dr[is_region_c(x)] = HP.DR_C
    return dr


def dt_by_region(x: np.ndarray) -> np.ndarray:
    dt = np.full_like(x, HP.DT_C, dtype=float)
    dt[is_region_a(x)] = HP.DT_A
    dt[is_region_b(x)] = HP.DT_B
    dt[is_region_c(x)] = HP.DT_C
    return dt


def mucus_mu(x: np.ndarray) -> np.ndarray:
    """平滑黏度因子 mu(x)，范围 [MU0, MU1]（越高表示约束越强）。"""
    s = sigmoid((x - HP.MU_X0) / HP.MU_SCALE)
    return HP.MU0 + (HP.MU1 - HP.MU0) * s


def apply_mucus_scaling(v0: np.ndarray, dr0: np.ndarray, dt0: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = mucus_mu(x)
    v = v0 * np.power(mu, -HP.MU_SPEED_POWER)
    dr = dr0 * np.power(mu, HP.MU_DR_POWER)
    dt = dt0 * np.power(mu, HP.MU_DT_POWER)
    return v, dr, dt


def mucus_density_M(x: np.ndarray) -> np.ndarray:
    """黏膜致密度/网孔强度场 M(x)：B 层低、C 层高（无量纲）。"""
    s = sigmoid((x - HP.M_X0) / HP.M_SCALE)
    return HP.M_B + (HP.M_C - HP.M_B) * s


def mu_carreau_yasuda(gamma: np.ndarray) -> np.ndarray:
    """Carreau–Yasuda 剪切变稀黏度（返回相对黏度因子）。"""
    g = np.maximum(gamma, 0.0)
    term = 1.0 + (HP.LAM_CY * g) ** HP.A_CY
    return HP.MU_INF + (HP.MU0_CY - HP.MU_INF) * (term ** ((HP.N_CY - 1.0) / HP.A_CY))


def apply_mucus_phys(
    v0: np.ndarray,
    dr0: np.ndarray,
    p_trap_base: np.ndarray,
    x: np.ndarray,
    gamma_eff: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将黏膜物理化学性质映射到运动/困陷：
      - 致密度/网孔强度 M(x) -> 速度衰减、Dr 变化、困陷增强
      - 剪切变稀 mu(gamma) -> 速度相对提高（高剪切黏度下降）、Dr 轻微变化
    """
    M = mucus_density_M(x)

    # 1) 网孔/致密度：更致密 -> 更慢、更不易转向（此处选择 Dr 下降）、更易困陷
    v0 = v0 * np.exp(-HP.V_M_ALPHA * M)
    dr0 = dr0 * np.exp(-HP.DR_M_ALPHA * M)
    p_trap_base = p_trap_base * (1.0 + HP.TRAP_M_GAIN * M)

    # 2) 剪切变稀：mu 随 gamma 降低 -> 阻力降低 -> 速度相对提高
    mu_eff = mu_carreau_yasuda(gamma_eff)
    v0 = v0 * np.power(mu_eff, -HP.V_MU_POWER)
    dr0 = dr0 * np.power(mu_eff, HP.DR_MU_POWER)

    return v0, dr0, p_trap_base


# =============================================================================
# 5) 光场（时空）
# =============================================================================

def light_profile_x(x: np.ndarray) -> np.ndarray:
    """有界空间分布（sigmoid），范围 [0,1]。"""
    return sigmoid((x - HP.GRAD_CENTER) / HP.GRAD_SCALE)


def time_gate_blue(t: float) -> float:
    return HP.BLUE_ON if HP.USE_TIME_POLICY else 1.0


def time_gate_green(t: float) -> float:
    if not HP.USE_TIME_POLICY:
        return 0.0
    phase = t % HP.GREEN_PULSE_PERIOD
    return HP.GREEN_ON if phase < HP.GREEN_PULSE_WIDTH else 0.0


def light_blue(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """蓝光通道强度，范围 [0,1]。"""
    if HP.LIGHT_MODE in {"green", "off"}:
        return np.zeros_like(x, dtype=float)
    return np.clip(light_profile_x(x) * time_gate_blue(t), 0.0, 1.0)


def light_green(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    """绿光通道强度，范围 [0,1]。"""
    if HP.LIGHT_MODE in {"blue", "off"}:
        return np.zeros_like(x, dtype=float)
    return np.clip(light_profile_x(x) * time_gate_green(t), 0.0, 1.0)


# =============================================================================
# 6) CheY 风格内部动力学 + 门控控制（可选）
# =============================================================================

class GateController:
    """
    使用方向性 A（mean cos(theta-theta_d)）的闭环门控控制器。
    - 若 A >= A_ON  -> 开启（u=1）
    - 若 A <= A_OFF -> 关闭（u=0）
    包含最小保持时间以避免频繁抖动。
    """
    def __init__(self, a_on: float, a_off: float, min_hold_s: float):
        self.a_on = a_on
        self.a_off = a_off
        self.min_hold_s = min_hold_s
        self.u = 0.0
        self.hold_left = 0.0

    def step(self, A: float, dt: float) -> float:
        self.hold_left = max(0.0, self.hold_left - dt)
        if self.hold_left <= 0.0:
            if self.u <= 0.5 and A >= self.a_on:
                self.u = 1.0
                self.hold_left = self.min_hold_s
            elif self.u >= 0.5 and A <= self.a_off:
                self.u = 0.0
                self.hold_left = self.min_hold_s
        return self.u


def chey_u_of_t(t: float) -> float:
    """未启用门控时，CheY 映射的开环 u(t)。"""
    if HP.CHEY_MODE == "off":
        return 0.0
    if HP.CHEY_MODE == "continuous":
        return 1.0
    # 脉冲模式
    period = 1.0 / max(HP.CHEY_PULSE_FREQ_HZ, 1e-12)
    phase = (t + HP.CHEY_PULSE_PHASE_S) % period
    return 1.0 if phase < HP.CHEY_PULSE_DUTY * period else 0.0


def chey_c_inf(u_eff: np.ndarray | float) -> np.ndarray | float:
    """Map effective light input to the CheY-P steady state.

    ``u_eff`` is dimensionless in [0, 1]. The output uses the same normalized
    CheY-P units as ``C_INF_DARK`` and ``C_INF_LIT``.
    """
    u_array = np.asarray(u_eff, dtype=float)
    if np.any(~np.isfinite(u_array)) or np.any((u_array < 0.0) | (u_array > 1.0)):
        raise ValueError("u_eff must be finite and lie in [0, 1]")
    result = HP.C_INF_DARK + (HP.C_INF_LIT - HP.C_INF_DARK) * u_array
    return float(result) if result.ndim == 0 else result


def chey_update_c(c: np.ndarray, u_eff: np.ndarray | float, dt: float) -> np.ndarray:
    """
    一阶低通模型的精确指数更新：
      dc/dt = (c_inf(u_eff) - c)/tau
    """
    c_inf = chey_c_inf(u_eff)
    alpha = np.exp(-dt / max(HP.TAU_C_S, 1e-12))
    return c_inf + (c - c_inf) * alpha


def chey_c_to_lambda(c: np.ndarray) -> np.ndarray:
    """
    通过 Hill 形状的 CW-bias 曲线，将内部状态 c 映射到 tumble 速率 lambda。
    常见约定是 c 越高 -> lambda 越大（更容易 tumble，对应 CheY-P）。
    """
    cw = hill01(np.clip(c, 0.0, 1.0), HP.C_HALF, HP.C_HILL_N)  # 范围 [0,1]
    lam = HP.LAMBDA_MIN + (HP.LAMBDA_MAX - HP.LAMBDA_MIN) * cw
    return np.clip(lam, HP.LAMBDA_FLOOR, HP.LAMBDA_CEIL)


# =============================================================================
# 7) 趋化（可选）：营养信号 -> lambda 调制
# =============================================================================

def nutrient_C(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    C = HP.C0 + HP.GX * x + HP.GY * y
    return np.maximum(C, HP.C_FLOOR)


def chem_signal(C: np.ndarray) -> np.ndarray:
    if HP.CHEM_SIGNAL == "linear":
        return C
    # 常用对数信号来近似受体的对数感知
    return np.log(C)


def chem_update_memory(m: np.ndarray, S: np.ndarray, dt: float) -> np.ndarray:
    alpha = np.exp(-dt / max(HP.TAU_MEM_S, 1e-12))
    return S + (m - S) * alpha


def chem_modulate_lambda(lam: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    乘法调制：
      lam_eff = lam * exp(-gain * r)
    即：r>0（环境改善/沿梯度上升）时 -> lam 下降 -> run 变长。
    """
    r_clip = np.clip(r, -HP.CHEM_CLIP, HP.CHEM_CLIP)
    factor = np.exp(-HP.CHEM_GAIN * r_clip)
    return np.clip(lam * factor, HP.LAMBDA_FLOOR, HP.LAMBDA_CEIL)


# =============================================================================
# 8) 困陷
# =============================================================================

def trap_prob_base_by_region(x: np.ndarray) -> np.ndarray:
    # 区域基础值
    p = np.full_like(x, HP.P_TRAP_MUCUS, dtype=float)
    p[is_region_a(x)] = HP.P_TRAP_OUT
    p[is_region_b(x)] = HP.P_TRAP_INT
    p[is_region_c(x)] = HP.P_TRAP_MUCUS
    # 界面带覆写（最难穿越）
    p[is_int_ab(x)] = HP.P_TRAP_AB_LAYER
    p[is_int_bc(x)] = HP.P_TRAP_BC_LAYER
    return p


def pass_prob(gamma: np.ndarray, ib: np.ndarray, p0: float) -> np.ndarray:
    """
    p = p0 * exp(-alpha*gamma) * (1 + beta*Hill(ib))
    - gamma: 有效剪切（1/s）
    - ib: 局部蓝光强度，范围 [0,1]
    """
    shear_factor = np.exp(-HP.PASS_ALPHA_GAMMA * np.maximum(gamma, 0.0))
    blue_factor = 1.0 + HP.PASS_BETA_BLUE * hill01(
        np.clip(ib, 0.0, 1.0),
        HP.PASS_BLUE_I_HALF,
        HP.PASS_BLUE_HILL_N,
    )
    p = p0 * shear_factor * blue_factor
    return np.clip(p, HP.PASS_CLIP_MIN, HP.PASS_CLIP_MAX)


# =============================================================================
# 8b) 逐秒帧导出（用于后期 ffmpeg 拼视频）
# =============================================================================

def _frame_count(hp: HyperParams) -> int:
    return max(1, int(round(hp.T_S / hp.FRAME_STEP_S)))


def _get_plot_video_specs(hp: HyperParams) -> List[Tuple[str, str]]:
    """9 个视频，与 9 类动态分析图一一对应（05_u_of_t 为静态控制量，仅保存终态 PNG）。"""
    return [
        ("01_sample_trajectories", "01_sample_trajectories_30fps.mp4"),
        ("02_directionality", "02_directionality_30fps.mp4"),
        ("03_mean_x_position", "03_mean_x_position_30fps.mp4"),
        ("03b_msd", "03b_msd_30fps.mp4"),
        ("04_target_arrival_metrics", "04_target_arrival_metrics_30fps.mp4"),
        ("06_xt_density_heatmap", "06_xt_density_heatmap_30fps.mp4"),
        ("07_xy_occupancy_heatmap", "07_xy_occupancy_heatmap_30fps.mp4"),
        ("08_state_fractions", "08_state_fractions_30fps.mp4"),
        ("09_mechanism_diagnostics", "09_mechanism_diagnostics_30fps.mp4"),
    ]


def _save_plot_frame(fig: plt.Figure, out_path: Path, hp: HyperParams) -> None:
    fig.savefig(out_path, dpi=hp.FRAME_PNG_DPI)
    plt.close(fig)


def _save_all_plot_frames(
    plot_dirs: Dict[str, Path],
    frame_idx: int,
    t_s: float,
    step_k: int,
    hp: HyperParams,
    *,
    traj_x: np.ndarray,
    traj_y: np.ndarray,
    dir_t: np.ndarray,
    mean_x: np.ndarray,
    msd: np.ndarray,
    u_t: np.ndarray,
    frac_target: np.ndarray,
    frac_interface: np.ndarray,
    cross_flux: np.ndarray,
    cross_flux_int: np.ndarray,
    frac_run: np.ndarray,
    frac_tumble: np.ndarray,
    frac_trapped: np.ndarray,
    mean_lambda_t: np.ndarray,
    mean_c_t: np.ndarray,
    xt_density: np.ndarray,
    occupancy_xy: np.ndarray,
    x_edges_xt: np.ndarray,
    penetration_rate: float,
    window_steps: int,
) -> None:
    """按当前仿真进度保存各分析图的逐秒帧（与终态 PNG 风格一致）。"""
    tag = f"frame_{frame_idx:04d}"
    fig_tag = f"mode={hp.LIGHT_MODE}, mapping={hp.LIGHT_MAPPING}"
    n = step_k + 1
    time_axis = (np.arange(n) + 1) * hp.DT_S
    t_max = float(hp.T_S)

    def _smooth(arr: np.ndarray) -> np.ndarray:
        return moving_average(arr[:n], window_steps)

    # 01 轨迹
    fig1, ax1 = plt.subplots(figsize=(16, 9))
    ax1.set_facecolor("#f8fafc")
    for i in range(traj_x.shape[0]):
        xx = traj_x[i, : n + 1].copy()
        yy = traj_y[i, : n + 1].copy()
        if hp.USE_Y_PERIODIC:
            span = hp.Y_MAX - hp.Y_MIN
            wrap_jump = np.abs(np.diff(yy)) > 0.5 * span
            if np.any(wrap_jump):
                yy[np.where(wrap_jump)[0] + 1] = np.nan
        ax1.plot(xx, yy, linewidth=0.75, alpha=0.8)
    ax1.axvspan(hp.X_MIN, hp.REG_A_END, alpha=0.10, color="#dbeafe")
    ax1.axvspan(hp.REG_A_END, hp.REG_B_END, alpha=0.12, color="#fde68a")
    ax1.axvspan(hp.REG_B_END, hp.X_MAX, alpha=0.10, color="#e9d5ff")
    ax1.axvline(hp.REG_A_END, linestyle="--", linewidth=1.2, color="#1d4ed8")
    ax1.axvline(hp.REG_B_END, linestyle="--", linewidth=1.2, color="#1d4ed8")
    ax1.text(
        0.02, 0.98,
        f"Penetration @ {hp.X_MAX:.0f} um: {penetration_rate * 100.0:.1f}%",
        transform=ax1.transAxes, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    ax1.set_title(f"Sample Trajectories ({fig_tag})  t={t_s:.1f}s")
    ax1.set_xlabel("x (um)")
    ax1.set_ylabel("y (um)")
    ax1.set_xlim(hp.X_MIN, hp.X_MAX)
    ax1.set_ylim(hp.Y_MIN, hp.Y_MAX)
    ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    fig1.tight_layout()
    _save_plot_frame(fig1, plot_dirs["01_sample_trajectories"] / f"{tag}.png", hp)

    # 02 方向性
    fig2 = plt.figure()
    plt.plot(time_axis, dir_t[:n], label="Raw")
    plt.plot(time_axis, _smooth(dir_t), label=f"Smoothed ({hp.WINDOW_S:.1f}s)")
    plt.title(f"Directionality ({fig_tag})  t={t_s:.1f}s")
    plt.xlabel("t (s)")
    plt.ylabel("mean cos(theta - theta_d)")
    plt.xlim(0, t_max)
    plt.legend()
    plt.tight_layout()
    _save_plot_frame(fig2, plot_dirs["02_directionality"] / f"{tag}.png", hp)

    # 03 mean x
    fig3 = plt.figure()
    plt.plot(time_axis, mean_x[:n], label="Raw")
    plt.plot(time_axis, _smooth(mean_x), label=f"Smoothed ({hp.WINDOW_S:.1f}s)")
    plt.title(f"Mean x Position ({fig_tag})  t={t_s:.1f}s")
    plt.xlabel("t (s)")
    plt.ylabel("mean x (um)")
    plt.xlim(0, t_max)
    plt.legend()
    plt.tight_layout()
    _save_plot_frame(fig3, plot_dirs["03_mean_x_position"] / f"{tag}.png", hp)

    # 03b MSD
    fig3b = plt.figure()
    plt.plot(time_axis, msd[:n], label="Raw")
    plt.plot(time_axis, _smooth(msd), label=f"Smoothed ({hp.WINDOW_S:.1f}s)")
    plt.title(f"Mean Squared Displacement ({fig_tag})  t={t_s:.1f}s")
    plt.xlabel("t (s)")
    plt.ylabel("MSD (um^2)")
    plt.xlim(0, t_max)
    plt.legend()
    plt.tight_layout()
    _save_plot_frame(fig3b, plot_dirs["03b_msd"] / f"{tag}.png", hp)

    # 04 到达指标
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    ax4a.plot(time_axis, frac_target[:n], color="#2563eb", alpha=0.22, linewidth=1.0, label="Target Fraction (Raw)")
    ax4a.plot(time_axis, _smooth(frac_target), color="#2563eb", linewidth=2.0, label=f"Target Fraction (Smoothed {hp.WINDOW_S:.1f}s)")
    ax4a.plot(time_axis, frac_interface[:n], color="#16a34a", alpha=0.22, linewidth=1.0, label="Interface Fraction (Raw)")
    ax4a.plot(time_axis, _smooth(frac_interface), color="#16a34a", linewidth=2.0, label=f"Interface Fraction (Smoothed {hp.WINDOW_S:.1f}s)")
    ax4a.set_ylabel("fraction")
    ax4a.set_xlim(0, t_max)
    ax4a.legend(loc="upper left", ncol=2, fontsize=9)
    ax4a.set_title(f"Target Arrival Metrics ({fig_tag})  t={t_s:.1f}s")
    ax4a.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    ax4b.plot(time_axis, cross_flux[:n], color="#f97316", alpha=0.22, linewidth=1.0, label="Crossing Flux @ X_TARGET (Raw)")
    ax4b.plot(time_axis, _smooth(cross_flux), color="#f97316", linewidth=2.0, label=f"Crossing Flux @ X_TARGET (Smoothed {hp.WINDOW_S:.1f}s)")
    ax4b.plot(time_axis, cross_flux_int[:n], color="#dc2626", alpha=0.22, linewidth=1.0, label="Crossing Flux @ X_INTERFACE (Raw)")
    ax4b.plot(time_axis, _smooth(cross_flux_int), color="#dc2626", linewidth=2.0, label=f"Crossing Flux @ X_INTERFACE (Smoothed {hp.WINDOW_S:.1f}s)")
    ax4b.set_ylabel("forward crossings/s")
    ax4b.set_xlabel("t (s)")
    ax4b.legend(loc="upper left", ncol=2, fontsize=9)
    ax4b.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    fig4.tight_layout()
    _save_plot_frame(fig4, plot_dirs["04_target_arrival_metrics"] / f"{tag}.png", hp)

    # 06 x-t 密度热图
    fig6, ax6 = plt.subplots(figsize=(10.8, 5.2))
    xt_slice = xt_density[:n].T
    vmax = float(max(xt_density[:n].max(), 1.0))
    im6 = ax6.imshow(
        xt_slice,
        origin="lower",
        aspect="auto",
        extent=[time_axis[0], time_axis[-1], hp.X_MIN, hp.X_MAX],
        vmin=0.0,
        vmax=vmax,
    )
    ax6.axhline(hp.REG_A_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
    ax6.axhline(hp.REG_B_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
    ax6.set_title(f"x-t Density Heatmap ({fig_tag})  t={t_s:.1f}s")
    ax6.set_xlabel("t (s)")
    ax6.set_ylabel("x (um)")
    ax6.set_xlim(0, t_max)
    fig6.colorbar(im6, ax=ax6).set_label("cell count per x-bin")
    fig6.tight_layout()
    _save_plot_frame(fig6, plot_dirs["06_xt_density_heatmap"] / f"{tag}.png", hp)

    # 07 x-y 停留热图（累计）
    fig7, ax7 = plt.subplots(figsize=(12.0, 5.5))
    vmax_xy = float(max(occupancy_xy.max(), 1.0))
    im7 = ax7.imshow(
        occupancy_xy,
        origin="lower",
        aspect="auto",
        extent=[hp.X_MIN, hp.X_MAX, hp.Y_MIN, hp.Y_MAX],
        vmin=0.0,
        vmax=vmax_xy,
    )
    ax7.axvline(hp.REG_A_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
    ax7.axvline(hp.REG_B_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
    ax7.set_title(f"x-y Occupancy Heatmap ({fig_tag})  t={t_s:.1f}s")
    ax7.set_xlabel("x (um)")
    ax7.set_ylabel("y (um)")
    fig7.colorbar(im7, ax=ax7).set_label("accumulated occupancy")
    fig7.tight_layout()
    _save_plot_frame(fig7, plot_dirs["07_xy_occupancy_heatmap"] / f"{tag}.png", hp)

    # 08 状态占比
    fig8, ax8 = plt.subplots(figsize=(11.2, 5.0))
    ax8.stackplot(time_axis, frac_run[:n], frac_tumble[:n], frac_trapped[:n], labels=["run", "tumble", "trapped"], alpha=0.85)
    ax8.set_ylim(0.0, 1.0)
    ax8.set_xlim(0, t_max)
    ax8.set_title(f"State Fractions Over Time ({fig_tag})  t={t_s:.1f}s")
    ax8.set_xlabel("t (s)")
    ax8.set_ylabel("fraction of alive cells")
    ax8.legend(loc="upper right")
    ax8.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    fig8.tight_layout()
    _save_plot_frame(fig8, plot_dirs["08_state_fractions"] / f"{tag}.png", hp)

    # 09 机制诊断
    fig9, axes9 = plt.subplots(3, 1, figsize=(11.2, 8.2), sharex=True)
    axes9[0].plot(time_axis, dir_t[:n], alpha=0.25, linewidth=1.0, label="A raw")
    axes9[0].plot(time_axis, _smooth(dir_t), linewidth=2.0, label="A smoothed")
    axes9[0].set_ylabel("A")
    axes9[0].set_xlim(0, t_max)
    axes9[0].legend(loc="upper right")
    axes9[0].grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    axes9[1].plot(time_axis, mean_lambda_t[:n], linewidth=1.8, label="mean lambda")
    axes9[1].plot(time_axis, mean_c_t[:n], linewidth=1.5, label="mean c")
    if hp.LIGHT_MAPPING == "chey":
        axes9[1].plot(time_axis, u_t[:n], linewidth=1.2, label="u(t)")
    axes9[1].set_ylabel("control / internal state")
    axes9[1].legend(loc="upper right")
    axes9[1].grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    axes9[2].plot(time_axis, _smooth(cross_flux_int), linewidth=1.8, label="cross flux @ interface")
    axes9[2].plot(time_axis, _smooth(frac_target), linewidth=1.8, label="target fraction")
    axes9[2].set_xlabel("t (s)")
    axes9[2].set_ylabel("transport metrics")
    axes9[2].legend(loc="upper right")
    axes9[2].grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
    fig9.suptitle(f"Mechanism Diagnostics ({fig_tag})  t={t_s:.1f}s")
    fig9.tight_layout()
    _save_plot_frame(fig9, plot_dirs["09_mechanism_diagnostics"] / f"{tag}.png", hp)


def _pad_frame_even(frame: np.ndarray) -> np.ndarray:
    """H.264/yuv420p 要求宽高为偶数；matplotlib PNG 常为奇数高。"""
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _compose_frames_to_video(
    frames_dir: Path,
    out_path: Path,
    hp: HyperParams,
    *,
    desc: str | None = None,
) -> Path | None:
    """将某类图的 frame_XXXX.png 合成为 MP4（默认 30 fps）。"""
    if not hp.MAKE_VIDEO:
        return None

    png_paths = sorted(frames_dir.glob("frame_*.png"))
    if not png_paths:
        print(f"[VIDEO] 未找到帧图，跳过: {frames_dir.name}")
        return None

    if out_path.exists():
        out_path.unlink()
    n = len(png_paths)
    fps = float(hp.VIDEO_FPS)
    label = desc or out_path.stem
    print(f"[VIDEO] {label}: 合成 {n} 帧 -> {out_path.name}（{fps:g} fps）")
    last_err: Exception | None = None

    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=1,
        )
        for p in png_paths:
            writer.append_data(_pad_frame_even(imageio.imread(p)))
        writer.close()
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[VIDEO] 已保存: {out_path}（时长约 {n / fps:.1f}s）")
            return out_path
        raise RuntimeError("imageio 未写出有效视频文件")
    except Exception as exc_imageio:
        last_err = exc_imageio
        if out_path.exists():
            out_path.unlink()

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin and png_paths[0].name.startswith("frame_"):
        cmd = [
            ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-frames:v", str(n),
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[VIDEO] 已保存: {out_path}（时长约 {n / fps:.1f}s）")
                return out_path
            raise RuntimeError("ffmpeg 未写出有效视频文件")
        except subprocess.CalledProcessError as exc_ff:
            last_err = exc_ff
            if out_path.exists():
                out_path.unlink()

    print(f"[VIDEO] {label} 合成失败: {last_err}")
    return None


def _compose_all_plot_videos(
    plot_dirs: Dict[str, Path],
    hp: HyperParams,
    run_dir: Path,
) -> List[Dict[str, str]]:
    """将 9 类分析图帧分别合成为 9 个 MP4。"""
    if not hp.MAKE_VIDEO:
        return []

    videos_dir = run_dir / hp.VIDEOS_SUBDIR
    videos_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, str]] = []

    print(f"[VIDEO] 开始合成 {len(_get_plot_video_specs(hp))} 个视频 …")
    for plot_id, video_name in _get_plot_video_specs(hp):
        out_path = videos_dir / video_name
        composed = _compose_frames_to_video(
            plot_dirs[plot_id],
            out_path,
            hp,
            desc=plot_id,
        )
        if composed is not None:
            results.append({"plot_id": plot_id, "video": str(composed)})

    print(f"[VIDEO] 完成，共 {len(results)} 个视频 -> {videos_dir}")
    return results


# =============================================================================
# 9) 仿真主流程
# =============================================================================

def simulate(
    hp: HyperParams,
    *,
    make_plots: bool = True,
    save_outputs: bool = True,
    return_diagnostics: bool = False,
) -> Dict[str, float]:
    global HP
    # 关键修正：本文件里大量场函数（区域、流场、剪切、trap、光场等）直接读取全局 HP。
    # 如果 sweep 时只修改局部 hp 而不同步到全局 HP，
    # 那么 x 轴的 gamma_ref 会变，但实际动力学仍然在用旧的全局参数，
    # 最终就会出现“不同剪切点对应同一条水平直线”的假象。
    # 因此每次 simulate 开始时，都把当前 hp 同步成全局 HP 的工作副本。
    HP = HyperParams(**asdict(hp))
    hp = HP

    # 若用户在运行前修改了层厚参数，这里再次同步一次几何边界。
    apply_mucus_layer_thickness(hp)
    if not 0.0 <= hp.SEQUENCE_RESPONSE_D_TILDE <= 1.0:
        raise ValueError("SEQUENCE_RESPONSE_D_TILDE must lie in [0, 1]")
    if not 0.0 <= hp.LIGHT_INTENSITY <= 1.0:
        raise ValueError("LIGHT_INTENSITY must lie in [0, 1]")
    rng = np.random.default_rng(hp.SEED)
    steps = int(hp.T_S / hp.DT_S)
    n_frames = _frame_count(hp)
    frame_steps = max(1, int(round(hp.FRAME_STEP_S / hp.DT_S)))

    # 输出目录
    run_dir: Path | None = None
    plot_dirs: Dict[str, Path] = {}
    frame_manifest: List[Dict[str, str]] = []
    window_steps = max(1, int(hp.WINDOW_S / hp.DT_S))
    if save_outputs:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(hp.OUTPUT_DIR) / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if hp.SAVE_FRAMES:
            plots_root = run_dir / hp.PLOT_FRAMES_SUBDIR
            for plot_id, _ in _get_plot_video_specs(hp):
                plot_dirs[plot_id] = plots_root / plot_id
                plot_dirs[plot_id].mkdir(parents=True, exist_ok=True)

        # 保存超参数快照
        (run_dir / "hyperparams.json").write_text(json.dumps(asdict(hp), indent=2), encoding="utf-8")

    # 初始化位置：从 A 区开始，模拟“入口注入”
    x = rng.uniform(20.0, hp.REG_A_END - 20.0, size=hp.N)
    y = rng.uniform(hp.Y_MIN + 20.0, hp.Y_MAX - 20.0, size=hp.N)
    theta = rng.uniform(-np.pi, np.pi, size=hp.N)

    # y_unwrap 用于记录真实的“沿主流方向”的累计位移（不做周期包裹），
    # 这样 MSD 等统计不会因为周期边界而被人为压小。
    y_unwrap = y.copy()

    x0 = x.copy()
    y0 = y_unwrap.copy()

    # 连续 run 时长：自上次 tumble 以来的连续运行时间（用于界面穿越）
    run_streak = np.zeros(hp.N, dtype=float)

    # 困陷状态
    trapped = np.zeros(hp.N, dtype=bool)
    trap_time_left = np.zeros(hp.N, dtype=float)

    # 存活标记：True 表示仍在芯片模拟域内；触碰右边界 x>=X_MAX 认为已穿透，置为 False（删除）
    alive = np.ones(hp.N, dtype=bool)
    penetrated_count = 0
    # 累计到达标记（避免因删除/离开域导致“到达比例”回落）
    reached_interface = np.zeros(hp.N, dtype=bool)  # 曾经到达/越过 X_INTERFACE
    reached_target = np.zeros(hp.N, dtype=bool)     # 曾经到达/越过 X_TARGET
    reached_exit = np.zeros(hp.N, dtype=bool)       # 曾经到达/越过 X_MAX（穿透）
    release_prob = 1.0 - np.exp(-hp.TRAP_RELEASE_RATE * hp.DT_S) if hp.USE_TRAP else 0.0

    # 趋化记忆状态（逐个体）
    m = np.zeros(hp.N, dtype=float)

    # 光趋向“记忆/适应”状态（逐个体）：用于构造 r_photo = I - I_memory
    photo_mem = np.zeros(hp.N, dtype=float)

    # CheY 内部状态（逐个体）
    c = np.full(hp.N, 0.5 * (hp.C_INF_DARK + hp.C_INF_LIT), dtype=float)
    gate = GateController(hp.GATE_A_ON, hp.GATE_A_OFF, hp.GATE_MIN_HOLD_S)

    # 绘图采样
    plot_idx = rng.choice(hp.N, size=min(hp.N_PLOT, hp.N), replace=False)
    traj_x = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_y = np.empty((plot_idx.size, steps + 1), dtype=np.float32)          # 用于绘图（可能会被周期包裹）
    traj_y_unwrap = np.empty((plot_idx.size, steps + 1), dtype=np.float32)   # 用于检测包裹跳变
    traj_x[:, 0] = x[plot_idx]
    traj_y[:, 0] = y[plot_idx]
    traj_y_unwrap[:, 0] = y_unwrap[plot_idx]

    # 时间序列指标
    dir_t = np.empty(steps, dtype=np.float32)
    mean_x = np.empty(steps, dtype=np.float32)
    frac_interface = np.empty(steps, dtype=np.float32)
    cross_flux_int = np.empty(steps, dtype=np.float32)
    frac_target = np.empty(steps, dtype=np.float32)
    cross_flux = np.empty(steps, dtype=np.float32)
    msd = np.empty(steps, dtype=np.float32)
    u_t = np.empty(steps, dtype=np.float32)  # （用于 CheY 映射）记录 u(t)
    frac_run = np.empty(steps, dtype=np.float32)
    frac_tumble = np.empty(steps, dtype=np.float32)
    frac_trapped = np.empty(steps, dtype=np.float32)
    mean_lambda_t = np.empty(steps, dtype=np.float32)
    mean_c_t = np.empty(steps, dtype=np.float32)
    mean_c_inf_t = np.full(steps, np.nan, dtype=np.float32)

    # ---- 额外诊断：用于画 “trap/界面停留 vs 剪切” 与 “穿透率 vs 剪切” ----
    trap_events = np.zeros(steps, dtype=np.int32)          # 本步 trap 进入事件数（全域）
    trap_events_int = np.zeros(steps, dtype=np.int32)      # 本步 trap 进入事件数（仅界面带）
    frac_in_interface = np.zeros(steps, dtype=np.float32)  # 本步在界面带内的活跃比例
    gamma_int_mean = np.full(steps, np.nan, dtype=np.float32)  # 本步界面带内的平均剪切

    # ---- 额外可视化缓存 ----
    x_edges_xt = np.linspace(hp.X_MIN, hp.X_MAX, hp.XT_BINS_X + 1)
    xt_density = np.zeros((steps, hp.XT_BINS_X), dtype=np.float32)

    x_edges_xy = np.linspace(hp.X_MIN, hp.X_MAX, hp.X_BINS_HEATMAP + 1)
    y_edges_xy = np.linspace(hp.Y_MIN, hp.Y_MAX, hp.Y_BINS_HEATMAP + 1)
    occupancy_xy = np.zeros((hp.Y_BINS_HEATMAP, hp.X_BINS_HEATMAP), dtype=np.float32)

    t_first_interface = np.full(hp.N, np.nan, dtype=np.float32)
    t_first_target = np.full(hp.N, np.nan, dtype=np.float32)
    t_first_exit = np.full(hp.N, np.nan, dtype=np.float32)

    x_prev = x.copy()

    # 目标方向（+x）
    d_hat = np.array([1.0, 0.0], dtype=float)

    use_frame_export = bool(save_outputs and hp.SAVE_FRAMES and plot_dirs)
    progress = tqdm(
        total=n_frames if use_frame_export else steps,
        desc="仿真进度（按秒存帧）" if use_frame_export else "ABM 仿真",
        unit="帧" if use_frame_export else "步",
        disable=False,
    )

    for k in range(steps):
        t = (k + 1) * hp.DT_S

        x_prev[:] = x

        # ---------------------------------------------------------------------
        # 9.1) 计算方向性 A = mean cos(theta-theta_d)
        # 依据：ABM 中常用的序参量/对齐指标。
        # ---------------------------------------------------------------------
        vhatx = np.cos(theta)
        vhaty = np.sin(theta)
        dot = vhatx * d_hat[0] + vhaty * d_hat[1]
        dot[trapped] = 0.0  # 被困陷个体不贡献定向运动
        dot[~alive] = 0.0   # 已穿透删除的个体不参与统计
        A = float(dot.mean())
        dir_t[k] = A

        # ---------------------------------------------------------------------
        # 9.2) 决定光输入并计算 tumble 速率 lambda
        # ---------------------------------------------------------------------
        Ib_local = np.zeros(hp.N, dtype=float)  # 统一的“蓝光局部强度”占位（用于光趋化/困陷/穿越概率）
        if hp.LIGHT_MAPPING == "direct_dualcolor":
            # 双通道光场
            Ib = light_blue(x, y, t)
            Ig = light_green(x, y, t)
            # 统一的“蓝光局部强度”字段：后续用于光趋化偏置、困陷降低、界面穿越概率增强
            Ib_local = Ib.copy()
            Ib_local[~alive] = 0.0
            # 各通道的 Hill 激活
            fb = hill01(Ib, hp.BLUE_I_HALF, hp.BLUE_HILL_N)
            fg = hill01(Ig, hp.GREEN_I_HALF, hp.GREEN_HILL_N)
            # 蓝光 -> 更直行（lambda 下降），绿光 -> 更易 tumble（lambda 上升）
            lam = hp.LAMBDA_BASE - hp.DLAMBDA_BLUE * fb + hp.DLAMBDA_GREEN * fg
            lam = np.clip(lam, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            # 无光惩罚：无光时认为翻滚更频繁（lambda 增大）
            if hp.LIGHT_MODE == "off":
                lam = np.clip(lam + hp.LAMBDA_DARK_BOOST, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            u_t[k] = np.nan  # 不适用
        else:
            # CheY 映射：仅替换上游入口为
            # u_eff_i = D_tilde_i * I_light_i(t) * light_gate(t)。
            # photo_mem/kappa 继续使用未乘 D_tilde 的物理蓝光信号。
            light_gate = gate.step(A, hp.DT_S) if hp.USE_GATE_CONTROL else chey_u_of_t(t)
            if hp.LIGHT_MODE in {"blue", "dual"}:
                I_light_local = np.clip(hp.LIGHT_INTENSITY * light_profile_x(x), 0.0, 1.0)
            else:
                I_light_local = np.zeros(hp.N, dtype=float)
            u_eff = effective_light_input(
                hp.SEQUENCE_RESPONSE_D_TILDE,
                I_light_local,
                light_gate,
            )
            u_eff[~alive] = 0.0
            u_t[k] = float(np.mean(u_eff[alive])) if np.any(alive) else 0.0
            c_inf_now = chey_c_inf(u_eff)
            mean_c_inf_t[k] = float(np.mean(c_inf_now[alive])) if np.any(alive) else float(np.mean(c_inf_now))
            c[:] = chey_update_c(c, u_eff, hp.DT_S)
            lam = chey_c_to_lambda(c)
            Ib_local = np.clip(I_light_local * light_gate, 0.0, 1.0)
            Ib_local[~alive] = 0.0
        mean_lambda_t[k] = float(np.mean(lam[alive])) if np.any(alive) else float(np.mean(lam))
        mean_c_t[k] = float(np.mean(c[alive])) if np.any(alive) else float(np.mean(c))

        # ---------------------------------------------------------------------
        # 9.2b) 光趋向的“时间导数”信号（更接近真实的 E. coli 机制）
        # r_photo > 0 表示最近走向更亮处；只在这种情况下才增强“朝 +x 的偏置”。
        # ---------------------------------------------------------------------
        if hp.LIGHT_MODE in {"blue", "dual"}:
            # 这里用蓝光局部强度作为信号（0~1）
            I_sig = np.clip(Ib_local, 0.0, 1.0)
        else:
            I_sig = np.zeros(hp.N, dtype=float)

        # 额外计算绝对光强的“弱偏置权重”，用于避免：在强 y 平流下，
        # 细胞很难持续产生正的时间导数信号，导致趋向偏置几乎消失，从而 x 前进不足。
        phi_I = hill01(np.clip(I_sig, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)

        if hp.USE_PHOTO_MEMORY:
            alpha_p = np.exp(-hp.DT_S / max(hp.TAU_PHOTO_MEM_S, 1e-12))
            photo_mem[:] = I_sig + (photo_mem - I_sig) * alpha_p
            r_photo = I_sig - photo_mem
            r_photo = np.clip(r_photo, -hp.PHOTO_R_CLIP, hp.PHOTO_R_CLIP)
        else:
            # 不用记忆时，r_photo 直接等于绝对光强（0~1）
            r_photo = I_sig

        # 最终用于“tumble 后方向采样偏置”的强度：
        # - 主导项：正的时间导数（走向更亮处）
        # - 辅助项：绝对光强的弱权重（让亮区更倾向采样到 +x，避免偏置为 0）
        phi_d = np.maximum(r_photo, 0.0) / max(hp.PHOTO_R_CLIP, 1e-12)
        phi_d = np.clip(phi_d, 0.0, 1.0)
        phi_photo = np.clip(phi_d + hp.PHOTO_ABS_WEIGHT * phi_I, 0.0, 1.0)

        # ---------------------------------------------------------------------
        # 9.3) 可选趋化：营养信号 -> 记忆 -> 调制 lambda
        # 依据：可用低通记忆近似 E. coli 式“适应”，
        # 当环境改善时 run 变长（lambda 下降）。
        # ---------------------------------------------------------------------
        if hp.USE_CHEMOTAXIS:
            C = nutrient_C(x, y)
            S = chem_signal(C)
            m[:] = chem_update_memory(m, S, hp.DT_S)
            r = S - m
            lam = chem_modulate_lambda(lam, r)
            r_for_steer = r
        else:
            r_for_steer = None

        # ---------------------------------------------------------------------
        # 9.4) 流场与剪切场
        # 依据：平流搬运个体；剪切会影响黏附/困陷
        # 以及朝向漂移（最简 Jeffery 风格项）。
        # ---------------------------------------------------------------------
        ux = ux_field(x, y, t)
        uy = uy_field(x, y, t)
        gamma_eff = shear_rate_field(x, y, t)

        # ---------------------------------------------------------------------
        # 9.5) 困陷动力学
        # 依据：在黏液/界面中发生间歇性黏附或失活；
        # 释放为指数过程（无记忆），速率为 TRAP_RELEASE_RATE。
        # ---------------------------------------------------------------------
        if hp.USE_TRAP:
            # 仅对仍在域内的个体进行困陷/释放更新
            releasing = trapped & alive & (rng.random(hp.N) < release_prob)
            trapped[releasing] = False
            trap_time_left[releasing] = 0.0

            p_trap_base = trap_prob_base_by_region(x)

            # 无光惩罚：无光更易被困陷
            if hp.LIGHT_MODE == "off":
                p_trap_base = p_trap_base * hp.TRAP_DARK_MULT

            # 蓝光下：假设细菌更不容易被困陷/黏附（经验项，可通过 TRAP_BLUE_REDUCE 关闭）
            if hp.LIGHT_MODE in {"blue", "dual"}:
                phi_b = hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
                p_trap_base = p_trap_base * (1.0 - hp.TRAP_BLUE_REDUCE * phi_b)

            # 黏膜物理化学场：把“致密度/网孔 + 剪切变稀”映射到困陷概率（基础值）
            if hp.USE_MUCUS_PHYS:
                _, _, p_trap_base = apply_mucus_phys(
                    v0=np.ones_like(x, dtype=float),
                    dr0=np.ones_like(x, dtype=float),
                    p_trap_base=p_trap_base,
                    x=x,
                    gamma_eff=gamma_eff,
                )

            # 剪切单峰增强：模仿“中等剪切更易发生近界面困陷/黏附”的现象（Yeo 核心现象的工程化表达）
            if hp.USE_SHEAR_TRAP:
                p_trap = p_trap_base * (1.0 + hp.SHEAR_TRAP_GAIN * shear_peak_factor(gamma_eff, hp.GAMMA_CRIT))
            else:
                p_trap = p_trap_base
            p_trap = np.clip(p_trap, 0.0, 0.25)

            entering = alive & (~trapped) & (rng.random(hp.N) < p_trap)
            # 记录 trap 进入事件（用于后续画“黏附/困陷 vs 剪切”）
            trap_events[k] = int(entering.sum())
            in_int_layer = is_int_ab(x) | is_int_bc(x)
            trap_events_int[k] = int((entering & in_int_layer).sum())
            trap_time_left[entering] = rng.exponential(1.0 / hp.TRAP_RELEASE_RATE, size=entering.sum())
            trapped[entering] = True
            trap_time_left[trapped] = np.maximum(0.0, trap_time_left[trapped] - hp.DT_S)

        # ---------------------------------------------------------------------
        # 9.6) 判断本步哪些个体发生 tumble（泊松过程）
        # 依据：run->tumble 常可由无记忆速率过程描述，
        # 即 P(tumble in dt) = 1 - exp(-lambda*dt)。
        # ---------------------------------------------------------------------
        p_tumble = 1.0 - np.exp(-lam * hp.DT_S)
        tumble = alive & (~trapped) & (rng.random(hp.N) < p_tumble)
        live_count = max(int(alive.sum()), 1)
        run_mask_now = alive & (~trapped) & (~tumble)
        frac_run[k] = float(run_mask_now.sum()) / float(live_count)
        frac_tumble[k] = float(tumble.sum()) / float(live_count)
        frac_trapped[k] = float((alive & trapped).sum()) / float(live_count)

        # 更新连续 run 时长（用于穿越门槛判断）
        run_streak[trapped] = 0.0
        run_streak[tumble] = 0.0
        run_streak[alive & (~trapped) & (~tumble)] += hp.DT_S

        # ---------------------------------------------------------------------
        # 9.7) 更新朝向：
        # - tumble：重新均匀采样方向（最简模型）
        # - run：旋转扩散 + 可选剪切漂移
        # 依据：旋转扩散近似朝向噪声；剪切漂移表示剪切流中的简化对齐趋势。
        # ---------------------------------------------------------------------
        run_mask = alive & (~trapped) & (~tumble)

        # 光趋化转向（phototaxis steering）：run 期间把朝向弱拉向 +x（theta_target=0）
        # 这样即便主流沿 y，也能逐步积累净 +x 迁移，从而提高到达/穿透概率。
        if hp.USE_PHOTOTAXIS_STEERING and (hp.LIGHT_MODE in {"blue", "dual"}):
            # steering 也使用 phi_photo（与 tumble 偏置一致），但建议保持很弱，避免方向性不真实地接近 1
            theta_target = 0.0  # +x 方向
            theta[run_mask] += hp.PHOTO_TURN_GAIN * phi_photo[run_mask] * np.sin(theta_target - theta[run_mask]) * hp.DT_S

        # 趋化转向：在 run 期间把朝向轻微“拉向”营养梯度方向（让趋化效果更明显）
        if hp.USE_CHEMOTAXIS and hp.USE_CHEM_STEERING and (r_for_steer is not None):
            if abs(hp.GX) < 1e-12 and abs(hp.GY) < 1e-12:
                theta_grad = 0.0
            else:
                theta_grad = float(np.arctan2(hp.GY, hp.GX))
            r_clip = np.clip(r_for_steer, -hp.CHEM_CLIP, hp.CHEM_CLIP)
            theta[run_mask] += hp.CHEM_TURN_GAIN * r_clip[run_mask] * np.sin(theta_grad - theta[run_mask]) * hp.DT_S

        if hp.USE_SHEAR_DRIFT:
            theta[run_mask] += 0.5 * gamma_eff[run_mask] * (-1.0 + hp.BETA_SHAPE * np.cos(2.0 * theta[run_mask])) * hp.DT_S

        v0 = v0_by_region(x)
        dr0 = dr_by_region(x)
        dt0 = dt_by_region(x)

        if hp.USE_MUCUS_PROFILE:
            v0, dr0, dt0 = apply_mucus_scaling(v0, dr0, dt0, x)

        # 黏膜物理化学场：致密度/网孔 + 剪切变稀（Carreau–Yasuda）映射到 v0、Dr
        if hp.USE_MUCUS_PHYS:
            v0, dr0, _ = apply_mucus_phys(
                v0=v0,
                dr0=dr0,
                p_trap_base=np.ones_like(x, dtype=float),
                x=x,
                gamma_eff=gamma_eff,
            )

        # 蓝光下：假设电路/行为使得细菌更“直跑且更快”（经验项，可关掉）
        if hp.LIGHT_MODE in {"blue", "dual"}:
            phi_b = hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
            v0 = v0 * (1.0 + hp.V_BLUE_GAIN * phi_b)

        # 无光惩罚：无光时游动更慢
        if hp.LIGHT_MODE == "off":
            v0 = v0 * hp.V0_DARK_FACTOR

        theta[run_mask] += np.sqrt(2.0 * dr0[run_mask] * hp.DT_S) * rng.standard_normal(run_mask.sum())

        # tumble 发生时的“重定向规则”决定了是否会产生净迁移。
        # 1) 蓝光 phototaxis：tumble 后更倾向朝 +x（von Mises，kappa 随“走向更亮处”信号增强）
        # 2) 化学趋化（原实现）：tumble 后更倾向朝营养梯度方向（kappa 随 r>0 增强）
        # 3) 否则：均匀随机（无偏）
        if hp.USE_PHOTOTAXIS_BIASED_TUMBLE and (hp.LIGHT_MODE in {"blue", "dual"}):
            idx = np.where(tumble)[0]
            if idx.size > 0:
                # 用上面构造好的 phi_photo（时间导数为主 + 绝对光强为辅）来决定偏置强度
                kappa_b = hp.PHOTO_BIAS_KAPPA_BASE + hp.PHOTO_BIAS_KAPPA_GAIN * phi_photo
                kappa_b = np.clip(kappa_b, 0.0, hp.PHOTO_BIAS_KAPPA_MAX)
                theta_target = 0.0  # +x 方向
                # κ=0 时等价于均匀分布；κ 越大越集中在 theta_target
                theta[idx] = rng.vonmises(theta_target, kappa_b[idx], size=idx.size)

        elif hp.USE_CHEMOTAXIS and hp.USE_CHEM_BIASED_TUMBLE and (r_for_steer is not None):
            # 梯度方向（单位向量）对应的角度
            if abs(hp.GX) < 1e-12 and abs(hp.GY) < 1e-12:
                theta_grad = 0.0
            else:
                theta_grad = float(np.arctan2(hp.GY, hp.GX))

            # 用 r=S-m 的“正值部分”来增强偏置：r>0 表示环境在变好 -> tumble 后更愿意朝向梯度
            r_clip = np.clip(r_for_steer, -hp.CHEM_CLIP, hp.CHEM_CLIP)
            kappa_all = hp.CHEM_BIAS_KAPPA_BASE + hp.CHEM_BIAS_KAPPA_GAIN * np.maximum(r_clip, 0.0)
            kappa_all = np.clip(kappa_all, 0.0, hp.CHEM_BIAS_KAPPA_MAX)

            idx = np.where(tumble)[0]
            if idx.size > 0:
                theta[idx] = rng.vonmises(theta_grad, kappa_all[idx], size=idx.size)

        else:
            theta[tumble] = rng.uniform(-np.pi, np.pi, size=tumble.sum())

        # ---------------------------------------------------------------------
        # 9.8) 更新未困陷个体的位置
        # 依据：x,y 由平流 + 沿 theta 的自主推进共同更新。
        # 可选：将 tumble 视为纯重定向（不平移）。
        # ---------------------------------------------------------------------
        move_mask = alive & (~trapped)
        speed = v0.copy()
        if hp.TUMBLE_NO_TRANSLATION:
            speed[tumble] = 0.0

        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        # 版本B：主流沿 y。先更新未包裹的 y_unwrap，再根据需要把 y 映射回观测窗口。
        x[move_mask] += (ux[move_mask] + vx[move_mask]) * hp.DT_S

        # 版本B：主流沿 y。先更新未包裹的 y_unwrap，再根据需要把 y 映射回观测窗口。
        y_unwrap[move_mask] += (uy[move_mask] + vy[move_mask]) * hp.DT_S
        y[move_mask] = y_unwrap[move_mask]

        # 周期边界：把 y 包裹回 [Y_MIN, Y_MAX]，避免轨迹图被 y 方向平流“冲出画面”。
        if hp.USE_Y_PERIODIC:
            span = hp.Y_MAX - hp.Y_MIN
            y[move_mask] = ((y[move_mask] - hp.Y_MIN) % span) + hp.Y_MIN

        # 可选平移扩散
        if hp.USE_TRANSLATIONAL_DIFFUSION:
            x[move_mask] += np.sqrt(2.0 * dt0[move_mask] * hp.DT_S) * rng.standard_normal(move_mask.sum())

            # 对未包裹 y_unwrap 加噪声，之后再重新包裹到观测窗口
            y_unwrap[move_mask] += np.sqrt(2.0 * dt0[move_mask] * hp.DT_S) * rng.standard_normal(move_mask.sum())
            y[move_mask] = y_unwrap[move_mask]
            if hp.USE_Y_PERIODIC:
                span = hp.Y_MAX - hp.Y_MIN
                y[move_mask] = ((y[move_mask] - hp.Y_MIN) % span) + hp.Y_MIN

        # 先更新“累计到达”标记（用当前 x 判断；即使后续被删除也保留到达事实）
        newly_interface = (~reached_interface) & (x >= hp.X_INTERFACE)
        newly_target = (~reached_target) & (x >= hp.X_TARGET)
        t_first_interface[newly_interface] = t
        t_first_target[newly_target] = t
        reached_interface |= (x >= hp.X_INTERFACE)
        reached_target |= (x >= hp.X_TARGET)

        # ---------------------------------------------------------------------
        # 9.8b) 右边界穿透：触碰 x>=X_MAX 认为已穿透（删除）
        # 说明：被删除个体不再参与后续动力学/界面反弹/困陷等事件。
        # 同时将其 x 固定为 X_MAX，方便指标统计（例如到达比例/通量）。
        # ---------------------------------------------------------------------
        exited = alive & (x >= hp.X_MAX)
        if np.any(exited):
            penetrated_count += int(exited.sum())
            t_first_exit[exited] = t
            reached_exit[exited] = True
            alive[exited] = False
            trapped[exited] = False
            trap_time_left[exited] = 0.0
            run_streak[exited] = 0.0
            x[exited] = hp.X_MAX

        # ---------------------------------------------------------------------
        # 9.9) 区域界面的穿越规则：
        # 需要最小连续 run 时长才能穿越 A->B 和 B->C。
        # 依据：用粗粒度方式表示“需要持续直跑才能穿透屏障”。
        # ---------------------------------------------------------------------
        attempted_ab = alive & (x_prev < hp.REG_A_END) & (x >= hp.REG_A_END)
        gate_fail_ab = attempted_ab & (run_streak < hp.RUN_REQ_AB_S)

        if hp.USE_PROB_PASS:
            p_ab = pass_prob(gamma_eff, Ib_local, hp.PASS_P0_AB)
            rand_fail_ab = attempted_ab & (rng.random(hp.N) > p_ab)
        else:
            rand_fail_ab = np.zeros(hp.N, dtype=bool)

        blocked_ab = gate_fail_ab | rand_fail_ab
        if np.any(blocked_ab):
            # 被界面阻挡：位置卡在界面前一丢丢
            x[blocked_ab] = hp.REG_A_END - hp.X_INT_EPS
            # 关闭“碰撞反射”时，不翻转朝向；保持原朝向，让其继续在界面附近被流/随机项带走
            if hp.USE_REFLECT_INTERFACE:
                theta[blocked_ab] = np.pi - theta[blocked_ab]

        attempted_bc = alive & (x_prev < hp.REG_B_END) & (x >= hp.REG_B_END)
        gate_fail_bc = attempted_bc & (run_streak < hp.RUN_REQ_BC_S)

        if hp.USE_PROB_PASS:
            p_bc = pass_prob(gamma_eff, Ib_local, hp.PASS_P0_BC)
            rand_fail_bc = attempted_bc & (rng.random(hp.N) > p_bc)
        else:
            rand_fail_bc = np.zeros(hp.N, dtype=bool)

        blocked_bc = gate_fail_bc | rand_fail_bc
        if np.any(blocked_bc):
            # BC 失败时不再钳死在同一条线：在界面带内做小回退，减少“反复抽签卡死”
            idx_bc = np.where(blocked_bc)[0]
            backstep = rng.uniform(hp.BC_FAIL_BACKSTEP_MIN_UM, hp.BC_FAIL_BACKSTEP_MAX_UM, size=idx_bc.size)
            x_new = np.minimum(x[idx_bc] - backstep, hp.REG_B_END - hp.X_INT_EPS)
            x[idx_bc] = np.maximum(hp.REG_B_END - hp.DELTA_INT_BC, x_new)
            if hp.USE_REFLECT_INTERFACE:
                theta[idx_bc] = np.pi - theta[idx_bc]

        # ---------------------------------------------------------------------
        # 9.10) 边界处理
        # 说明：不再对右侧 x=X_MAX 和上下 y 边界做反射；仅保留左侧反射。
        # ---------------------------------------------------------------------
        under = alive & (x < hp.X_MIN)
        if np.any(under):
            if hp.USE_REFLECT_LEFT_WALL:
                # 镜面反射（旧行为）
                x[under] = 2 * hp.X_MIN - x[under]
                theta[under] = np.pi - theta[under]
            else:
                # 关闭反射：把位置钳制回边界，不改变朝向
                x[under] = hp.X_MIN

        # ---------------------------------------------------------------------
        # 9.11) 指标统计
        # ---------------------------------------------------------------------
        # ---- 界面停留比例 & 界面剪切均值（用于单峰/非单调分析） ----
        in_int_now = alive & (is_int_ab(x) | is_int_bc(x))
        frac_in_interface[k] = float(in_int_now.sum()) / float(hp.N)
        if np.any(in_int_now):
            gamma_int_mean[k] = float(np.mean(gamma_eff[in_int_now]))

        # mean_x 仅统计仍在域内的个体（已穿透删除者不再影响域内平均位置）
        mean_x[k] = x[alive].mean() if np.any(alive) else hp.X_MAX

        # 到达比例使用“累计到达”而非仅统计 alive，避免删除后曲线掉回 0
        frac_interface[k] = reached_interface.mean()
        forward_cross_int = (x_prev < hp.X_INTERFACE) & (x >= hp.X_INTERFACE)
        forward_cross_int = forward_cross_int & (x_prev < hp.X_MAX)
        cross_flux_int[k] = forward_cross_int.sum() / hp.DT_S

        frac_target[k] = reached_target.mean()
        forward_cross = (x_prev < hp.X_TARGET) & (x >= hp.X_TARGET)
        forward_cross = forward_cross & (x_prev < hp.X_MAX)
        cross_flux[k] = forward_cross.sum() / hp.DT_S

        dx = x - x0
        dy = y_unwrap - y0
        msd[k] = (dx * dx + dy * dy).mean()

        counts_xt, _ = np.histogram(x[alive], bins=x_edges_xt)
        xt_density[k] = counts_xt.astype(np.float32)

        if (k % max(1, hp.STATE_SAMPLE_STRIDE) == 0) and np.any(alive):
            counts_xy, _, _ = np.histogram2d(
                y[alive],
                x[alive],
                bins=[y_edges_xy, x_edges_xy],
            )
            occupancy_xy += counts_xy.astype(np.float32)

        traj_x[:, k + 1] = x[plot_idx]
        traj_y[:, k + 1] = y[plot_idx]
        traj_y_unwrap[:, k + 1] = y_unwrap[plot_idx]

        # 每隔 FRAME_STEP_S 保存一帧（共 n_frames 份，便于后期 ffmpeg 拼视频）
        if use_frame_export and ((k + 1) % frame_steps == 0):
            frame_idx = (k + 1) // frame_steps - 1
            if frame_idx < n_frames:
                t_frame = float((k + 1) * hp.DT_S)
                frame_metrics = {
                    "directionality": float(dir_t[k]),
                    "mean_x_um": float(mean_x[k]),
                    "frac_target_cumulative": float(reached_target.mean()),
                    "penetration_rate_cumulative": float(reached_exit.mean()),
                    "alive_count": int(alive.sum()),
                    "mean_lambda": float(mean_lambda_t[k]),
                    "frac_run": float(frac_run[k]),
                    "frac_tumble": float(frac_tumble[k]),
                    "frac_trapped": float(frac_trapped[k]),
                }
                _save_all_plot_frames(
                    plot_dirs,
                    frame_idx,
                    t_frame,
                    k,
                    hp,
                    traj_x=traj_x,
                    traj_y=traj_y,
                    dir_t=dir_t,
                    mean_x=mean_x,
                    msd=msd,
                    u_t=u_t,
                    frac_target=frac_target,
                    frac_interface=frac_interface,
                    cross_flux=cross_flux,
                    cross_flux_int=cross_flux_int,
                    frac_run=frac_run,
                    frac_tumble=frac_tumble,
                    frac_trapped=frac_trapped,
                    mean_lambda_t=mean_lambda_t,
                    mean_c_t=mean_c_t,
                    xt_density=xt_density,
                    occupancy_xy=occupancy_xy,
                    x_edges_xt=x_edges_xt,
                    penetration_rate=float(reached_exit.mean()),
                    window_steps=window_steps,
                )
                frame_manifest.append({
                    "frame_idx": str(frame_idx),
                    "t_s": f"{t_frame:.3f}",
                    **{k_m: str(v_m) for k_m, v_m in frame_metrics.items()},
                })
                progress.update(1)
        elif not use_frame_export:
            progress.update(1)

    progress.close()

    video_entries: List[Dict[str, str]] = []
    if use_frame_export and run_dir is not None:
        video_entries = _compose_all_plot_videos(plot_dirs, hp, run_dir)
        manifest = {
            "T_s": hp.T_S,
            "FRAME_STEP_s": hp.FRAME_STEP_S,
            "DT_s": hp.DT_S,
            "n_frames": len(frame_manifest),
            "n_frames_expected": n_frames,
            "video_fps": hp.VIDEO_FPS,
            "n_videos": len(video_entries),
            "videos": video_entries,
            "plot_frame_dirs": {pid: str(pdir) for pid, pdir in plot_dirs.items()},
            "frames": frame_manifest,
        }
        (run_dir / "frames_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"逐秒帧已保存: {run_dir / hp.PLOT_FRAMES_SUBDIR} "
            f"（{len(frame_manifest)} 秒 × {len(plot_dirs)} 类图）；视频 -> {run_dir / hp.VIDEOS_SUBDIR}"
        )

    # =============================================================================
    # 10) 后处理与保存
    # =============================================================================
    dir_smooth = moving_average(dir_t, window_steps)
    mean_x_smooth = moving_average(mean_x, window_steps)
    frac_interface_smooth = moving_average(frac_interface, window_steps)
    cross_flux_int_smooth = moving_average(cross_flux_int, window_steps)
    frac_target_smooth = moving_average(frac_target, window_steps)
    cross_flux_smooth = moving_average(cross_flux, window_steps)
    msd_smooth = moving_average(msd, window_steps)
    time_axis = (np.arange(steps) + 1) * hp.DT_S

    # 保存时间序列 CSV
    if save_outputs and run_dir is not None:
        cols = np.column_stack([
            time_axis,
            dir_t, dir_smooth,
            mean_x, mean_x_smooth,
            frac_interface, frac_interface_smooth,
            cross_flux_int, cross_flux_int_smooth,
            frac_target, frac_target_smooth,
            cross_flux, cross_flux_smooth,
            msd, msd_smooth,
            u_t, mean_c_inf_t, mean_c_t, mean_lambda_t
        ])
        header = (
            "t_s,dir_raw,dir_smoothed,mean_x_raw,mean_x_smoothed,"
            "frac_int_raw,frac_int_smoothed,cross_flux_int_raw,cross_flux_int_smoothed,"
            "frac_target_raw,frac_target_smoothed,cross_flux_raw,cross_flux_smoothed,"
            "msd_raw,msd_smoothed,u_eff_mean,c_inf_mean,c_mean,lambda_mean"
        )
        np.savetxt(run_dir / "metrics_timeseries.csv", cols, delimiter=",", header=header, comments="")

    # 汇总指标（仿真结束）
    forward_disp = float((x - x0).mean())  # 沿 +x 方向
    directionality_end = float(dir_t[-1])
    light_dose = float(np.nansum(u_t) * hp.DT_S) if hp.LIGHT_MAPPING == "chey" else float("nan")
    frac_target_end = float(reached_target.mean())
    penetration_rate = float(reached_exit.mean())
    valid_exit_times = t_first_exit[np.isfinite(t_first_exit)]
    mfpt_s = float(np.mean(valid_exit_times)) if valid_exit_times.size > 0 else float("nan")
    inverse_mfpt_per_s = 1.0 / mfpt_s if np.isfinite(mfpt_s) and mfpt_s > 0.0 else float("nan")
    common_metrics = summarize_transport(
        t_first_target,
        n_agents=hp.N,
        penetration_count=int(reached_exit.sum()),
        final_x=x,
        initial_x=x0,
        final_theta=theta,
        trap_events=int(trap_events.sum()),
        duration_s=hp.T_S,
    )

    # ---- sweep/论文对齐用的汇总标量 ----
    gamma_int_empirical = float(np.nanmean(gamma_int_mean)) if np.any(np.isfinite(gamma_int_mean)) else float("nan")
    gamma_int_ref = interface_gamma_reference(hp)
    gamma_int_avg = gamma_int_ref if not np.isfinite(gamma_int_empirical) else gamma_int_empirical
    trap_rate_per_s_per_cell = float(trap_events.sum()) / max(hp.T_S * hp.N, 1e-12)
    trap_rate_int_per_s_per_cell = float(trap_events_int.sum()) / max(hp.T_S * hp.N, 1e-12)
    interface_residence_mean = interface_contact_reference_rate(frac_in_interface, hp)

    summary = {
        "N": hp.N,
        "T_s": hp.T_S,
        "DT_s": hp.DT_S,
        "forward_displacement_um_mean": forward_disp,
        "directionality_mean_last": float(dir_t[-window_steps:].mean()),
        "frac_target_end": frac_target_end,
        "penetration_rate_xmax": penetration_rate,
        "penetrated_count": penetrated_count,
        "MFPT_exit_s": mfpt_s,
        "inverse_MFPT_per_s": inverse_mfpt_per_s,
        "MFPT_successful_samples": int(valid_exit_times.size),
        "MFPT_censored_samples": int(hp.N - valid_exit_times.size),
        "light_dose_s": light_dose,
        "D_tilde": hp.SEQUENCE_RESPONSE_D_TILDE,
        "I_light": hp.LIGHT_INTENSITY,
        "LIGHT_MODE": hp.LIGHT_MODE,
        "LIGHT_MAPPING": hp.LIGHT_MAPPING,
        "USE_CHEMOTAXIS": hp.USE_CHEMOTAXIS,
        "USE_MUCUS_PROFILE": hp.USE_MUCUS_PROFILE,
        "USE_MUCUS_PHYS": hp.USE_MUCUS_PHYS,
        # --- sweep 相关 ---
        "gamma_interface_mean": gamma_int_avg,
        "gamma_interface_empirical": gamma_int_empirical,
        "gamma_interface_reference": gamma_int_ref,
        "trap_rate_per_s_per_cell": trap_rate_per_s_per_cell,
        "trap_rate_int_per_s_per_cell": trap_rate_int_per_s_per_cell,
        "interface_residence_mean": interface_residence_mean,
        "mean_lambda_last": float(np.mean(mean_lambda_t[-window_steps:])),
        "mean_c_last": float(np.mean(mean_c_t[-window_steps:])),
        "mean_run_fraction_last": float(np.mean(frac_run[-window_steps:])),
        "mean_tumble_fraction_last": float(np.mean(frac_tumble[-window_steps:])),
        "mean_trapped_fraction_last": float(np.mean(frac_trapped[-window_steps:])),
        "first_passage_interface_mean_s": float(np.nanmean(t_first_interface)) if np.any(np.isfinite(t_first_interface)) else float("nan"),
        "first_passage_target_mean_s": float(np.nanmean(t_first_target)) if np.any(np.isfinite(t_first_target)) else float("nan"),
        "first_passage_exit_mean_s": mfpt_s,
        "GAMMA_MULT": hp.GAMMA_MULT,
    }
    summary.update(common_metrics)
    if save_outputs and run_dir is not None:
        first_passage_table = np.column_stack([
            np.arange(hp.N, dtype=int),
            t_first_interface,
            t_first_target,
            t_first_exit,
        ])
        np.savetxt(
            run_dir / "first_passage_times.csv",
            first_passage_table,
            delimiter=",",
            header="agent_id,interface_time_s,target_time_s,exit_time_s",
            comments="",
            fmt=["%d", "%.9g", "%.9g", "%.9g"],
        )
        (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 运行结束后直接打印核心结果，便于终端里快速看穿透表现
    print(
        "[RESULT] "
        f"mode={hp.LIGHT_MODE}, "
        f"penetration_rate_xmax={penetration_rate:.4f}, "
        f"frac_target_end={frac_target_end:.4f}, "
        f"penetrated_count={penetrated_count}, "
        f"forward_disp_um_mean={forward_disp:.2f}, "
        f"D_tilde={hp.SEQUENCE_RESPONSE_D_TILDE:.3f}, "
        f"MFPT={mfpt_s:.6g}s, "
        f"1/MFPT={inverse_mfpt_per_s:.6g}/s"
    )

    # =============================================================================
    # 11) 绘图
    # =============================================================================
    if make_plots:
        fig_tag = f"mode={hp.LIGHT_MODE}, mapping={hp.LIGHT_MAPPING}"

        # 图 1：轨迹（增强可视化：更清晰的区域背景 + 轨迹减噪 + 轻网格）
        # 轨迹图按 16:9 输出（横向拉长）
        fig1, ax = plt.subplots(figsize=(16, 9))
        ax.set_facecolor("#f8fafc")
        for i in range(traj_x.shape[0]):
            # 周期包裹会造成 y 从 Y_MAX 跳到 Y_MIN（或相反），直接连线会出现整屏竖线。
            # 这里在包裹跳变处插入 NaN 断开曲线。
            xx = traj_x[i].copy()
            yy = traj_y[i].copy()
            if hp.USE_Y_PERIODIC:
                span = hp.Y_MAX - hp.Y_MIN
                wrap_jump = np.abs(np.diff(traj_y[i])) > 0.5 * span
                if np.any(wrap_jump):
                    cut_idx = np.where(wrap_jump)[0] + 1
                    yy[cut_idx] = np.nan
            ax.plot(xx, yy, linewidth=0.75, alpha=0.8)
        ax.axvspan(hp.X_MIN, hp.REG_A_END, alpha=0.10, color="#dbeafe")
        ax.axvspan(hp.REG_A_END, hp.REG_B_END, alpha=0.12, color="#fde68a")
        ax.axvspan(hp.REG_B_END, hp.X_MAX, alpha=0.10, color="#e9d5ff")
        ax.axvline(hp.REG_A_END, linestyle="--", linewidth=1.2, color="#1d4ed8")
        ax.axvline(hp.REG_B_END, linestyle="--", linewidth=1.2, color="#1d4ed8")
        ax.text(0.5 * (hp.X_MIN + hp.REG_A_END), -0.08, "Region A", transform=ax.get_xaxis_transform(),
                ha="center", va="top", clip_on=False)
        ax.text(0.5 * (hp.REG_A_END + hp.REG_B_END), -0.08, "Region B (mucus surface)", transform=ax.get_xaxis_transform(),
                ha="center", va="top", clip_on=False)
        ax.text(0.5 * (hp.REG_B_END + hp.X_MAX), -0.08, "Region C (mucus deep)", transform=ax.get_xaxis_transform(),
                ha="center", va="top", clip_on=False)
        ax.text(
            0.02,
            0.98,
            f"Penetration @ {hp.X_MAX:.0f} um: {penetration_rate * 100.0:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
        )
        ax.set_title(f"Sample Trajectories ({fig_tag})")
        ax.set_xlabel("x (um)")
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.set_ylabel("y (um)")
        ax.set_xlim(hp.X_MIN, hp.X_MAX)
        ax.set_ylim(hp.Y_MIN, hp.Y_MAX)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        fig1.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
        if save_outputs and run_dir is not None:
            fig1.savefig(run_dir / "01_sample_trajectories.png", dpi=300)

        # 图 2：方向性
        fig2 = plt.figure()
        plt.plot(time_axis, dir_t, label="Raw")
        plt.plot(time_axis, dir_smooth, label=f"Smoothed ({hp.WINDOW_S:.1f}s)")
        plt.title(f"Directionality ({fig_tag})")
        plt.xlabel("t (s)")
        plt.ylabel("mean cos(theta - theta_d)")
        plt.legend()
        plt.tight_layout()
        if save_outputs and run_dir is not None:
            fig2.savefig(run_dir / "02_directionality.png", dpi=300)

        # 图 3：x 均值
        fig3 = plt.figure()
        plt.plot(time_axis, mean_x, label="Raw")
        plt.plot(time_axis, mean_x_smooth, label=f"Smoothed ({hp.WINDOW_S:.1f}s)")
        plt.title(f"Mean x Position ({fig_tag})")
        plt.xlabel("t (s)")
        plt.ylabel("mean x (um)")
        plt.legend()
        plt.tight_layout()
        if save_outputs and run_dir is not None:
            fig3.savefig(run_dir / "03_mean_x_position.png", dpi=300)

        # 图 3b：均方位移（MSD）
        fig3b = plt.figure()
        plt.plot(time_axis, msd, label="Raw")
        plt.plot(time_axis, msd_smooth, label=f"Smoothed ({hp.WINDOW_S:.1f}s)")
        plt.title(f"Mean Squared Displacement ({fig_tag})")
        plt.xlabel("t (s)")
        plt.ylabel("MSD (um^2)")
        plt.legend()
        plt.tight_layout()
        if save_outputs and run_dir is not None:
            fig3b.savefig(run_dir / "03b_msd.png", dpi=300)

        # 图 4：到达指标（增强可视化：fraction/flux 分面显示，原始值降透明度避免遮挡）
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(11.5, 7.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        # 上图：fraction
        ax1.plot(time_axis, frac_target, color="#2563eb", alpha=0.22, linewidth=1.0, label="Target Fraction (Raw)")
        ax1.plot(time_axis, frac_target_smooth, color="#2563eb", linewidth=2.0, label=f"Target Fraction (Smoothed {hp.WINDOW_S:.1f}s)")
        ax1.plot(time_axis, frac_interface, color="#16a34a", alpha=0.22, linewidth=1.0, label="Interface Fraction (Raw)")
        ax1.plot(time_axis, frac_interface_smooth, color="#16a34a", linewidth=2.0, label=f"Interface Fraction (Smoothed {hp.WINDOW_S:.1f}s)")
        ax1.set_ylabel("fraction")
        ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        ax1.legend(loc="upper left", ncol=2, fontsize=9)
        ax1.set_title(f"Target Arrival Metrics ({fig_tag})")

        # 下图：crossing flux
        ax2.plot(time_axis, cross_flux, color="#f97316", alpha=0.22, linewidth=1.0, label="Crossing Flux @ X_TARGET (Raw)")
        ax2.plot(time_axis, cross_flux_smooth, color="#f97316", linewidth=2.0, label=f"Crossing Flux @ X_TARGET (Smoothed {hp.WINDOW_S:.1f}s)")
        ax2.plot(time_axis, cross_flux_int, color="#dc2626", alpha=0.22, linewidth=1.0, label="Crossing Flux @ X_INTERFACE (Raw)")
        ax2.plot(time_axis, cross_flux_int_smooth, color="#dc2626", linewidth=2.0, label=f"Crossing Flux @ X_INTERFACE (Smoothed {hp.WINDOW_S:.1f}s)")
        ax2.set_ylabel("forward crossings/s")
        ax2.set_xlabel("t (s)")
        ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        ax2.legend(loc="upper left", ncol=2, fontsize=9)

        fig.tight_layout()
        if save_outputs and run_dir is not None:
            fig.savefig(run_dir / "04_target_arrival_metrics.png", dpi=300)

        # 可选：若使用 CheY 映射则绘制 u(t)
        if hp.LIGHT_MAPPING == "chey":
            figu = plt.figure()
            plt.plot(time_axis, u_t, linewidth=1.2)
            plt.title("Control input u(t) (CheY mapping)")
            plt.xlabel("t (s)")
            plt.ylabel("u (0/1)")
            plt.ylim(-0.05, 1.05)
            plt.tight_layout()
            if save_outputs and run_dir is not None:
                figu.savefig(run_dir / "05_u_of_t.png", dpi=300)

        # 图 6：x-t 群体密度热图
        fig6, ax6 = plt.subplots(figsize=(10.8, 5.2))
        xt_show = xt_density.T
        im6 = ax6.imshow(
            xt_show,
            origin="lower",
            aspect="auto",
            extent=[time_axis[0], time_axis[-1], hp.X_MIN, hp.X_MAX],
        )
        ax6.axhline(hp.REG_A_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
        ax6.axhline(hp.REG_B_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
        ax6.set_title(f"x-t Density Heatmap ({fig_tag})")
        ax6.set_xlabel("t (s)")
        ax6.set_ylabel("x (um)")
        cbar6 = fig6.colorbar(im6, ax=ax6)
        cbar6.set_label("cell count per x-bin")
        fig6.tight_layout()
        if save_outputs and run_dir is not None:
            fig6.savefig(run_dir / "06_xt_density_heatmap.png", dpi=300)

        # 图 7：x-y 停留热图
        fig7, ax7 = plt.subplots(figsize=(12.0, 5.5))
        occ_show = occupancy_xy.copy()
        im7 = ax7.imshow(
            occ_show,
            origin="lower",
            aspect="auto",
            extent=[hp.X_MIN, hp.X_MAX, hp.Y_MIN, hp.Y_MAX],
        )
        ax7.axvline(hp.REG_A_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
        ax7.axvline(hp.REG_B_END, linestyle="--", linewidth=1.0, color="white", alpha=0.9)
        ax7.set_title(f"x-y Occupancy Heatmap ({fig_tag})")
        ax7.set_xlabel("x (um)")
        ax7.set_ylabel("y (um)")
        cbar7 = fig7.colorbar(im7, ax=ax7)
        cbar7.set_label("accumulated occupancy")
        fig7.tight_layout()
        if save_outputs and run_dir is not None:
            fig7.savefig(run_dir / "07_xy_occupancy_heatmap.png", dpi=300)

        # 图 8：run / tumble / trapped 状态堆叠图
        fig8, ax8 = plt.subplots(figsize=(11.2, 5.0))
        ax8.stackplot(
            time_axis,
            frac_run,
            frac_tumble,
            frac_trapped,
            labels=["run", "tumble", "trapped"],
            alpha=0.85,
        )
        ax8.set_ylim(0.0, 1.0)
        ax8.set_title(f"State Fractions Over Time ({fig_tag})")
        ax8.set_xlabel("t (s)")
        ax8.set_ylabel("fraction of alive cells")
        ax8.legend(loc="upper right")
        ax8.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        fig8.tight_layout()
        if save_outputs and run_dir is not None:
            fig8.savefig(run_dir / "08_state_fractions.png", dpi=300)

        # 图 9：CheY / lambda 机制链图
        fig9, axes9 = plt.subplots(3, 1, figsize=(11.2, 8.2), sharex=True)
        axes9[0].plot(time_axis, dir_t, alpha=0.25, linewidth=1.0, label="A raw")
        axes9[0].plot(time_axis, dir_smooth, linewidth=2.0, label="A smoothed")
        axes9[0].set_ylabel("A")
        axes9[0].legend(loc="upper right")
        axes9[0].grid(True, linestyle=":", linewidth=0.7, alpha=0.25)

        axes9[1].plot(time_axis, mean_lambda_t, linewidth=1.8, label="mean lambda")
        axes9[1].plot(time_axis, mean_c_t, linewidth=1.5, label="mean c")
        if hp.LIGHT_MAPPING == "chey":
            axes9[1].plot(time_axis, u_t, linewidth=1.2, label="u(t)")
        axes9[1].set_ylabel("control / internal state")
        axes9[1].legend(loc="upper right")
        axes9[1].grid(True, linestyle=":", linewidth=0.7, alpha=0.25)

        axes9[2].plot(time_axis, cross_flux_int_smooth, linewidth=1.8, label="cross flux @ interface")
        axes9[2].plot(time_axis, frac_target_smooth, linewidth=1.8, label="target fraction")
        axes9[2].set_xlabel("t (s)")
        axes9[2].set_ylabel("transport metrics")
        axes9[2].legend(loc="upper right")
        axes9[2].grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        fig9.suptitle(f"Mechanism Diagnostics ({fig_tag})")
        fig9.tight_layout()
        if save_outputs and run_dir is not None:
            fig9.savefig(run_dir / "09_mechanism_diagnostics.png", dpi=300)

        # 图 10：首次到达时间分布
        valid_interface = t_first_interface[np.isfinite(t_first_interface)]
        valid_target = t_first_target[np.isfinite(t_first_target)]
        valid_exit = t_first_exit[np.isfinite(t_first_exit)]
        fig10, axes10 = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=False)
        datasets10 = [
            (valid_interface, "First passage to interface"),
            (valid_target, "First passage to target"),
            (valid_exit, "First passage to exit"),
        ]
        for ax10, (data10, title10) in zip(axes10, datasets10):
            if data10.size > 0:
                ax10.hist(data10, bins=24, alpha=0.85)
                ax10.axvline(float(np.mean(data10)), linestyle="--", linewidth=1.2, color="black", label=f"mean={np.mean(data10):.1f}s")
                ax10.legend(loc="upper right", fontsize=8)
            else:
                ax10.text(0.5, 0.5, "No arrivals", ha="center", va="center", transform=ax10.transAxes)
            ax10.set_title(title10)
            ax10.set_xlabel("t (s)")
            ax10.set_ylabel("cell count")
            ax10.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        fig10.suptitle(f"First-passage Time Distributions ({fig_tag})")
        fig10.tight_layout()
        if save_outputs and run_dir is not None:
            fig10.savefig(run_dir / "10_first_passage_histograms.png", dpi=300)

        # 图 11：最终朝向极坐标分布（按区域）
        fig11 = plt.figure(figsize=(12.0, 4.0))
        region_masks = [
            (alive & is_region_a(x), "Region A"),
            (alive & is_region_b(x), "Region B"),
            (alive & is_region_c(x), "Region C"),
        ]
        theta_bins = np.linspace(-np.pi, np.pi, hp.THETA_HIST_BINS + 1)
        theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
        theta_width = theta_bins[1] - theta_bins[0]
        for i_region, (mask_region, title_region) in enumerate(region_masks, start=1):
            axp = fig11.add_subplot(1, 3, i_region, projection="polar")
            if np.any(mask_region):
                hist_theta, _ = np.histogram(theta[mask_region], bins=theta_bins)
                axp.bar(theta_centers, hist_theta, width=theta_width, bottom=0.0, alpha=0.85)
            axp.set_title(title_region)
        fig11.suptitle(f"Final Heading Distribution by Region ({fig_tag})")
        fig11.tight_layout()
        if save_outputs and run_dir is not None:
            fig11.savefig(run_dir / "11_heading_polar_by_region.png", dpi=300)

        if save_outputs:
            print(f"Results saved to: {run_dir}")
        if hp.SHOW_PLOTS:
            plt.show()
        else:
            plt.close("all")

    if return_diagnostics:
        # Added only after JSON saving so NumPy arrays never enter the summary file.
        summary["_diagnostics"] = {
            "occupancy_xy": occupancy_xy.copy(),
            "x_edges_xy": x_edges_xy.copy(),
            "y_edges_xy": y_edges_xy.copy(),
            "first_passage_interface_s": t_first_interface.copy(),
            "first_passage_target_s": t_first_target.copy(),
            "first_passage_exit_s": t_first_exit.copy(),
        }
    return summary
def _hp_copy(base: HyperParams) -> HyperParams:
    """浅拷贝一份超参数，便于 sweep 时逐点修改。"""
    return HyperParams(**asdict(base))


def run_shear_sweep(base_hp: HyperParams) -> None:
    """
    扫描等效剪切强度（通过 GAMMA_MULT 缩放 gamma_eff），并对比：
      A) trap/界面停留 vs 剪切（期望单峰）
      B) 穿透率 vs 剪切（期望非单调：中剪切最低）
    同时对比两种模式：蓝光 vs 无光。
    """
    apply_mucus_layer_thickness(base_hp)

    gammas = list(base_hp.SWEEP_GAMMA_MULT_LIST)
    modes = list(base_hp.SWEEP_LIGHT_MODES)
    R = max(1, int(base_hp.SWEEP_REPEATS))

    out = {
        m: {
            "gamma": [],
            "gamma_emp": [],
            "trap": [],
            "trap_int": [],
            "res_int": [],
            "pen": [],
            "trap_sd": [],
            "trap_int_sd": [],
            "res_int_sd": [],
            "pen_sd": [],
        }
        for m in modes
    }
    # 固定并对调两条主曲线颜色：off=橙色，blue=蓝色
    mode_colors = {"off": "#ff7f0e", "blue": "#1f77b4"}

    for mode in modes:
        for gm in gammas:
            trap_vals = []
            trap_int_vals = []
            res_int_vals = []
            pen_vals = []
            gamma_ref_vals = []
            gamma_emp_vals = []

            for r in range(R):
                hp = _hp_copy(base_hp)
                hp.LIGHT_MODE = mode
                # sweep 时固定使用当前配置指定的映射；
                # 若你想专门比较 direct_dualcolor，可直接把 base_hp.LIGHT_MAPPING 设成该值。
                hp.LIGHT_MAPPING = base_hp.LIGHT_MAPPING

                hp.GAMMA_MULT = float(gm)
                hp.SEED = int(base_hp.SEED + 1000 * (modes.index(mode) + 1) + 37 * r)

                summ = simulate(hp, make_plots=False, save_outputs=bool(base_hp.SWEEP_SAVE_EACH_RUN))
                # 调试保护：如果 sweep 后不同点的动力学指标几乎不变，
                # 说明还有参数没有真正进入动力学。
                # 这里先把关键量都存下来，便于检查是否真的随剪切变化。
                trap_vals.append(float(summ["trap_rate_per_s_per_cell"]))
                trap_int_vals.append(float(summ["trap_rate_int_per_s_per_cell"]))
                res_int_vals.append(float(summ["interface_residence_mean"]))
                pen_vals.append(float(summ["penetration_rate_xmax"]))
                gamma_ref_vals.append(float(summ["gamma_interface_reference"]))
                gamma_emp_vals.append(
                    float(summ["gamma_interface_empirical"])
                    if np.isfinite(summ["gamma_interface_empirical"])
                    else np.nan
                )

            out[mode]["gamma"].append(float(np.mean(gamma_ref_vals)))
            out[mode]["gamma_emp"].append(float(np.nanmean(gamma_emp_vals)) if np.any(np.isfinite(gamma_emp_vals)) else float("nan"))
            out[mode]["trap"].append(float(np.mean(trap_vals)))
            out[mode]["trap_int"].append(float(np.mean(trap_int_vals)))
            out[mode]["res_int"].append(float(np.mean(res_int_vals)))
            out[mode]["pen"].append(float(np.mean(pen_vals)))
            out[mode]["trap_sd"].append(float(np.std(trap_vals, ddof=0)))
            out[mode]["trap_int_sd"].append(float(np.std(trap_int_vals, ddof=0)))
            out[mode]["res_int_sd"].append(float(np.std(res_int_vals, ddof=0)))
            out[mode]["pen_sd"].append(float(np.std(pen_vals, ddof=0)))

    out_dir = Path(base_hp.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_csv = out_dir / "sweep_metrics.csv"
    with sweep_csv.open("w", encoding="utf-8") as f:
        f.write("mode,gamma_ref,gamma_empirical,trap_mean,trap_sd,trap_int_mean,trap_int_sd,res_int_mean,res_int_sd,pen_mean,pen_sd\n")
        for mode in modes:
            for i in range(len(out[mode]["gamma"])):
                f.write(
                    f"{mode},{out[mode]['gamma'][i]},{out[mode]['gamma_emp'][i]},{out[mode]['trap'][i]},{out[mode]['trap_sd'][i]},{out[mode]['trap_int'][i]},{out[mode]['trap_int_sd'][i]},{out[mode]['res_int'][i]},{out[mode]['res_int_sd'][i]},{out[mode]['pen'][i]},{out[mode]['pen_sd'][i]}\n"
                )

    figA = plt.figure(figsize=(7.2, 4.8))
    for mode in modes:
        g = np.array(out[mode]["gamma"], dtype=float)
        y = np.array(out[mode]["trap_int"], dtype=float)
        yerr = np.array(out[mode]["trap_int_sd"], dtype=float)
        order = np.argsort(g)
        c = mode_colors.get(mode, None)
        plt.errorbar(
            g[order],
            y[order],
            yerr=yerr[order],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"trap@interface ({mode})",
            color=c,
            ecolor=c,
        )
    plt.axvline(base_hp.GAMMA_CRIT, linestyle="--", linewidth=1.0, label="gamma_crit (model)")
    plt.xlabel("mean gamma_eff in interface")
    plt.ylabel("trap entry rate (1/s/cell) in interface")
    plt.title("A. Trap / adhesion proxy vs shear")
    plt.legend()
    plt.tight_layout()
    figA.savefig(out_dir / base_hp.SWEEP_FIG_TRAP, dpi=300)

    figB = plt.figure(figsize=(7.2, 4.8))
    for mode in modes:
        g = np.array(out[mode]["gamma"], dtype=float)
        y = np.array(out[mode]["pen"], dtype=float)
        yerr = np.array(out[mode]["pen_sd"], dtype=float)
        order = np.argsort(g)
        c = mode_colors.get(mode, None)
        plt.errorbar(
            g[order],
            y[order],
            yerr=yerr[order],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"penetration ({mode})",
            color=c,
            ecolor=c,
        )
    plt.axvline(base_hp.GAMMA_CRIT, linestyle="--", linewidth=1.0, label="gamma_crit (model)")
    plt.xlabel("mean gamma_eff in interface")
    plt.ylabel("penetration rate (x>=X_MAX)")
    plt.title("B. Penetration vs shear")
    plt.legend()
    plt.tight_layout()
    figB.savefig(out_dir / base_hp.SWEEP_FIG_PEN, dpi=300)

    # 同时输出一份简短控制台摘要，方便快速判断 sweep 是否真的扫出了变化。
    for mode in modes:
        print(f"[SWEEP] mode={mode}")
        for i in range(len(out[mode]["gamma"])):
            print(
                f"  gamma_ref={out[mode]['gamma'][i]:.4f}, "
                f"trap_int={out[mode]['trap_int'][i]:.6f}, "
                f"res_int={out[mode]['res_int'][i]:.6f}, "
                f"pen={out[mode]['pen'][i]:.6f}"
            )
    print(f"Shear sweep figures saved to: {out_dir / base_hp.SWEEP_FIG_TRAP} and {out_dir / base_hp.SWEEP_FIG_PEN}")
    print(f"Sweep metrics saved to: {sweep_csv}")


# 说明：
# 本次修正后，若 A/B 两张 sweep 图仍接近水平直线，
# 说明当前参数区间下动力学对剪切并不敏感，而不是 sweep 代码失效。
# 但在修正前，更主要的问题是：局部 hp 没有同步到全局 HP，
# 导致“横轴变了，动力学没变”，所以图必然是假直线。


if __name__ == "__main__":
    if HP.ENABLE_SHEAR_SWEEP:
        run_shear_sweep(HP)
    else:
        simulate(HP)
