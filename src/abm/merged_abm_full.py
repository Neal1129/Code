from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["font.family"] = ["Times New Roman", "SimSun"]
mpl.rcParams["axes.unicode_minus"] = False
@dataclass
class HyperParams:
    SEED: int = 7
    N: int = 2000
    T_S: float = 180.0
    DT_S: float = 0.05
    OUTPUT_DIR: str = r"D:\SMILING\project2\建模组\Merged_ABM_outputs"
    SHOW_PLOTS: bool = False
    N_PLOT: int = 120
    WINDOW_S: float = 0.5
    X_BINS_HEATMAP: int = 80
    Y_BINS_HEATMAP: int = 80
    XT_BINS_X: int = 80
    STATE_SAMPLE_STRIDE: int = 5
    THETA_HIST_BINS: int = 36
    X_MIN: float = 0.0
    X_MAX: float = 600.0
    Y_MIN: float = -8000.0
    Y_MAX: float = 0.0
    USE_MUCUS_LAYER_THICKNESS_CONTROL: bool = True
    MUCUS_B_THICKNESS_UM: float = 200.0
    MUCUS_C_THICKNESS_UM: float = 200.0
    USE_Y_PERIODIC: bool = True
    REG_A_END: float = 200.0
    REG_B_END: float = 400.0
    RUN_REQ_AB_S: float = 0.40
    RUN_REQ_BC_S: float = 0.35
    X_INT_EPS: float = 1e-6
    DELTA_INT_AB: float = 12.0
    DELTA_INT_BC: float = 12.0
    USE_PROB_PASS: bool = True
    PASS_P0_AB: float = 0.55
    PASS_P0_BC: float = 0.8
    PASS_ALPHA_GAMMA: float = 0.12
    PASS_BETA_BLUE: float = 0.80
    PASS_BLUE_I_HALF: float = 0.45
    PASS_BLUE_HILL_N: float = 2.0
    PASS_CLIP_MIN: float = 0.02
    PASS_CLIP_MAX: float = 0.98
    BC_FAIL_BACKSTEP_MIN_UM: float = 1.0
    BC_FAIL_BACKSTEP_MAX_UM: float = 5.0
    USE_MUCUS_PHYS: bool = True
    M_B: float = 0.6
    M_C: float = 1.0
    M_X0: float = 400.0
    M_SCALE: float = 25.0
    TRAP_M_GAIN: float = 0.8
    V_M_ALPHA: float = 0.12
    DR_M_ALPHA: float = 0.15
    MU0_CY: float = 3.0
    MU_INF: float = 0.6
    LAM_CY: float = 0.8
    N_CY: float = 0.35
    A_CY: float = 2.0
    V_MU_POWER: float = 0.7
    DR_MU_POWER: float = 0.25
    DELTA_U: float = 8.0
    U_FAST: float = 80.0
    U_SLOW: float = 30.0
    GAMMA_WALL: float = 0.0
    GAMMA_MULT: float = 1.0
    ENABLE_SHEAR_SWEEP: bool = False
    SWEEP_GAMMA_MULT_LIST: Tuple[float, ...] = (0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    SWEEP_LIGHT_MODES: Tuple[str, ...] = ("off", "blue")
    SWEEP_REPEATS: int = 5
    SWEEP_SAVE_EACH_RUN: bool = False
    SWEEP_FIG_TRAP: str = "06_sweep_trap_vs_shear.png"
    SWEEP_FIG_PEN: str = "07_sweep_penetration_vs_shear.png"
    V0_A: float = 20.0
    DR_A: float = 0.35
    V0_B: float = 14.0
    DR_B: float = 0.30
    V0_C: float = 9.0
    DR_C: float = 0.18
    USE_MUCUS_PROFILE: bool = False
    MU0: float = 1.0
    MU1: float = 4.0
    MU_X0: float = 320.0
    MU_SCALE: float = 40.0
    MU_SPEED_POWER: float = 1.0
    MU_DR_POWER: float = 1.0
    MU_DT_POWER: float = 1.0
    USE_TRANSLATIONAL_DIFFUSION: bool = False
    DT_A: float = 0.2
    DT_B: float = 0.12
    DT_C: float = 0.08
    LAMBDA_BASE: float = 0.45
    LAMBDA_FLOOR: float = 0.05
    LAMBDA_CEIL: float = 2.00
    LAMBDA_DARK_BOOST: float = 0.35
    V0_DARK_FACTOR: float = 0.6
    TRAP_DARK_MULT: float = 3
    LIGHT_MODE: str = "blue"
    LIGHT_MAPPING: str = "chey"
    GRAD_CENTER: float = 30.0
    GRAD_SCALE: float = 60.0
    USE_TIME_POLICY: bool = True
    BLUE_ON: float = 1.0
    GREEN_ON: float = 1.0
    GREEN_PULSE_PERIOD: float = 2.0
    GREEN_PULSE_WIDTH: float = 0.25
    DLAMBDA_BLUE: float = 0.35
    BLUE_HILL_N: float = 2.0
    BLUE_I_HALF: float = 0.45
    DLAMBDA_GREEN: float = 0.80
    GREEN_HILL_N: float = 2.0
    GREEN_I_HALF: float = 0.45
    USE_PHOTOTAXIS_BIASED_TUMBLE: bool = True
    USE_PHOTOTAXIS_STEERING: bool = False
    USE_PHOTO_MEMORY: bool = True
    TAU_PHOTO_MEM_S: float = 1.2
    PHOTO_R_CLIP: float = 0.6
    PHOTO_ABS_WEIGHT: float = 0.30
    PHOTO_BIAS_I_HALF: float = 0.35
    PHOTO_BIAS_HILL_N: float = 2.0
    PHOTO_BIAS_KAPPA_BASE: float = 0.0
    PHOTO_BIAS_KAPPA_GAIN: float = 12.0
    PHOTO_BIAS_KAPPA_MAX: float = 40.0
    PHOTO_TURN_GAIN: float = 0.6
    V_BLUE_GAIN: float = 0.25
    TRAP_BLUE_REDUCE: float = 0.15
    USE_GATE_CONTROL: bool = False
    TAU_C_S: float = 0.6
    C_INF_DARK: float = 0.75
    C_INF_LIT: float = 0.25
    C_HALF: float = 0.5
    C_HILL_N: float = 3.0
    LAMBDA_MIN: float = 0.08
    LAMBDA_MAX: float = 1.20
    GATE_A_ON: float = 0.25
    GATE_A_OFF: float = 0.10
    GATE_MIN_HOLD_S: float = 0.25
    CHEY_PULSE_FREQ_HZ: float = 1.0
    CHEY_PULSE_DUTY: float = 0.50
    CHEY_PULSE_PHASE_S: float = 0.0
    CHEY_MODE: str = "continuous"
    USE_TRAP: bool = True
    P_TRAP_OUT: float = 0.0002
    P_TRAP_INT: float = 0.0060
    P_TRAP_MUCUS: float = 0.0020
    P_TRAP_AB_LAYER: float = 0.0030
    P_TRAP_BC_LAYER: float = 0.0015
    TRAP_RELEASE_RATE: float = 1.0
    USE_SHEAR_TRAP: bool = True
    GAMMA_CRIT: float = 2.0
    SHEAR_TRAP_GAIN: float = 2.0
    USE_SHEAR_DRIFT: bool = False
    BETA_SHAPE: float = 0.7
    USE_CHEMOTAXIS: bool = False
    C0: float = 1.0
    GX: float = 0.01
    GY: float = 0.0
    C_FLOOR: float = 1e-3
    CHEM_SIGNAL: str = "log"
    TAU_MEM_S: float = 0.6
    CHEM_GAIN: float = 6.0
    CHEM_CLIP: float = 1.5
    USE_CHEM_STEERING: bool = True
    CHEM_TURN_GAIN: float = 2.5
    USE_CHEM_BIASED_TUMBLE: bool = True
    CHEM_BIAS_KAPPA_BASE: float = 0.0
    CHEM_BIAS_KAPPA_GAIN: float = 16.0
    CHEM_BIAS_KAPPA_MAX: float = 200.0
    X_INTERFACE: float = 200.0
    X_TARGET: float = 500.0
    TUMBLE_NO_TRANSLATION: bool = True
    USE_REFLECT_INTERFACE: bool = False
    USE_REFLECT_LEFT_WALL: bool = False
def apply_mucus_layer_thickness(hp: HyperParams) -> None:
    if not hp.USE_MUCUS_LAYER_THICKNESS_CONTROL:
        return
    hp.REG_B_END = hp.REG_A_END + hp.MUCUS_B_THICKNESS_UM
    hp.X_MAX = hp.REG_B_END + hp.MUCUS_C_THICKNESS_UM
    hp.M_X0 = hp.REG_B_END
    hp.X_INTERFACE = hp.REG_A_END
    hp.X_TARGET = hp.REG_B_END + 0.5 * hp.MUCUS_C_THICKNESS_UM
HP = HyperParams()
apply_mucus_layer_thickness(HP)
if HP.LIGHT_MODE not in {"dual", "blue", "green", "off"}:
    raise ValueError("LIGHT_MODE must be one of: dual, blue, green, off")
if HP.LIGHT_MAPPING not in {"direct_dualcolor", "chey"}:
    raise ValueError("LIGHT_MAPPING must be one of: direct_dualcolor, chey")
def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))
def hill01(I: np.ndarray, k_half: float, n: float) -> np.ndarray:
    I = np.clip(I, 0.0, 1.0)
    num = I ** n
    den = (k_half ** n + num)
    return num / den
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[w:] - c[:-w]) / w
    pad_left = w // 2
    pad_right = len(x) - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode="edge")
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
def ux_base(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x, dtype=float)
def uy_base(x: np.ndarray) -> np.ndarray:
    s = sigmoid((HP.REG_A_END - x) / HP.DELTA_U)
    return -(HP.U_SLOW + (HP.U_FAST - HP.U_SLOW) * s)
def ux_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return ux_base(x)
def uy_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return uy_base(x) + HP.GAMMA_WALL * y
def shear_rate_field(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    s = sigmoid((HP.REG_A_END - x) / HP.DELTA_U)
    ds_dz = s * (1.0 - s)
    duy_dx = -(HP.U_FAST - HP.U_SLOW) * ds_dz / HP.DELTA_U
    duy_dy = HP.GAMMA_WALL
    return HP.GAMMA_MULT * (np.abs(duy_dx) + np.abs(duy_dy))
def shear_peak_factor(gamma: np.ndarray, gamma_crit: float) -> np.ndarray:
    x = np.maximum(gamma, 0.0) / max(gamma_crit, 1e-12)
    return x * np.exp(1.0 - x)
def interface_gamma_reference(hp: HyperParams) -> float:
    n_x = 33
    n_y = 65
    x_ab = np.linspace(hp.REG_A_END - hp.DELTA_INT_AB, hp.REG_A_END + hp.DELTA_INT_AB, n_x)
    x_bc = np.linspace(hp.REG_B_END - hp.DELTA_INT_BC, hp.REG_B_END + hp.DELTA_INT_BC, n_x)
    y_line = np.linspace(hp.Y_MIN, hp.Y_MAX, n_y)
    Xab, Yab = np.meshgrid(x_ab, y_line)
    Xbc, Ybc = np.meshgrid(x_bc, y_line)
    s_ab = sigmoid((hp.REG_A_END - Xab.ravel()) / hp.DELTA_U)
    s_bc = sigmoid((hp.REG_A_END - Xbc.ravel()) / hp.DELTA_U)
    duy_dx_ab = -(hp.U_FAST - hp.U_SLOW) * s_ab * (1.0 - s_ab) / hp.DELTA_U
    duy_dx_bc = -(hp.U_FAST - hp.U_SLOW) * s_bc * (1.0 - s_bc) / hp.DELTA_U
    duy_dy = np.abs(hp.GAMMA_WALL)
    gab = hp.GAMMA_MULT * (np.abs(duy_dx_ab) + duy_dy)
    gbc = hp.GAMMA_MULT * (np.abs(duy_dx_bc) + duy_dy)
    return float(np.mean(np.concatenate([gab, gbc])))
def interface_contact_reference_rate(frac_in_interface: np.ndarray, hp: HyperParams) -> float:
    _ = hp
    return float(np.mean(frac_in_interface))
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
    s = sigmoid((x - HP.MU_X0) / HP.MU_SCALE)
    return HP.MU0 + (HP.MU1 - HP.MU0) * s
def apply_mucus_scaling(v0: np.ndarray, dr0: np.ndarray, dt0: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = mucus_mu(x)
    v = v0 * np.power(mu, -HP.MU_SPEED_POWER)
    dr = dr0 * np.power(mu, HP.MU_DR_POWER)
    dt = dt0 * np.power(mu, HP.MU_DT_POWER)
    return v, dr, dt
def mucus_density_M(x: np.ndarray) -> np.ndarray:
    s = sigmoid((x - HP.M_X0) / HP.M_SCALE)
    return HP.M_B + (HP.M_C - HP.M_B) * s
def mu_carreau_yasuda(gamma: np.ndarray) -> np.ndarray:
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
    M = mucus_density_M(x)
    v0 = v0 * np.exp(-HP.V_M_ALPHA * M)
    dr0 = dr0 * np.exp(-HP.DR_M_ALPHA * M)
    p_trap_base = p_trap_base * (1.0 + HP.TRAP_M_GAIN * M)
    mu_eff = mu_carreau_yasuda(gamma_eff)
    v0 = v0 * np.power(mu_eff, -HP.V_MU_POWER)
    dr0 = dr0 * np.power(mu_eff, HP.DR_MU_POWER)
    return v0, dr0, p_trap_base
def light_profile_x(x: np.ndarray) -> np.ndarray:
    return sigmoid((x - HP.GRAD_CENTER) / HP.GRAD_SCALE)
def time_gate_blue(t: float) -> float:
    return HP.BLUE_ON if HP.USE_TIME_POLICY else 1.0
def time_gate_green(t: float) -> float:
    if not HP.USE_TIME_POLICY:
        return 0.0
    phase = t % HP.GREEN_PULSE_PERIOD
    return HP.GREEN_ON if phase < HP.GREEN_PULSE_WIDTH else 0.0
def light_blue(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:

    if HP.LIGHT_MODE in {"green", "off"}:
        return np.zeros_like(x, dtype=float)
    return np.clip(light_profile_x(x) * time_gate_blue(t), 0.0, 1.0)
def light_green(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    if HP.LIGHT_MODE in {"blue", "off"}:
        return np.zeros_like(x, dtype=float)
    return np.clip(light_profile_x(x) * time_gate_green(t), 0.0, 1.0)
class GateController:
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
    if HP.CHEY_MODE == "off":
        return 0.0
    if HP.CHEY_MODE == "continuous":
        return 1.0
    period = 1.0 / max(HP.CHEY_PULSE_FREQ_HZ, 1e-12)
    phase = (t + HP.CHEY_PULSE_PHASE_S) % period
    return 1.0 if phase < HP.CHEY_PULSE_DUTY * period else 0.0
def chey_update_c(c: np.ndarray, u: float, dt: float) -> np.ndarray:
    c_inf = HP.C_INF_LIT if u >= 0.5 else HP.C_INF_DARK
    alpha = np.exp(-dt / max(HP.TAU_C_S, 1e-12))
    return c_inf + (c - c_inf) * alpha
def chey_c_to_lambda(c: np.ndarray) -> np.ndarray:
    cw = hill01(np.clip(c, 0.0, 1.0), HP.C_HALF, HP.C_HILL_N)
    lam = HP.LAMBDA_MIN + (HP.LAMBDA_MAX - HP.LAMBDA_MIN) * cw
    return np.clip(lam, HP.LAMBDA_FLOOR, HP.LAMBDA_CEIL)

def nutrient_C(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    C = HP.C0 + HP.GX * x + HP.GY * y
    return np.maximum(C, HP.C_FLOOR)
def chem_signal(C: np.ndarray) -> np.ndarray:
    if HP.CHEM_SIGNAL == "linear":
        return C
    return np.log(C)
def chem_update_memory(m: np.ndarray, S: np.ndarray, dt: float) -> np.ndarray:
    alpha = np.exp(-dt / max(HP.TAU_MEM_S, 1e-12))
    return S + (m - S) * alpha
def chem_modulate_lambda(lam: np.ndarray, r: np.ndarray) -> np.ndarray:
    r_clip = np.clip(r, -HP.CHEM_CLIP, HP.CHEM_CLIP)
    factor = np.exp(-HP.CHEM_GAIN * r_clip)
    return np.clip(lam * factor, HP.LAMBDA_FLOOR, HP.LAMBDA_CEIL)
def trap_prob_base_by_region(x: np.ndarray) -> np.ndarray:
    p = np.full_like(x, HP.P_TRAP_MUCUS, dtype=float)
    p[is_region_a(x)] = HP.P_TRAP_OUT
    p[is_region_b(x)] = HP.P_TRAP_INT
    p[is_region_c(x)] = HP.P_TRAP_MUCUS
    p[is_int_ab(x)] = HP.P_TRAP_AB_LAYER
    p[is_int_bc(x)] = HP.P_TRAP_BC_LAYER
    return p
def pass_prob(gamma: np.ndarray, ib: np.ndarray, p0: float) -> np.ndarray:
    shear_factor = np.exp(-HP.PASS_ALPHA_GAMMA * np.maximum(gamma, 0.0))
    blue_factor = 1.0 + HP.PASS_BETA_BLUE * hill01(
        np.clip(ib, 0.0, 1.0),
        HP.PASS_BLUE_I_HALF,
        HP.PASS_BLUE_HILL_N,
    )
    p = p0 * shear_factor * blue_factor
    return np.clip(p, HP.PASS_CLIP_MIN, HP.PASS_CLIP_MAX)
def simulate(hp: HyperParams, *, make_plots: bool = True, save_outputs: bool = True) -> Dict[str, float]:
    global HP
    HP = HyperParams(**asdict(hp))
    hp = HP
    apply_mucus_layer_thickness(hp)
    rng = np.random.default_rng(hp.SEED)
    steps = int(hp.T_S / hp.DT_S)
    run_dir: Path | None = None
    if save_outputs:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(hp.OUTPUT_DIR) / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "hyperparams.json").write_text(json.dumps(asdict(hp), indent=2), encoding="utf-8")
    x = rng.uniform(20.0, hp.REG_A_END - 20.0, size=hp.N)
    y = rng.uniform(hp.Y_MIN + 20.0, hp.Y_MAX - 20.0, size=hp.N)
    theta = rng.uniform(-np.pi, np.pi, size=hp.N)
    y_unwrap = y.copy()
    x0 = x.copy()
    y0 = y_unwrap.copy()
    run_streak = np.zeros(hp.N, dtype=float)
    trapped = np.zeros(hp.N, dtype=bool)
    trap_time_left = np.zeros(hp.N, dtype=float)
    alive = np.ones(hp.N, dtype=bool)
    penetrated_count = 0
    reached_interface = np.zeros(hp.N, dtype=bool)
    reached_target = np.zeros(hp.N, dtype=bool)
    reached_exit = np.zeros(hp.N, dtype=bool)
    release_prob = 1.0 - np.exp(-hp.TRAP_RELEASE_RATE * hp.DT_S) if hp.USE_TRAP else 0.0
    m = np.zeros(hp.N, dtype=float)
    photo_mem = np.zeros(hp.N, dtype=float)
    c = np.full(hp.N, 0.5 * (hp.C_INF_DARK + hp.C_INF_LIT), dtype=float)
    gate = GateController(hp.GATE_A_ON, hp.GATE_A_OFF, hp.GATE_MIN_HOLD_S)
    plot_idx = rng.choice(hp.N, size=min(hp.N_PLOT, hp.N), replace=False)
    traj_x = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_y = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_y_unwrap = np.empty((plot_idx.size, steps + 1), dtype=np.float32)
    traj_x[:, 0] = x[plot_idx]
    traj_y[:, 0] = y[plot_idx]
    traj_y_unwrap[:, 0] = y_unwrap[plot_idx]
    dir_t = np.empty(steps, dtype=np.float32)
    mean_x = np.empty(steps, dtype=np.float32)
    frac_interface = np.empty(steps, dtype=np.float32)
    cross_flux_int = np.empty(steps, dtype=np.float32)
    frac_target = np.empty(steps, dtype=np.float32)
    cross_flux = np.empty(steps, dtype=np.float32)
    msd = np.empty(steps, dtype=np.float32)
    u_t = np.empty(steps, dtype=np.float32)
    frac_run = np.empty(steps, dtype=np.float32)
    frac_tumble = np.empty(steps, dtype=np.float32)
    frac_trapped = np.empty(steps, dtype=np.float32)
    mean_lambda_t = np.empty(steps, dtype=np.float32)
    mean_c_t = np.empty(steps, dtype=np.float32)
    trap_events = np.zeros(steps, dtype=np.int32)
    trap_events_int = np.zeros(steps, dtype=np.int32)
    frac_in_interface = np.zeros(steps, dtype=np.float32)
    gamma_int_mean = np.full(steps, np.nan, dtype=np.float32)
    x_edges_xt = np.linspace(hp.X_MIN, hp.X_MAX, hp.XT_BINS_X + 1)
    xt_density = np.zeros((steps, hp.XT_BINS_X), dtype=np.float32)
    x_edges_xy = np.linspace(hp.X_MIN, hp.X_MAX, hp.X_BINS_HEATMAP + 1)
    y_edges_xy = np.linspace(hp.Y_MIN, hp.Y_MAX, hp.Y_BINS_HEATMAP + 1)
    occupancy_xy = np.zeros((hp.Y_BINS_HEATMAP, hp.X_BINS_HEATMAP), dtype=np.float32)
    t_first_interface = np.full(hp.N, np.nan, dtype=np.float32)
    t_first_target = np.full(hp.N, np.nan, dtype=np.float32)
    t_first_exit = np.full(hp.N, np.nan, dtype=np.float32)
    x_prev = x.copy()
    d_hat = np.array([1.0, 0.0], dtype=float)
    for k in range(steps):
        t = (k + 1) * hp.DT_S
        x_prev[:] = x
        vhatx = np.cos(theta)
        vhaty = np.sin(theta)
        dot = vhatx * d_hat[0] + vhaty * d_hat[1]
        dot[trapped] = 0.0
        dot[~alive] = 0.0
        A = float(dot.mean())
        dir_t[k] = A
        Ib_local = np.zeros(hp.N, dtype=float)
        if hp.LIGHT_MAPPING == "direct_dualcolor":
            Ib = light_blue(x, y, t)
            Ig = light_green(x, y, t)
            Ib_local = Ib.copy()
            Ib_local[~alive] = 0.0
            fb = hill01(Ib, hp.BLUE_I_HALF, hp.BLUE_HILL_N)
            fg = hill01(Ig, hp.GREEN_I_HALF, hp.GREEN_HILL_N)
            lam = hp.LAMBDA_BASE - hp.DLAMBDA_BLUE * fb + hp.DLAMBDA_GREEN * fg
            lam = np.clip(lam, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            if hp.LIGHT_MODE == "off":
                lam = np.clip(lam + hp.LAMBDA_DARK_BOOST, hp.LAMBDA_FLOOR, hp.LAMBDA_CEIL)
            u_t[k] = np.nan
        else:
            u = gate.step(A, hp.DT_S) if hp.USE_GATE_CONTROL else chey_u_of_t(t)
            u_t[k] = u
            c[:] = chey_update_c(c, u, hp.DT_S)
            lam = chey_c_to_lambda(c)
            Ib_local = np.full(hp.N, u, dtype=float)
            Ib_local[~alive] = 0.0
        mean_lambda_t[k] = float(np.mean(lam[alive])) if np.any(alive) else float(np.mean(lam))
        mean_c_t[k] = float(np.mean(c[alive])) if np.any(alive) else float(np.mean(c))
        if hp.LIGHT_MODE in {"blue", "dual"}:
            I_sig = np.clip(Ib_local, 0.0, 1.0)
        else:
            I_sig = np.zeros(hp.N, dtype=float)
        phi_I = hill01(np.clip(I_sig, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
        if hp.USE_PHOTO_MEMORY:
            alpha_p = np.exp(-hp.DT_S / max(hp.TAU_PHOTO_MEM_S, 1e-12))
            photo_mem[:] = I_sig + (photo_mem - I_sig) * alpha_p
            r_photo = I_sig - photo_mem
            r_photo = np.clip(r_photo, -hp.PHOTO_R_CLIP, hp.PHOTO_R_CLIP)
        else:
            r_photo = I_sig
        phi_d = np.maximum(r_photo, 0.0) / max(hp.PHOTO_R_CLIP, 1e-12)
        phi_d = np.clip(phi_d, 0.0, 1.0)
        phi_photo = np.clip(phi_d + hp.PHOTO_ABS_WEIGHT * phi_I, 0.0, 1.0)
        if hp.USE_CHEMOTAXIS:
            C = nutrient_C(x, y)
            S = chem_signal(C)
            m[:] = chem_update_memory(m, S, hp.DT_S)
            r = S - m
            lam = chem_modulate_lambda(lam, r)
            r_for_steer = r
        else:
            r_for_steer = None
        ux = ux_field(x, y, t)
        uy = uy_field(x, y, t)
        gamma_eff = shear_rate_field(x, y, t)
        if hp.USE_TRAP:
            releasing = trapped & alive & (rng.random(hp.N) < release_prob)
            trapped[releasing] = False
            trap_time_left[releasing] = 0.0
            p_trap_base = trap_prob_base_by_region(x)
            if hp.LIGHT_MODE == "off":
                p_trap_base = p_trap_base * hp.TRAP_DARK_MULT
            if hp.LIGHT_MODE in {"blue", "dual"}:
                phi_b = hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
                p_trap_base = p_trap_base * (1.0 - hp.TRAP_BLUE_REDUCE * phi_b)
            if hp.USE_MUCUS_PHYS:
                _, _, p_trap_base = apply_mucus_phys(
                    v0=np.ones_like(x, dtype=float),
                    dr0=np.ones_like(x, dtype=float),
                    p_trap_base=p_trap_base,
                    x=x,
                    gamma_eff=gamma_eff,
                )
            if hp.USE_SHEAR_TRAP:
                p_trap = p_trap_base * (1.0 + hp.SHEAR_TRAP_GAIN * shear_peak_factor(gamma_eff, hp.GAMMA_CRIT))
            else:
                p_trap = p_trap_base
            p_trap = np.clip(p_trap, 0.0, 0.25)
            entering = alive & (~trapped) & (rng.random(hp.N) < p_trap)
            trap_events[k] = int(entering.sum())
            in_int_layer = is_int_ab(x) | is_int_bc(x)
            trap_events_int[k] = int((entering & in_int_layer).sum())
            trap_time_left[entering] = rng.exponential(1.0 / hp.TRAP_RELEASE_RATE, size=entering.sum())
            trapped[entering] = True
            trap_time_left[trapped] = np.maximum(0.0, trap_time_left[trapped] - hp.DT_S)
        p_tumble = 1.0 - np.exp(-lam * hp.DT_S)
        tumble = alive & (~trapped) & (rng.random(hp.N) < p_tumble)
        live_count = max(int(alive.sum()), 1)
        run_mask_now = alive & (~trapped) & (~tumble)
        frac_run[k] = float(run_mask_now.sum()) / float(live_count)
        frac_tumble[k] = float(tumble.sum()) / float(live_count)
        frac_trapped[k] = float((alive & trapped).sum()) / float(live_count)
        run_streak[trapped] = 0.0
        run_streak[tumble] = 0.0
        run_streak[alive & (~trapped) & (~tumble)] += hp.DT_S
        run_mask = alive & (~trapped) & (~tumble)
        if hp.USE_PHOTOTAXIS_STEERING and (hp.LIGHT_MODE in {"blue", "dual"}):
            theta_target = 0.0
            theta[run_mask] += hp.PHOTO_TURN_GAIN * phi_photo[run_mask] * np.sin(theta_target - theta[run_mask]) * hp.DT_S
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
        if hp.USE_MUCUS_PHYS:
            v0, dr0, _ = apply_mucus_phys(
                v0=v0,
                dr0=dr0,
                p_trap_base=np.ones_like(x, dtype=float),
                x=x,
                gamma_eff=gamma_eff,
            )
        if hp.LIGHT_MODE in {"blue", "dual"}:
            phi_b = hill01(np.clip(Ib_local, 0.0, 1.0), hp.PHOTO_BIAS_I_HALF, hp.PHOTO_BIAS_HILL_N)
            v0 = v0 * (1.0 + hp.V_BLUE_GAIN * phi_b)
        if hp.LIGHT_MODE == "off":
            v0 = v0 * hp.V0_DARK_FACTOR
        theta[run_mask] += np.sqrt(2.0 * dr0[run_mask] * hp.DT_S) * rng.standard_normal(run_mask.sum())
        if hp.USE_PHOTOTAXIS_BIASED_TUMBLE and (hp.LIGHT_MODE in {"blue", "dual"}):
            idx = np.where(tumble)[0]
            if idx.size > 0:
                kappa_b = hp.PHOTO_BIAS_KAPPA_BASE + hp.PHOTO_BIAS_KAPPA_GAIN * phi_photo
                kappa_b = np.clip(kappa_b, 0.0, hp.PHOTO_BIAS_KAPPA_MAX)
                theta_target = 0.0
                theta[idx] = rng.vonmises(theta_target, kappa_b[idx], size=idx.size)
        elif hp.USE_CHEMOTAXIS and hp.USE_CHEM_BIASED_TUMBLE and (r_for_steer is not None):
            if abs(hp.GX) < 1e-12 and abs(hp.GY) < 1e-12:
                theta_grad = 0.0
            else:
                theta_grad = float(np.arctan2(hp.GY, hp.GX))
            r_clip = np.clip(r_for_steer, -hp.CHEM_CLIP, hp.CHEM_CLIP)
            kappa_all = hp.CHEM_BIAS_KAPPA_BASE + hp.CHEM_BIAS_KAPPA_GAIN * np.maximum(r_clip, 0.0)
            kappa_all = np.clip(kappa_all, 0.0, hp.CHEM_BIAS_KAPPA_MAX)
            idx = np.where(tumble)[0]
            if idx.size > 0:
                theta[idx] = rng.vonmises(theta_grad, kappa_all[idx], size=idx.size)
        else:
            theta[tumble] = rng.uniform(-np.pi, np.pi, size=tumble.sum())
        move_mask = alive & (~trapped)
        speed = v0.copy()
        if hp.TUMBLE_NO_TRANSLATION:
            speed[tumble] = 0.0
        vx = speed * np.cos(theta)
        vy = speed * np.sin(theta)
        x[move_mask] += (ux[move_mask] + vx[move_mask]) * hp.DT_S
        y_unwrap[move_mask] += (uy[move_mask] + vy[move_mask]) * hp.DT_S
        y[move_mask] = y_unwrap[move_mask]
        if hp.USE_Y_PERIODIC:
            span = hp.Y_MAX - hp.Y_MIN
            y[move_mask] = ((y[move_mask] - hp.Y_MIN) % span) + hp.Y_MIN
        if hp.USE_TRANSLATIONAL_DIFFUSION:
            x[move_mask] += np.sqrt(2.0 * dt0[move_mask] * hp.DT_S) * rng.standard_normal(move_mask.sum())
            y_unwrap[move_mask] += np.sqrt(2.0 * dt0[move_mask] * hp.DT_S) * rng.standard_normal(move_mask.sum())
            y[move_mask] = y_unwrap[move_mask]
            if hp.USE_Y_PERIODIC:
                span = hp.Y_MAX - hp.Y_MIN
                y[move_mask] = ((y[move_mask] - hp.Y_MIN) % span) + hp.Y_MIN
        newly_interface = (~reached_interface) & (x >= hp.X_INTERFACE)
        newly_target = (~reached_target) & (x >= hp.X_TARGET)
        t_first_interface[newly_interface] = t
        t_first_target[newly_target] = t
        reached_interface |= (x >= hp.X_INTERFACE)
        reached_target |= (x >= hp.X_TARGET)
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
        attempted_ab = alive & (x_prev < hp.REG_A_END) & (x >= hp.REG_A_END)
        gate_fail_ab = attempted_ab & (run_streak < hp.RUN_REQ_AB_S)
        if hp.USE_PROB_PASS:
            p_ab = pass_prob(gamma_eff, Ib_local, hp.PASS_P0_AB)
            rand_fail_ab = attempted_ab & (rng.random(hp.N) > p_ab)
        else:
            rand_fail_ab = np.zeros(hp.N, dtype=bool)
        blocked_ab = gate_fail_ab | rand_fail_ab
        if np.any(blocked_ab):
            x[blocked_ab] = hp.REG_A_END - hp.X_INT_EPS
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
            idx_bc = np.where(blocked_bc)[0]
            backstep = rng.uniform(hp.BC_FAIL_BACKSTEP_MIN_UM, hp.BC_FAIL_BACKSTEP_MAX_UM, size=idx_bc.size)
            x_new = np.minimum(x[idx_bc] - backstep, hp.REG_B_END - hp.X_INT_EPS)
            x[idx_bc] = np.maximum(hp.REG_B_END - hp.DELTA_INT_BC, x_new)
            if hp.USE_REFLECT_INTERFACE:
                theta[idx_bc] = np.pi - theta[idx_bc]
        under = alive & (x < hp.X_MIN)
        if np.any(under):
            if hp.USE_REFLECT_LEFT_WALL:
                x[under] = 2 * hp.X_MIN - x[under]
                theta[under] = np.pi - theta[under]
            else:
                x[under] = hp.X_MIN
        in_int_now = alive & (is_int_ab(x) | is_int_bc(x))
        frac_in_interface[k] = float(in_int_now.sum()) / float(hp.N)
        if np.any(in_int_now):
            gamma_int_mean[k] = float(np.mean(gamma_eff[in_int_now]))
        mean_x[k] = x[alive].mean() if np.any(alive) else hp.X_MAX
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
    window_steps = max(1, int(hp.WINDOW_S / hp.DT_S))
    dir_smooth = moving_average(dir_t, window_steps)
    mean_x_smooth = moving_average(mean_x, window_steps)
    frac_interface_smooth = moving_average(frac_interface, window_steps)
    cross_flux_int_smooth = moving_average(cross_flux_int, window_steps)
    frac_target_smooth = moving_average(frac_target, window_steps)
    cross_flux_smooth = moving_average(cross_flux, window_steps)
    msd_smooth = moving_average(msd, window_steps)
    time_axis = (np.arange(steps) + 1) * hp.DT_S
    if save_outputs and run_dir is not None:
        cols = np.column_stack([
            time_axis,
            dir_t, dir_smooth,
            mean_x, mean_x_smooth,
            frac_interface, frac_interface_smooth,
            cross_flux_int, cross_flux_int_smooth,
            frac_target, frac_target_smooth,
            cross_flux, cross_flux_smooth,
            msd, msd_smooth
        ])
        header = (
            "t_s,dir_raw,dir_smoothed,mean_x_raw,mean_x_smoothed,"
            "frac_int_raw,frac_int_smoothed,cross_flux_int_raw,cross_flux_int_smoothed,"
            "frac_target_raw,frac_target_smoothed,cross_flux_raw,cross_flux_smoothed,"
            "msd_raw,msd_smoothed"
        )
        np.savetxt(run_dir / "metrics_timeseries.csv", cols, delimiter=",", header=header, comments="")
    forward_disp = float((x - x0).mean())
    directionality_end = float(dir_t[-1])
    light_dose = float(np.nansum(u_t) * hp.DT_S) if hp.LIGHT_MAPPING == "chey" else float("nan")
    frac_target_end = float(reached_target.mean())
    penetration_rate = float(reached_exit.mean())
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
        "light_dose_s": light_dose,
        "LIGHT_MODE": hp.LIGHT_MODE,
        "LIGHT_MAPPING": hp.LIGHT_MAPPING,
        "USE_CHEMOTAXIS": hp.USE_CHEMOTAXIS,
        "USE_MUCUS_PROFILE": hp.USE_MUCUS_PROFILE,
        "USE_MUCUS_PHYS": hp.USE_MUCUS_PHYS,
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
        "first_passage_exit_mean_s": float(np.nanmean(t_first_exit)) if np.any(np.isfinite(t_first_exit)) else float("nan"),
        "GAMMA_MULT": hp.GAMMA_MULT,
    }
    if save_outputs and run_dir is not None:
        (run_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if make_plots:
        fig_tag = f"mode={hp.LIGHT_MODE}, mapping={hp.LIGHT_MAPPING}"
        fig1, ax = plt.subplots(figsize=(16, 9))
        ax.set_facecolor("#f8fafc")
        for i in range(traj_x.shape[0]):
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
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(11.5, 7.2),
            sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        ax1.plot(time_axis, frac_target, color="#2563eb", alpha=0.22, linewidth=1.0, label="Target Fraction (Raw)")
        ax1.plot(time_axis, frac_target_smooth, color="#2563eb", linewidth=2.0, label=f"Target Fraction (Smoothed {hp.WINDOW_S:.1f}s)")
        ax1.plot(time_axis, frac_interface, color="#16a34a", alpha=0.22, linewidth=1.0, label="Interface Fraction (Raw)")
        ax1.plot(time_axis, frac_interface_smooth, color="#16a34a", linewidth=2.0, label=f"Interface Fraction (Smoothed {hp.WINDOW_S:.1f}s)")
        ax1.set_ylabel("fraction")
        ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.25)
        ax1.legend(loc="upper left", ncol=2, fontsize=9)
        ax1.set_title(f"Target Arrival Metrics ({fig_tag})")
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
    return summary
def _hp_copy(base: HyperParams) -> HyperParams:
    return HyperParams(**asdict(base))
def run_shear_sweep(base_hp: HyperParams) -> None:
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
                hp.LIGHT_MAPPING = base_hp.LIGHT_MAPPING
                hp.GAMMA_MULT = float(gm)
                hp.SEED = int(base_hp.SEED + 1000 * (modes.index(mode) + 1) + 37 * r)
                summ = simulate(hp, make_plots=False, save_outputs=bool(base_hp.SWEEP_SAVE_EACH_RUN))
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
if __name__ == "__main__":
    if HP.ENABLE_SHEAR_SWEEP:
        run_shear_sweep(HP)
    else:
        simulate(HP)
