from __future__ import annotations
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
os.environ.setdefault("MPLBACKEND", "Agg")
from PIL import Image, ImageTk
import merged_abm_full as model
APP_NAME = "光黏领航（LightMucusPilot）微纳生物机器人黏膜给药仿真分析软件 V1.0"
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
def entry_field(label: str, key: str, default: object, *, value_type: str = "string") -> dict[str, object]:
    return {
        "kind": "entry",
        "label": label,
        "key": key,
        "default": default,
        "value_type": value_type,
    }
def check_field(label: str, key: str, default: bool) -> dict[str, object]:
    return {
        "kind": "check",
        "label": label,
        "key": key,
        "default": default,
        "value_type": "bool",
    }
def combo_field(label: str, key: str, options: tuple[str, ...], *, default: object | None = None, value_type: str = "string") -> dict[str, object]:
    return {
        "kind": "combo",
        "label": label,
        "key": key,
        "default": options[0] if default is None else default,
        "options": options,
        "value_type": value_type,
    }
FORM_SECTIONS = (
    (
        "仿真规模",
        (
            entry_field("随机种子", "SEED", 7, value_type="int"),
            entry_field("细菌数量 N", "N", 2000, value_type="int"),
            entry_field("总时长 T (s)", "T_S", 180.0, value_type="float"),
            entry_field("时间步 dt (s)", "DT_S", 0.05, value_type="float"),
            entry_field("轨迹显示数量", "N_PLOT", 120, value_type="int"),
        ),
    ),
    (
        "物理尺寸",
        (
            check_field("自动同步黏膜层厚度", "USE_MUCUS_LAYER_THICKNESS_CONTROL", True),
            entry_field("A区终点 REG_A_END (um)", "REG_A_END", 200.0, value_type="float"),
            entry_field("B层厚度 (um)", "MUCUS_B_THICKNESS_UM", 200.0, value_type="float"),
            entry_field("C层厚度 (um)", "MUCUS_C_THICKNESS_UM", 200.0, value_type="float"),
            entry_field("底部 Y_MIN (um)", "Y_MIN", -8000.0, value_type="float"),
            entry_field("顶部 Y_MAX (um)", "Y_MAX", 0.0, value_type="float"),
            entry_field("AB界面层半宽 (um)", "DELTA_INT_AB", 12.0, value_type="float"),
            entry_field("BC界面层半宽 (um)", "DELTA_INT_BC", 12.0, value_type="float"),
            entry_field("目标位置 X_TARGET (um)", "X_TARGET", 500.0, value_type="float"),
            entry_field("出口位置 X_MAX (um)", "X_MAX", 600.0, value_type="float"),
        ),
    ),
    (
        "细菌运动",
        (
            entry_field("A区速度 v0_A", "V0_A", 20.0, value_type="float"),
            entry_field("B区速度 v0_B", "V0_B", 14.0, value_type="float"),
            entry_field("C区速度 v0_C", "V0_C", 9.0, value_type="float"),
            entry_field("A区旋转扩散 Dr_A", "DR_A", 0.35, value_type="float"),
            entry_field("B区旋转扩散 Dr_B", "DR_B", 0.30, value_type="float"),
            entry_field("C区旋转扩散 Dr_C", "DR_C", 0.18, value_type="float"),
            check_field("启用平动扩散", "USE_TRANSLATIONAL_DIFFUSION", False),
            entry_field("A区平动扩散 Dt_A", "DT_A", 0.2, value_type="float"),
            entry_field("B区平动扩散 Dt_B", "DT_B", 0.12, value_type="float"),
            entry_field("C区平动扩散 Dt_C", "DT_C", 0.08, value_type="float"),
            entry_field("基础 tumble 速率", "LAMBDA_BASE", 0.45, value_type="float"),
        ),
    ),
    (
        "黏膜物化",
        (
            check_field("启用黏膜物化场", "USE_MUCUS_PHYS", True),
            entry_field("表层致密度 M_B", "M_B", 0.6, value_type="float"),
            entry_field("深层致密度 M_C", "M_C", 1.0, value_type="float"),
            entry_field("M 过渡宽度", "M_SCALE", 25.0, value_type="float"),
            entry_field("零剪切黏度 MU0_CY", "MU0_CY", 3.0, value_type="float"),
            entry_field("高剪切黏度 MU_INF", "MU_INF", 0.6, value_type="float"),
            entry_field("Carreau 时间常数", "LAM_CY", 0.8, value_type="float"),
            entry_field("Carreau 指数 n", "N_CY", 0.35, value_type="float"),
            entry_field("Yasuda 参数 a", "A_CY", 2.0, value_type="float"),
            entry_field("困陷增益 TRAP_M_GAIN", "TRAP_M_GAIN", 0.8, value_type="float"),
            entry_field("速度衰减 V_M_ALPHA", "V_M_ALPHA", 0.12, value_type="float"),
            entry_field("Dr 衰减 DR_M_ALPHA", "DR_M_ALPHA", 0.15, value_type="float"),
            entry_field("速度黏度幂 V_MU_POWER", "V_MU_POWER", 0.7, value_type="float"),
            entry_field("Dr 黏度幂 DR_MU_POWER", "DR_MU_POWER", 0.25, value_type="float"),
        ),
    ),
    (
        "屏障与困陷",
        (
            check_field("启用困陷", "USE_TRAP", True),
            entry_field("界面穿越要求 AB (s)", "RUN_REQ_AB_S", 0.40, value_type="float"),
            entry_field("界面穿越要求 BC (s)", "RUN_REQ_BC_S", 0.35, value_type="float"),
            entry_field("穿越概率 P0_AB", "PASS_P0_AB", 0.55, value_type="float"),
            entry_field("穿越概率 P0_BC", "PASS_P0_BC", 0.8, value_type="float"),
            entry_field("困陷释放率", "TRAP_RELEASE_RATE", 1.0, value_type="float"),
            entry_field("界面困陷 P_TRAP_AB", "P_TRAP_AB_LAYER", 0.0030, value_type="float"),
            entry_field("深层界面困陷 P_TRAP_BC", "P_TRAP_BC_LAYER", 0.0015, value_type="float"),
            entry_field("临界剪切 GAMMA_CRIT", "GAMMA_CRIT", 2.0, value_type="float"),
            entry_field("剪切倍率 GAMMA_MULT", "GAMMA_MULT", 1.0, value_type="float"),
        ),
    ),
    (
        "光控与映射",
        (
            combo_field("单次运行模式", "LIGHT_MODE", ("blue", "off", "green", "dual")),
            combo_field("映射方式", "LIGHT_MAPPING", ("chey", "direct_dualcolor")),
            check_field("启用门控控制", "USE_GATE_CONTROL", True),
            entry_field("门控开启阈值 A_on", "GATE_A_ON", 0.25, value_type="float"),
            entry_field("门控关闭阈值 A_off", "GATE_A_OFF", 0.10, value_type="float"),
            entry_field("最短保持时间", "GATE_MIN_HOLD_S", 0.25, value_type="float"),
            combo_field("CheY 开环模式", "CHEY_MODE", ("pulse", "continuous", "off")),
            entry_field("CheY 脉冲频率", "CHEY_PULSE_FREQ_HZ", 1.0, value_type="float"),
            entry_field("CheY 占空比", "CHEY_PULSE_DUTY", 0.50, value_type="float"),
        ),
    ),
)
IMAGE_ORDER = [
    "01_sample_trajectories.png",
    "02_directionality.png",
    "03_mean_x_position.png",
    "03b_msd.png",
    "04_target_arrival_metrics.png",
    "05_u_of_t.png",
    "06_xt_density_heatmap.png",
    "07_xy_occupancy_heatmap.png",
    "08_state_fractions.png",
    "09_mechanism_diagnostics.png",
    "10_first_passage_histograms.png",
    "11_heading_polar_by_region.png",
]
def iter_fields() -> list[dict[str, object]]:
    fields: list[dict[str, object]] = []
    for _, specs in FORM_SECTIONS:
        fields.extend(specs)
    return fields
def resource_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent
def runtime_root() -> Path:
    env_override = os.environ.get("MERGED_ABM_STUDIO_HOME")
    if env_override:
        return ensure_dir(Path(env_override).expanduser())
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "MergedABMStudio"
    elif platform.system() == "Windows":
        base = Path.home() / "AppData" / "Local" / "MergedABMStudio"
    else:
        base = Path.home() / ".merged-abm-studio"
    try:
        return ensure_dir(base)
    except PermissionError:
        return ensure_dir(resource_root() / "merged_abm_runtime")
RUNTIME_ROOT = runtime_root()
OUTPUT_ROOT = ensure_dir(RUNTIME_ROOT / "outputs")
def build_run_dir(kind: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return ensure_dir(OUTPUT_ROOT / kind / stamp)
def open_in_file_manager(path: Path) -> None:
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    elif platform.system() == "Windows":
        subprocess.Popen(["explorer", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
def metric_text(value: object, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "N/A"
        return f"{value:.{digits}f}"
    return str(value)
def format_percent(value: object) -> str:
    if not isinstance(value, (int, float)) or value != value:
        return "N/A"
    return f"{float(value) * 100.0:.2f}%"
def format_signed_percent_delta(off_value: object, blue_value: object) -> str:
    if not isinstance(off_value, (int, float)) or not isinstance(blue_value, (int, float)):
        return "N/A"
    if off_value != off_value or blue_value != blue_value:
        return "N/A"
    delta = (float(blue_value) - float(off_value)) * 100.0
    return f"{delta:+.2f}%"
def format_um(value: object) -> str:
    if not isinstance(value, (int, float)) or value != value:
        return "N/A"
    return f"{float(value):.1f} um"
def format_signed_um_delta(off_value: object, blue_value: object) -> str:
    if not isinstance(off_value, (int, float)) or not isinstance(blue_value, (int, float)):
        return "N/A"
    if off_value != off_value or blue_value != blue_value:
        return "N/A"
    return f"{float(blue_value) - float(off_value):+.1f} um"
def format_seconds(value: object) -> str:
    if not isinstance(value, (int, float)) or value != value:
        return "N/A"
    return f"{float(value):.1f} s"
def format_signed_seconds_delta(off_value: object, blue_value: object) -> str:
    if not isinstance(off_value, (int, float)) or not isinstance(blue_value, (int, float)):
        return "N/A"
    if off_value != off_value or blue_value != blue_value:
        return "N/A"
    return f"{float(blue_value) - float(off_value):+.1f} s"
def clone_default_hp() -> model.HyperParams:
    return model.HyperParams(**asdict(model.HP))
def detect_run_dir(parent: Path) -> Path:
    children = [item for item in parent.iterdir() if item.is_dir()]
    if len(children) != 1:
        run_dirs = sorted(children, key=lambda item: item.stat().st_mtime, reverse=True)
        if not run_dirs:
            raise FileNotFoundError(f"No run directory found under {parent}")
        return run_dirs[0]
    return children[0]
def list_output_images(run_dir: Path) -> list[Path]:
    found = {path.name: path for path in run_dir.glob("*.png")}
    ordered = [found[name] for name in IMAGE_ORDER if name in found]
    extras = sorted(path for name, path in found.items() if name not in IMAGE_ORDER)
    return ordered + extras
def run_single_case(hp: model.HyperParams, output_root: Path, *, make_plots: bool = True) -> tuple[dict[str, float], Path]:
    hp.OUTPUT_DIR = str(output_root)
    hp.SHOW_PLOTS = False
    summary = model.simulate(hp, make_plots=make_plots, save_outputs=True)
    run_dir = detect_run_dir(output_root)
    return summary, run_dir
def build_comparison_rows(off_summary: dict[str, float], blue_summary: dict[str, float], hp: model.HyperParams, target_x: float, exit_x: float) -> list[tuple[str, str, str, str]]:
    return [
        (
            f"穿透率 (x >= {exit_x:.0f} um)",
            format_percent(off_summary.get("penetration_rate_xmax")),
            format_percent(blue_summary.get("penetration_rate_xmax")),
            format_signed_percent_delta(off_summary.get("penetration_rate_xmax"), blue_summary.get("penetration_rate_xmax")),
        ),
        (
            f"目标到达率 (x >= {target_x:.0f} um)",
            format_percent(off_summary.get("frac_target_end")),
            format_percent(blue_summary.get("frac_target_end")),
            format_signed_percent_delta(off_summary.get("frac_target_end"), blue_summary.get("frac_target_end")),
        ),
        (
            "平均位移 (+x)",
            format_um(off_summary.get("forward_displacement_um_mean")),
            format_um(blue_summary.get("forward_displacement_um_mean")),
            format_signed_um_delta(off_summary.get("forward_displacement_um_mean"), blue_summary.get("forward_displacement_um_mean")),
        ),
        (
            "方向性（最后）",
            metric_text(off_summary.get("directionality_mean_last")),
            metric_text(blue_summary.get("directionality_mean_last")),
            metric_text(
                (
                    float(blue_summary["directionality_mean_last"]) - float(off_summary["directionality_mean_last"])
                    if isinstance(off_summary.get("directionality_mean_last"), (int, float))
                    and isinstance(blue_summary.get("directionality_mean_last"), (int, float))
                    else float("nan")
                )
            ),
        ),
        (
            "困陷率",
            format_percent(off_summary.get("mean_trapped_fraction_last")),
            format_percent(blue_summary.get("mean_trapped_fraction_last")),
            format_signed_percent_delta(off_summary.get("mean_trapped_fraction_last"), blue_summary.get("mean_trapped_fraction_last")),
        ),
        (
            "首次到达界面平均时间",
            format_seconds(off_summary.get("first_passage_interface_mean_s")),
            format_seconds(blue_summary.get("first_passage_interface_mean_s")),
            format_signed_seconds_delta(off_summary.get("first_passage_interface_mean_s"), blue_summary.get("first_passage_interface_mean_s")),
        ),
    ]
class ImagePreview(ttk.Frame):
    def __init__(self, parent: tk.Misc, *, placeholder: str) -> None:
        super().__init__(parent, style="Card.TFrame")
        self.placeholder = placeholder
        self.canvas = tk.Canvas(self, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_resize)
        self._photo: ImageTk.PhotoImage | None = None
        self._source_image: Image.Image | None = None
        self._source_path: Path | None = None
        self._render_retry_count = 0
        self._max_render_retries = 10
        self._show_text(self.placeholder)
    def set_image(self, path: Path) -> None:
        if not path.exists():
            self.clear("图片不存在")
            return
        self._source_path = path
        with Image.open(path) as image:
            self._source_image = image.copy()
        self._render_retry_count = 0
        self._render()
    def clear(self, message: str | None = None) -> None:
        self._source_path = None
        self._source_image = None
        self._photo = None
        self._show_text(self.placeholder if message is None else message)
    def _on_resize(self, _event: object) -> None:
        if self._source_image is not None:
            self._render_retry_count = 0
            self._render()
        else:
            self._show_text(self.placeholder)
    def _render(self) -> None:
        if self._source_image is None:
            self._show_text(self.placeholder)
            return
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        if width <= 4 or height <= 4:
            self._render_retry_count += 1
            if self._render_retry_count < self._max_render_retries:
                self.after(50, self._render)
            else:
                self._show_text("预览区域尺寸过小")
            return
        self._render_retry_count = 0
        target = self._source_image.copy()
        target.thumbnail((max(width - 16, 1), max(height - 16, 1)), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(target)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self._photo, anchor="center")
    def _show_text(self, text: str) -> None:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        self.canvas.delete("all")
        self.canvas.create_text(
            width // 2,
            height // 2,
            text=text,
            fill="#5c6f7c",
            font=("Avenir Next", 14),
            anchor="center",
        )
class ScrollableSidebar(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style="Card.TFrame")
        self.canvas = tk.Canvas(self, bg="#edf3f6", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview, style="Sidebar.Vertical.TScrollbar")
        self.content = ttk.Frame(self.canvas, style="Card.TFrame")
        self._wheel_accumulator = 0.0
        self.content.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind_scroll_events(self.canvas)
        self.bind_scroll_events(self.content)
    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)
    def bind_scroll_events(self, widget: tk.Misc) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel, add="+")
        widget.bind("<Shift-MouseWheel>", self._on_shift_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_linux_scroll_up, add="+")
        widget.bind("<Button-5>", self._on_linux_scroll_down, add="+")
        widget.bind("<Prior>", lambda _event: self.canvas.yview_scroll(-1, "pages"), add="+")
        widget.bind("<Next>", lambda _event: self.canvas.yview_scroll(1, "pages"), add="+")
        widget.bind("<Home>", lambda _event: self.canvas.yview_moveto(0.0), add="+")
        widget.bind("<End>", lambda _event: self.canvas.yview_moveto(1.0), add="+")
        widget.bind("<Up>", lambda _event: self.canvas.yview_scroll(-2, "units"), add="+")
        widget.bind("<Down>", lambda _event: self.canvas.yview_scroll(2, "units"), add="+")
    def _vertical_units(self, event: tk.Event) -> int:
        system = platform.system()
        if system == "Darwin":
            self._wheel_accumulator += -float(event.delta) / 3.0
            step = int(self._wheel_accumulator)
            self._wheel_accumulator -= step
            if step == 0:
                return 0
            return max(-10, min(10, step))
        self._wheel_accumulator = 0.0
        delta = int(event.delta / 120)
        if delta == 0:
            return 0
        return -delta
    def _on_mousewheel(self, event: tk.Event) -> None:
        step = self._vertical_units(event)
        if step != 0:
            self.canvas.yview_scroll(step, "units")
        return "break"
    def _on_shift_mousewheel(self, event: tk.Event) -> None:
        step = self._vertical_units(event)
        if step != 0:
            self.canvas.xview_scroll(step, "units")
        return "break"
    def _on_linux_scroll_up(self, _event: object) -> None:
        self.canvas.yview_scroll(-1, "units")
        return "break"
    def _on_linux_scroll_down(self, _event: object) -> None:
        self.canvas.yview_scroll(1, "units")
        return "break"
class DesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self._is_destroyed = False
        self.title(APP_NAME)
        self.geometry("1620x960")
        self.minsize(1380, 860)
        self.configure(bg="#edf3f6")
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        self._configure_theme()
        self.vars: dict[str, tk.Variable] = {}
        self.task_queue: queue.Queue[tuple[str, dict]] = queue.Queue()
        self.worker_busy = False
        self.status_var = tk.StringVar(value="就绪")
        self.output_dir_var = tk.StringVar(value=str(OUTPUT_ROOT))
        self.single_metric_vars = {
            "penetration": tk.StringVar(value="N/A"),
            "target": tk.StringVar(value="N/A"),
            "disp": tk.StringVar(value="N/A"),
            "dir": tk.StringVar(value="N/A"),
            "trapped": tk.StringVar(value="N/A"),
            "interface_time": tk.StringVar(value="N/A"),
        }
        self.last_single_dir: Path | None = None
        self.last_compare_root: Path | None = None
        self.last_compare_off_dir: Path | None = None
        self.last_compare_blue_dir: Path | None = None
        self.single_images: list[Path] = []
        self.compare_off_images: dict[str, Path] = {}
        self.compare_blue_images: dict[str, Path] = {}
        self._build_ui()
        self.load_defaults()
        self.refresh_config_preview()
        self.after(250, self._poll_task_queue)
    def destroy(self) -> None:
        self._is_destroyed = True
        super().destroy()
    def _configure_theme(self) -> None:
        self.style.configure(".", font=("SF Pro Display", 12))
        self.style.configure("TFrame", background="#edf3f6")
        self.style.configure("Card.TFrame", background="#ffffff")
        self.style.configure("TLabelframe", background="#ffffff", borderwidth=1, relief="solid")
        self.style.configure("TLabelframe.Label", background="#ffffff", foreground="#143642", font=("Avenir Next", 12, "bold"))
        self.style.configure("Header.TLabel", background="#0f5c73", foreground="#ffffff", font=("Avenir Next", 28, "bold"))
        self.style.configure("SubHeader.TLabel", background="#0f5c73", foreground="#dceff5", font=("Avenir Next", 16, "bold"))
        self.style.configure("Section.TLabel", background="#edf3f6", foreground="#2a5562", font=("Avenir Next", 12, "bold"))
        self.style.configure("MetricTitle.TLabel", background="#ffffff", foreground="#47626d", font=("Avenir Next", 11))
        self.style.configure("MetricValue.TLabel", background="#ffffff", foreground="#102a43", font=("Avenir Next", 22, "bold"))
        self.style.configure("Primary.TButton", font=("Avenir Next", 12, "bold"), padding=(14, 10))
        self.style.configure("Secondary.TButton", font=("Avenir Next", 12, "bold"), padding=(14, 10))
        self.style.map(
            "Primary.TButton",
            background=[("active", "#127e9b"), ("!disabled", "#0f5c73"), ("disabled", "#8eb9c4")],
            foreground=[("!disabled", "#ffffff"), ("disabled", "#f4fbfd")],
        )
        self.style.configure("Sidebar.Vertical.TScrollbar", arrowsize=18, gripcount=0, troughcolor="#dce6ec", background="#9fb2bf", width=18)
        self.style.configure("Treeview", rowheight=34, font=("Menlo", 16))
        self.style.configure("Treeview.Heading", font=("Avenir Next", 16, "bold"))
        self.style.configure("AppTop.TNotebook", background="#edf3f6", borderwidth=0, tabmargins=(0, 0, 0, 0))
        self.style.configure(
            "AppTop.TNotebook.Tab",
            font=("Avenir Next", 13, "bold"),
            padding=(18, 12),
            width=12,
            focuscolor="none",
        )
        self.style.map(
            "AppTop.TNotebook.Tab",
            padding=[("selected", (18, 12)), ("!selected", (18, 12))],
            expand=[("selected", (0, 0, 0, 0)), ("!selected", (0, 0, 0, 0))],
            background=[("selected", "#ffffff"), ("!selected", "#e5ebef")],
            foreground=[("selected", "#102a43"), ("!selected", "#102a43")],
            lightcolor=[("selected", "#ffffff"), ("!selected", "#e5ebef")],
            bordercolor=[("selected", "#a8b7c2"), ("!selected", "#a8b7c2")],
        )
        self.style.configure("AppSub.TNotebook", background="#edf3f6", borderwidth=0, tabmargins=(0, 0, 0, 0))
        self.style.configure(
            "AppSub.TNotebook.Tab",
            font=("Avenir Next", 12, "bold"),
            padding=(14, 10),
            width=10,
            focuscolor="none",
        )
        self.style.map(
            "AppSub.TNotebook.Tab",
            padding=[("selected", (14, 10)), ("!selected", (14, 10))],
            expand=[("selected", (0, 0, 0, 0)), ("!selected", (0, 0, 0, 0))],
            background=[("selected", "#ffffff"), ("!selected", "#e7edf1")],
            foreground=[("selected", "#102a43"), ("!selected", "#102a43")],
            lightcolor=[("selected", "#ffffff"), ("!selected", "#e7edf1")],
            bordercolor=[("selected", "#a8b7c2"), ("!selected", "#a8b7c2")],
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg="#0f5c73", padx=26, pady=20)
        header.pack(fill="x")
        title_box = tk.Frame(header, bg="#0f5c73")
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text=APP_NAME, style="Header.TLabel").pack(anchor="w")
        status_box = tk.Frame(header, bg="#0f5c73")
        status_box.pack(side="right", anchor="e")
        tk.Label(status_box, textvariable=self.status_var, bg="#0f5c73", fg="#ffffff", font=("Avenir Next", 12, "bold")).pack(anchor="e")
        toolbar = tk.Frame(self, bg="#edf3f6", padx=24, pady=14)
        toolbar.pack(fill="x")
        self.run_single_btn = ttk.Button(toolbar, text="运行单次仿真", style="Primary.TButton", command=self.run_single)
        self.run_single_btn.pack(side="left")
        self.run_compare_btn = ttk.Button(toolbar, text="运行 Off / Blue 对照", style="Primary.TButton", command=self.run_compare)
        self.run_compare_btn.pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="打开输出目录", style="Secondary.TButton", command=lambda: open_in_file_manager(OUTPUT_ROOT)).pack(side="left", padx=(12, 0))
        ttk.Button(toolbar, text="刷新配置预览", style="Secondary.TButton", command=self.refresh_config_preview).pack(side="left", padx=(12, 0))
        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        sidebar = ttk.Frame(body, style="Card.TFrame", padding=10)
        content = ttk.Frame(body)
        body.add(sidebar, weight=0)
        body.add(content, weight=1)
        self._build_sidebar(sidebar)
        self._build_content(content)
    def _build_sidebar(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, style="Card.TFrame")
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="参数控制台", style="Section.TLabel").pack(anchor="w")
        ttk.Button(top, text="恢复脚本默认参数", command=self.load_defaults).pack(fill="x", pady=(8, 0))
        sidebar_scroll = ScrollableSidebar(parent)
        sidebar_scroll.pack(fill="both", expand=True)
        self.sidebar_scroll = sidebar_scroll
        for title, specs in FORM_SECTIONS:
            section = self._section(sidebar_scroll.content, title)
            sidebar_scroll.bind_scroll_events(section)
            for spec in specs:
                self._build_field(section, spec)
    def _build_content(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent, style="AppTop.TNotebook", takefocus=False)
        notebook.pack(fill="both", expand=True)
        notebook.bind("<ButtonRelease-1>", lambda event: self._select_tab_from_event(notebook, event))
        self.main_notebook = notebook
        self.results_tab = ttk.Frame(notebook, padding=18)
        self.config_tab = ttk.Frame(notebook, padding=18)
        notebook.add(self.results_tab, text="结果输出")
        notebook.add(self.config_tab, text="配置预览")
        self._build_results_tab()
        self._build_config_tab()
    def _build_results_tab(self) -> None:
        self.single_tab = ttk.Frame(self.results_tab, padding=12)
        self.compare_tab = ttk.Frame(self.results_tab, padding=12)
        self._build_single_tab()
        self._build_compare_tab()
        self._show_results_view("single")
    def _show_results_view(self, mode: str) -> None:
        self.single_tab.pack_forget()
        self.compare_tab.pack_forget()
        if mode == "compare":
            self.compare_tab.pack(fill="both", expand=True)
        else:
            self.single_tab.pack(fill="both", expand=True)
        self.main_notebook.select(self.results_tab)
    def _select_tab_from_event(self, notebook: ttk.Notebook, event: tk.Event) -> None:
        try:
            index = notebook.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return
        notebook.select(index)
    def _build_single_tab(self) -> None:
        metrics_grid = ttk.Frame(self.single_tab)
        metrics_grid.pack(fill="x", pady=(0, 14))
        specs = [
            ("穿透率", "penetration"),
            ("目标到达率", "target"),
            ("平均位移", "disp"),
            ("方向性", "dir"),
            ("困陷率", "trapped"),
            ("界面首次到达时间", "interface_time"),
        ]
        num_rows = (len(specs) + 2) // 3
        for  r  in range(num_rows):
            metrics_grid.grid_rowconfigure(r, weight=1)
        for idx, (title, key) in enumerate(specs):
            card = ttk.Frame(metrics_grid, style="Card.TFrame", padding=16)
            row = idx // 3
            col = idx % 3
            card.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0), pady=(0 if row == 0 else 10, 0))
            metrics_grid.grid_columnconfigure(col, weight=1)
            ttk.Label(card, text=title, style="MetricTitle.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.single_metric_vars[key], style="MetricValue.TLabel").pack(anchor="w", pady=(8, 0))
        meta_row = ttk.Frame(self.single_tab)
        meta_row.pack(fill="x", pady=(0, 14))
        self.single_output_label = ttk.Label(meta_row, text="暂无输出目录")
        self.single_output_label.pack(side="left")
        ttk.Button(meta_row, text="打开本次输出", command=self.open_last_single_dir).pack(side="right")
        body = ttk.Panedwindow(self.single_tab, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Card.TFrame", padding=10)
        right = ttk.Frame(body, style="Card.TFrame", padding=10)
        body.add(left, weight=0)
        body.add(right, weight=1)
        ttk.Label(left, text="输出图片列表", style="Section.TLabel").pack(anchor="w")
        self.single_image_list = tk.Listbox(left, height=18, font=("Menlo", 10))
        self.single_image_list.pack(fill="both", expand=True, pady=(8, 0))
        self.single_image_list.bind("<<ListboxSelect>>", self._on_single_image_select)
        self.single_preview = ImagePreview(right, placeholder="等待结果")
        self.single_preview.pack(fill="both", expand=True)
    def _build_compare_tab(self) -> None:
        top = ttk.Frame(self.compare_tab)
        top.pack(fill="x", pady=(0, 14))
        self.compare_output_label = ttk.Label(top, text="暂无对照输出")
        self.compare_output_label.pack(side="left")
        ttk.Button(top, text="打开对照输出目录", command=self.open_last_compare_root).pack(side="right")
        table_frame = ttk.LabelFrame(self.compare_tab, text="关键指标对照", padding=10)
        table_frame.pack(fill="x", pady=(0, 24))
        self.compare_table = ttk.Treeview(table_frame, columns=["metric", "off", "blue", "delta"], show="headings", height=7)
        for col, title, width in [
            ("metric", "指标", 360),
            ("off", "off", 160),
            ("blue", "blue", 160),
            ("delta", "变化", 160),
        ]:
            self.compare_table.heading(col, text=title)
            self.compare_table.column(col, width=width, anchor="center")
        self.compare_table.pack(fill="x", expand=True)
        selector = ttk.Frame(self.compare_tab)
        selector.pack(fill="x", pady=(0, 10))
        ttk.Label(selector, text="对比图片").pack(side="left")
        self.compare_image_var = tk.StringVar(value="")
        self.compare_image_combo = ttk.Combobox(selector, textvariable=self.compare_image_var, state="readonly")
        self.compare_image_combo.pack(side="left", fill="x", expand=True, padx=(8, 0))
        self.compare_image_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_compare_preview())
        preview_row = ttk.Frame(self.compare_tab)
        preview_row.pack(fill="both", expand=True)
        off_panel = ttk.LabelFrame(preview_row, text="off", padding=8)
        blue_panel = ttk.LabelFrame(preview_row, text="blue", padding=8)
        off_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        blue_panel.pack(side="left", fill="both", expand=True)
        self.compare_off_preview = ImagePreview(off_panel, placeholder="等待结果")
        self.compare_blue_preview = ImagePreview(blue_panel, placeholder="等待结果")
        self.compare_off_preview.pack(fill="both", expand=True)
        self.compare_blue_preview.pack(fill="both", expand=True)
    def _build_config_tab(self) -> None:
        top = ttk.Frame(self.config_tab)
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="当前 HyperParams JSON", style="Section.TLabel").pack(side="left")
        ttk.Button(top, text="复制到剪贴板", command=self.copy_config).pack(side="right")
        self.config_text = tk.Text(self.config_tab, wrap="none", font=("Menlo", 11), bg="#0f172a", fg="#d9f4f1", insertbackground="#ffffff")
        self.config_text.pack(fill="both", expand=True)
    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill="x", pady=(0, 10))
        return frame
    def _build_field(self, parent: ttk.LabelFrame, spec: dict[str, object]) -> None:
        kind = str(spec["kind"])
        label = str(spec["label"])
        key = str(spec["key"])
        default = spec["default"]
        if kind == "entry":
            self._entry(parent, label, key, str(default))
        elif kind == "check":
            self._check(parent, label, key, bool(default))
        elif kind == "combo":
            self._combo(parent, label, key, [str(item) for item in spec["options"]], str(default))
        else:
            raise ValueError(f"Unknown field kind: {kind}")
    def _entry(self, parent: ttk.LabelFrame, label: str, key: str, default: str) -> None:
        var = self.vars.setdefault(key, tk.StringVar(value=default))
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        label_widget = ttk.Label(row, text=label)
        label_widget.pack(side="left")
        entry_widget = ttk.Entry(row, textvariable=var, width=18)
        entry_widget.pack(side="right")
        self.sidebar_scroll.bind_scroll_events(row)
        self.sidebar_scroll.bind_scroll_events(label_widget)
        self.sidebar_scroll.bind_scroll_events(entry_widget)
    def _check(self, parent: ttk.LabelFrame, label: str, key: str, default: bool) -> None:
        var = self.vars.setdefault(key, tk.BooleanVar(value=default))
        check = ttk.Checkbutton(parent, text=label, variable=var)
        check.pack(anchor="w", pady=2)
        self.sidebar_scroll.bind_scroll_events(check)
    def _combo(self, parent: ttk.LabelFrame, label: str, key: str, values: list[str], default: str) -> None:
        var = self.vars.setdefault(key, tk.StringVar(value=default))
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        label_widget = ttk.Label(row, text=label)
        label_widget.pack(side="left")
        combo_widget = ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=16)
        combo_widget.pack(side="right")
        self.sidebar_scroll.bind_scroll_events(row)
        self.sidebar_scroll.bind_scroll_events(label_widget)
        self.sidebar_scroll.bind_scroll_events(combo_widget)
    def _two_col_table(self, parent: ttk.LabelFrame) -> ttk.Treeview:
        tree = ttk.Treeview(parent, columns=["metric", "value"], show="headings", height=10)
        tree.heading("metric", text="指标")
        tree.heading("value", text="值")
        tree.column("metric", width=220, anchor="w")
        tree.column("value", width=140, anchor="center")
        tree.pack(fill="both", expand=True)
        return tree
    def load_defaults(self) -> None:
        hp = clone_default_hp()
        for spec in iter_fields():
            key = str(spec["key"])
            default = spec["default"]
            value = getattr(hp, key, default) if hasattr(hp, key) else default
            self._set_var(key, value)
        self.refresh_config_preview()
        self.status_var.set("已恢复脚本默认参数")
    def _set_var(self, key: str, value: object) -> None:
        var = self.vars.get(key)
        if var is None:
            if isinstance(value, bool):
                self.vars[key] = tk.BooleanVar(value=value)
            else:
                self.vars[key] = tk.StringVar(value=str(value))
            return
        if isinstance(var, tk.BooleanVar):
            var.set(bool(value))
        else:
            var.set(str(value))
    def _read_var(self, key: str, value_type: str) -> object:
        if value_type == "bool":
            return bool(self.vars[key].get())
        raw = str(self.vars[key].get()).strip()
        try:
            if value_type == "int":
               return int(float(raw))
            if value_type == "float":
               return float(raw)
            return raw
        except ValueError:
            raise ValueError(f"参数 '{key}' 输入无效，期望类型为 {value_type}，实际输入：'{raw}'")
    def collect_hp(self) -> model.HyperParams:
        hp = clone_default_hp()
        for spec in iter_fields():
            key = str(spec["key"])
            if not hasattr(hp, key):
                continue
            setattr(hp, key, self._read_var(key, str(spec["value_type"])))
        model.apply_mucus_layer_thickness(hp)
        hp.SHOW_PLOTS = False
        return hp
    def refresh_config_preview(self) -> None:
        try:
            hp = self.collect_hp()
        except Exception as exc:
            self.status_var.set(f"配置校验失败：{exc}")
            return
        self.config_text.delete("1.0", tk.END)
        self.config_text.insert("1.0", json.dumps(asdict(hp), ensure_ascii=False, indent=2))
    def copy_config(self) -> None:
        self.refresh_config_preview()
        text = self.config_text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("当前配置已复制到剪贴板")
    def set_busy(self, busy: bool, message: str | None = None) -> None:
        self.worker_busy = busy
        state = "disabled" if busy else "normal"
        self.run_single_btn.configure(state=state)
        self.run_compare_btn.configure(state=state)
        if message:
            self.status_var.set(message)
    def run_single(self) -> None:
        if self.worker_busy:
            return
        try:
            hp = self.collect_hp()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        out_root = build_run_dir("single")
        self.set_busy(True, "正在运行单次仿真...")
        threading.Thread(target=self._single_worker, args=(hp, out_root), daemon=True).start()
    def _single_worker(self, hp: model.HyperParams, out_root: Path) -> None:
        try:
            summary, run_dir = run_single_case(hp, out_root, make_plots=True)
            self.task_queue.put(("single_ok", {"summary": summary, "run_dir": run_dir}))
        except Exception as exc:
            traceback.print_exc()
            self.task_queue.put(("error", {"message": f"单次仿真失败：{exc}"}))
    def run_compare(self) -> None:
        if self.worker_busy:
            return
        try:
            hp = self.collect_hp()
        except Exception as exc:
            messagebox.showerror("配置错误", str(exc))
            return
        out_root = build_run_dir("compare")
        self.set_busy(True, "正在运行 Off / Blue 对照...")
        threading.Thread(target=self._compare_worker, args=(hp, out_root), daemon=True).start()
    def _compare_worker(self, hp: model.HyperParams, out_root: Path) -> None:
        try:
            off_root = ensure_dir(out_root / "off")
            blue_root = ensure_dir(out_root / "blue")
            off_hp = model.HyperParams(**asdict(hp))
            blue_hp = model.HyperParams(**asdict(hp))
            off_hp.LIGHT_MODE = "off"
            blue_hp.LIGHT_MODE = "blue"
            model.apply_mucus_layer_thickness(off_hp)
            model.apply_mucus_layer_thickness(blue_hp)
            off_summary, off_dir = run_single_case(off_hp, off_root, make_plots=True)
            blue_summary, blue_dir = run_single_case(blue_hp, blue_root, make_plots=True)
            rows = build_comparison_rows(off_summary, blue_summary, blue_hp, blue_hp.X_TARGET, blue_hp.X_MAX)
            self.task_queue.put(
                (
                    "compare_ok",
                    {
                        "root": out_root,
                        "off_dir": off_dir,
                        "blue_dir": blue_dir,
                        "rows": rows,
                    },
                )
            )
        except Exception as exc:
            traceback.print_exc()
            self.task_queue.put(("error", {"message": f"对照运行失败：{exc}"}))
    def _poll_task_queue(self) -> None:
        if self._is_destroyed:
            return
        try:
            while True:
                kind, payload = self.task_queue.get_nowait()
                if kind == "single_ok":
                    self.last_single_dir = payload["run_dir"]
                    self._show_single_result(payload["summary"], payload["run_dir"])
                    self._show_results_view("single")
                    self.set_busy(False, f"单次仿真完成：{payload['run_dir'].name}")
                elif kind == "compare_ok":
                    self.last_compare_root = payload["root"]
                    self.last_compare_off_dir = payload["off_dir"]
                    self.last_compare_blue_dir = payload["blue_dir"]
                    self._show_compare_result(payload["rows"], payload["off_dir"], payload["blue_dir"])
                    self._show_results_view("compare")
                    self.set_busy(False, f"对照运行完成：{payload['root'].name}")
                elif kind == "error":
                    self.set_busy(False, payload["message"])
                    messagebox.showerror("运行失败", payload["message"])
        except queue.Empty:
            pass
        except tk.TclError:
            return
        self.after(250, self._poll_task_queue)
    def _show_single_result(self, summary: dict[str, float], run_dir: Path) -> None:
        self.single_metric_vars["penetration"].set(format_percent(summary.get("penetration_rate_xmax")))
        self.single_metric_vars["target"].set(format_percent(summary.get("frac_target_end")))
        self.single_metric_vars["disp"].set(format_um(summary.get("forward_displacement_um_mean")))
        self.single_metric_vars["dir"].set(metric_text(summary.get("directionality_mean_last")))
        self.single_metric_vars["trapped"].set(format_percent(summary.get("mean_trapped_fraction_last")))
        self.single_metric_vars["interface_time"].set(format_seconds(summary.get("first_passage_interface_mean_s")))
        self.single_output_label.configure(text=f"输出目录：{run_dir}")
        self.single_images = list_output_images(run_dir)
        self.single_image_list.delete(0, tk.END)
        for image_path in self.single_images:
            self.single_image_list.insert(tk.END, image_path.name)
        if self.single_images:
            self.single_image_list.selection_clear(0, tk.END)
            self.single_image_list.selection_set(0)
            self.single_preview.set_image(self.single_images[0])
        else:
            self.single_preview.clear("没有可预览图片")
    def _show_compare_result(self, rows: list[tuple[str, str, str, str]], off_dir: Path, blue_dir: Path) -> None:
        self.compare_output_label.configure(text=f"对照输出：off={off_dir.name} / blue={blue_dir.name}")
        self.compare_table.delete(*self.compare_table.get_children())
        self.compare_table.configure(height=max(len(rows), 1))
        for row in rows:
            self.compare_table.insert("", tk.END, values=row)
        self.compare_off_images = {path.name: path for path in list_output_images(off_dir)}
        self.compare_blue_images = {path.name: path for path in list_output_images(blue_dir)}
        common = [name for name in IMAGE_ORDER if name in self.compare_off_images and name in self.compare_blue_images]
        if not common:
            common = sorted(set(self.compare_off_images) & set(self.compare_blue_images))
        self.compare_image_combo["values"] = common
        if common:
            self.compare_image_var.set(common[0])
            self._update_compare_preview()
        else:
            self.compare_image_var.set("")
            self.compare_off_preview.clear("没有可预览图片")
            self.compare_blue_preview.clear("没有可预览图片")
    def _fill_two_col_table(self, tree: ttk.Treeview, rows: list[tuple[str, str]]) -> None:
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", tk.END, values=row)
    def _on_single_image_select(self, _event: object) -> None:
        selection = self.single_image_list.curselection()
        if not selection:
            return
        image_path = self.single_images[int(selection[0])]
        self.single_preview.set_image(image_path)
    def _update_compare_preview(self) -> None:
        image_name = self.compare_image_var.get().strip()
        if not image_name:
            return
        off_path = self.compare_off_images.get(image_name)
        blue_path = self.compare_blue_images.get(image_name)
        if off_path is not None:
            self.compare_off_preview.set_image(off_path)
        else:
            self.compare_off_preview.clear("图片不存在")
        if blue_path is not None:
            self.compare_blue_preview.set_image(blue_path)
        else:
            self.compare_blue_preview.clear("图片不存在")
    def open_last_single_dir(self) -> None:
        if self.last_single_dir is not None:
            open_in_file_manager(self.last_single_dir)
    def open_last_compare_root(self) -> None:
        if self.last_compare_root is not None:
            open_in_file_manager(self.last_compare_root)
def main() -> None:
    app = DesktopApp()
    app.mainloop()
if __name__ == "__main__":
    main()