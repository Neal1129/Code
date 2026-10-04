import csv
import json
import os
import sys
import threading
import time
import traceback
import uuid
import matplotlib
from functools import wraps
try:
    from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, session, url_for
except ModuleNotFoundError:
    # The V3.2 computer-vision environment contains OpenCV, while Flask is in
    # the user's base Anaconda environment. Appending keeps the V3.2 numeric
    # libraries first and only supplies the missing web packages.
    sys.path.append(r"C:\Users\515\anaconda3\Lib\site-packages")
    from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from web_engine import run_analysis
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
matplotlib.use("Agg")
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DB_PATH = os.path.join(BASE_DIR, "users.json")
USERS_LOCK = threading.Lock()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}
DEFAULT_CONFIG = {
    "DIFF_THRESHOLD": 8,
    "HIGHPASS_THRESHOLD": 3,
    "MIN_COMPONENT_AREA": 3,
    "MAX_COMPONENT_AREA": 140,
    "MAX_COMPONENT_SIDE": 32,
    "CANNY_LOW": 30,
    "CANNY_HIGH": 100,
    "HOUGH_DP": 1.2,
    "HOUGH_MIN_DIST": 15,
    "HOUGH_PARAM1": 100,
    "HOUGH_PARAM2": 12,
    "MIN_RADIUS": 4,
    "MAX_RADIUS": 15,
    "CENTER_INTENSITY_MAX": 180,
    "DISTANCE_THRESHOLD": 30,
    "DOUBLE_RING_THRESHOLD": 10,
    "TRACK_MAX_MISS": 12,
    "MATCH_DISTANCE": 18,
    "MAX_MATCH_COST": 0.95,
    "MIN_CONFIRM_HITS": 5,
    "TRACK_STITCH_MAX_GAP": 15,
    "TRACK_STITCH_MAX_ERROR": 22.0,
    "TRACK_STITCH_DISTANCE": 30,
    "TRACK_STITCH_MAX_SCORE": 0.80,
    "TRACK_STITCH_AMBIGUITY_MARGIN": 0.08,
    "VISUAL_GAP_FILL": 20,
    "BUBBLE_MIN_FRAMES": 75,
    "BUBBLE_DARK_INTENSITY_MAX": 125,
    "BUBBLE_MIN_DARK_AREA": 220,
    "BUBBLE_SEARCH_RADIUS": 35,
    "BUBBLE_LOCAL_CONTRAST": 18,
    "BUBBLE_CENTER_TOLERANCE": 14,
    "BUBBLE_BACKGROUND_SIGMA": 15,
    "BUBBLE_CLUSTER_RADIUS": 8.0,
    "BUBBLE_FILTER_MARGIN": 8,
    "WORKING_MAX_WIDTH": 1280,
    "WORKING_MAX_HEIGHT": 960,
    "DISPLAY_TRACK_LIMIT": 10,
    "DISPLAY_TRACK_PERCENT": 0.10,
    "DISPLAY_MIN_DIRECTIONALITY": 0.75,
    "DISPLAY_MIN_DURATION_FRAMES": 30,
    "DISPLAY_MIN_DISPLACEMENT": 80.0,
    "DISPLAY_MIN_COVERAGE": 0.40,
    "DISPLAY_MAX_STEP_P90": 30.0,
    "DISPLAY_TRAIL_SECONDS": 3.0,
    "QUALITY_MIN_DURATION": 30,
    "QUALITY_MIN_HITS": 15,
    "QUALITY_MIN_COVERAGE": 0.35,
    "QUALITY_MIN_DISPLACEMENT": 80.0,
    "QUALITY_MAX_STEP_P90": 30.0,
    "STRAIGHT_THRESHOLD": 0.75,
    "SLOW_MIN_DURATION": 12,
    "SLOW_MAX_NET_DISPLACEMENT": 10.0,
    "SLOW_MAX_MEDIAN_SPEED": 0.60,
    "SLOW_MAX_STATIONARY_RADIUS": 7.0,
    "BUBBLE_MIN_DURATION": 10,
    "BUBBLE_MAX_MEDIAN_SPEED": 1.0,
    "BUBBLE_MIN_MEDIAN_BBOX_AREA": 160.0,
    "BUBBLE_MAX_STATIONARY_RADIUS": 12.0,
    "LOW_MOTION_MIN_DURATION": 10,
    "LOW_MOTION_MAX_MEDIAN_SPEED": 0.35,
    "NEAR_STATIC_MAX_NET_DISPLACEMENT": 6.0,
    "NEAR_STATIC_MAX_RADIUS": 4.0,
    "V342_CANDIDATE_MIN_DURATION": 8,
    "V342_CANDIDATE_MAX_MEDIAN_SPEED": 0.90,
    "V342_CANDIDATE_MAX_STATIONARY_RADIUS": 10.0,
    "V342_CANDIDATE_MAX_NET_DISPLACEMENT": 15.0,
    "V342_CLUSTER_DISTANCE": 18.0,
    "V342_CLUSTER_MIN_TRACKS": 3,
    "V342_CLUSTER_MIN_SIMULTANEOUS_TRACKS": 3,
    "V342_CLUSTER_MAX_P90_RADIUS": 20.0,
    "V342_CLUSTER_MIN_SPAN": 20.0,
    "V342_CLUSTER_MAX_SPAN": 60.0,
    "V342_CLUSTER_MIN_TEMPORAL_SPAN": 20,
    "V342_CLUSTER_MAX_MEDIAN_SPEED": 0.75,
    "PIXEL_TO_MICRON": 1.0,
    "speed_min": 5,
    "max_lag": 20,
    "INTERFACE_Y_RATIO": 0.5,
    "CHIP_LENGTH_MICRON": 600,
    "TARGET_THRESHOLD_MICRON": 500,
    "INTERFACE_Y_MICRON": 600,
    "INTERFACE_RATIO": 0.5,
}
CURRENT_CONFIG = dict(DEFAULT_CONFIG)
TASKS = {}
TASKS_LOCK = threading.Lock()
PAUSE_EVENTS = {}
PAUSE_EVENTS_LOCK = threading.Lock()
def ensure_directories():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
def _load_users_db():
    return safe_read_json(USERS_DB_PATH, {"users": {}}).get("users", {})
def _save_users_db(users):
    safe_write_json(USERS_DB_PATH, {"users": users})
def _ensure_default_admin():
    with USERS_LOCK:
        users = _load_users_db()
        if "admin" not in users:
            users["admin"] = {
                "password_hash": generate_password_hash("admin123"),
                "created_at": time.time(),
                "task_ids": [],
            }
            _save_users_db(users)
def get_user(username):
    with USERS_LOCK:
        return _load_users_db().get(username)
def create_user(username, password):
    with USERS_LOCK:
        users = _load_users_db()
        if username in users:
            return False, "用户名已存在"
        users[username] = {
            "password_hash": generate_password_hash(password),
            "created_at": time.time(),
            "task_ids": [],
        }
        _save_users_db(users)
    return True, ""
def verify_user(username, password):
    user = get_user(username)
    if not user:
        return False
    return check_password_hash(user["password_hash"], password)
def link_task_to_user(username, task_id):
    with USERS_LOCK:
        users = _load_users_db()
        user = users.get(username)
        if not user:
            return
        if "task_ids" not in user:
            user["task_ids"] = []
        if task_id not in user["task_ids"]:
            user["task_ids"].append(task_id)
            _save_users_db(users)
def get_user_task_ids(username):
    user = get_user(username)
    if not user:
        return []
    return list(user.get("task_ids", []))
def json_error(message, status_code):
    return jsonify({"error": message}), status_code
def require_task(task_id):
    task = get_task(task_id)
    return task, None if task else json_error("任务不存在", 404)
def allowed_file(filename):
    _, ext = os.path.splitext(filename or "")
    return ext.lower() in ALLOWED_EXTENSIONS
def cast_config_value(default_value, value):
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        return int(float(value))
    if isinstance(default_value, float):
        return float(value)
    return value
def sanitize_config(payload):
    config = dict(CURRENT_CONFIG)
    if not isinstance(payload, dict):
        return config
    for key, default_value in DEFAULT_CONFIG.items():
        if key in payload:
            config[key] = cast_config_value(default_value, payload[key])
    return config
def as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
def as_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default
def sanitize_note(value):
    return "" if value is None else str(value).strip()[:200]
def sanitize_tags(value):
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",") if value.strip() else []
    elif isinstance(value, list):
        raw_items = value
    else:
        return []
    tags = []
    seen = set()
    for item in raw_items:
        tag = str(item).strip()[:30]
        if not tag or tag.lower() in seen:
            continue
        seen.add(tag.lower())
        tags.append(tag)
        if len(tags) >= 8:
            break
    return tags
def relpath_for_ui(path):
    return os.path.relpath(path, BASE_DIR).replace("\\", "/")
def clone_default(default):
    return dict(default) if isinstance(default, dict) else default
def safe_read_json(path, default=None):
    default = {} if default is None else default
    if not os.path.exists(path):
        return clone_default(default)
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (OSError, json.JSONDecodeError):
        return clone_default(default)
def safe_write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
def task_result_dir(task_id):
    return os.path.join(RESULTS_DIR, task_id)
def task_metrics_path(task_id):
    return os.path.join(task_result_dir(task_id), "metrics.json")
def task_trajectory_path(task_id):
    return os.path.join(task_result_dir(task_id), "trajectory_observed.csv")
def task_sort_timestamp(task):
    return as_float(task.get("finished_at"), as_float(task.get("created_at"), 0.0))
def create_task_record(task_id, input_path, original_filename, config_snapshot):
    now = time.time()
    task = {
        "task_id": task_id,
        "status": "queued",
        "message": "已接收文件，等待开始处理",
        "error": "",
        "created_at": now,
        "updated_at": now,
        "started_at": 0.0,
        "finished_at": 0.0,
        "processing_seconds": 0.0,
        "progress": 0,
        "progress_message": "",
        "is_paused": False,
        "note": "",
        "tags": [],
        "starred": False,
        "input_path": input_path,
        "original_filename": original_filename,
        "result_dir": task_result_dir(task_id),
        "config": config_snapshot,
        "result": None,
    }
    with TASKS_LOCK:
        TASKS[task_id] = task
    return task
def update_task(task_id, **changes):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if not task:
            return
        task.update(changes)
        task["updated_at"] = time.time()
def discover_input_path(task_id):
    for ext in sorted(ALLOWED_EXTENSIONS):
        candidate = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return ""
def find_existing_file(directory, candidates):
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return name
    return ""
def compute_average_speed(result_dir):
    csv_path = os.path.join(result_dir, "trajectory_observed.csv")
    if not os.path.exists(csv_path):
        return 0.0
    speed_sum = 0.0
    speed_count = 0
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                speed = as_float(row.get("speed"), None)
                if speed is None:
                    continue
                speed_sum += speed
                speed_count += 1
    except OSError:
        return 0.0
    return speed_sum / speed_count if speed_count else 0.0
def normalize_metrics(metrics, result_dir):
    normalized = dict(metrics) if isinstance(metrics, dict) else {}
    for key in ["penetration", "trapped", "first_time", "target_rate", "avg_displacement", "avg_directionality"]:
        normalized.setdefault(key, 0.0)
    normalized["avg_speed"] = as_float(normalized.get("avg_speed", compute_average_speed(result_dir)), 0.0)
    return normalized
def build_task_info(task, metrics):
    metrics_info = metrics.get("task_info") if isinstance(metrics.get("task_info"), dict) else {}
    created_at = as_float(task.get("created_at"), as_float(metrics_info.get("created_at"), 0.0))
    started_at = as_float(task.get("started_at"), as_float(metrics_info.get("started_at"), 0.0))
    finished_at = as_float(task.get("finished_at"), as_float(metrics_info.get("finished_at"), as_float(task.get("updated_at"), created_at)))
    processing_seconds = as_float(task.get("processing_seconds"), as_float(metrics_info.get("processing_seconds"), max(0.0, finished_at - started_at) if started_at else 0.0))
    original_filename = task.get("original_filename") or metrics_info.get("original_filename") or "未知原文件名"
    return {
        "task_id": task["task_id"],
        "status": task.get("status", "done"),
        "original_filename": original_filename,
        "created_at": created_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "processing_seconds": processing_seconds,
        "note": sanitize_note(task.get("note", metrics_info.get("note", ""))),
        "tags": sanitize_tags(task.get("tags", metrics_info.get("tags", []))),
        "starred": as_bool(task.get("starred", metrics_info.get("starred", False))),
    }
def load_task_from_disk(task_id):
    metrics_path = task_metrics_path(task_id)
    if not os.path.exists(metrics_path):
        return None
    result_dir = task_result_dir(task_id)
    metrics = normalize_metrics(safe_read_json(metrics_path, {}), result_dir)
    info = metrics.get("task_info") if isinstance(metrics.get("task_info"), dict) else {}
    finished_at = as_float(info.get("finished_at"), os.path.getmtime(metrics_path))
    created_at = as_float(info.get("created_at"), finished_at)
    started_at = as_float(info.get("started_at"), 0.0)
    if started_at > 0:
        processing_seconds = as_float(info.get("processing_seconds"), max(0.0, finished_at - started_at))
    else:
        processing_seconds = as_float(info.get("processing_seconds"), 0.0)
    return {
        "task_id": task_id,
        "status": "done",
        "message": "处理完成",
        "error": "",
        "created_at": created_at,
        "updated_at": finished_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "processing_seconds": processing_seconds,
        "input_path": discover_input_path(task_id),
        "original_filename": info.get("original_filename", ""),
        "note": sanitize_note(info.get("note", "")),
        "tags": sanitize_tags(info.get("tags", [])),
        "starred": as_bool(info.get("starred", False)),
        "result_dir": result_dir,
        "config": {},
        "result": None,
    }
def get_task(task_id):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
        if task:
            return dict(task)
    loaded = load_task_from_disk(task_id)
    if loaded:
        with TASKS_LOCK:
            TASKS.setdefault(task_id, loaded)
        return dict(loaded)
    return None
def persist_task_metadata(task_id):
    task = get_task(task_id)
    if not task:
        return {}
    metrics_path = task_metrics_path(task_id)
    if not os.path.exists(metrics_path):
        return {}
    metrics = normalize_metrics(safe_read_json(metrics_path, {}), task["result_dir"])
    task_info = metrics.get("task_info") if isinstance(metrics.get("task_info"), dict) else {}
    task_info.update({
        "task_id": task_id,
        "original_filename": task.get("original_filename") or task_info.get("original_filename") or "未知原文件名",
        "created_at": as_float(task.get("created_at"), as_float(task_info.get("created_at"), time.time())),
        "started_at": as_float(task.get("started_at"), as_float(task_info.get("started_at"), 0.0)),
        "finished_at": as_float(task.get("finished_at"), as_float(task_info.get("finished_at"), time.time())),
        "processing_seconds": as_float(task.get("processing_seconds"), as_float(task_info.get("processing_seconds"), 0.0)),
        "note": sanitize_note(task.get("note", task_info.get("note", ""))),
        "tags": sanitize_tags(task.get("tags", task_info.get("tags", []))),
        "starred": as_bool(task.get("starred", task_info.get("starred", False))),
    })
    metrics["task_info"] = task_info
    safe_write_json(metrics_path, metrics)
    return metrics
def file_url_for_task(task_id, result_dir, filename):
    if not filename:
        return ""
    full_path = os.path.join(result_dir, filename)
    return f"/tasks/{task_id}/files/{filename}" if os.path.exists(full_path) else ""
def build_metric_summary(metrics):
    return {
        "penetration_rate": as_float(metrics.get("penetration"), 0.0),
        "target_arrival_rate": as_float(metrics.get("target_rate"), 0.0),
        "average_speed": as_float(metrics.get("avg_speed"), 0.0),
    }
def build_result_payload(task_id):
    task = get_task(task_id)
    if not task:
        raise KeyError(task_id)
    result_dir = task["result_dir"]
    metrics_path = task_metrics_path(task_id)
    metrics = normalize_metrics(safe_read_json(metrics_path, {}), result_dir)
    task_info = build_task_info(task, metrics)
    metrics["task_info"] = {key: task_info[key] for key in ["task_id", "original_filename", "created_at", "started_at", "finished_at", "processing_seconds", "note", "tags", "starred"]}
    # WebM/VP8 is generated specifically for Chrome/Edge/Firefox playback.
    # The MP4V file remains available as the downloadable archival copy.
    video_filename = find_existing_file(result_dir, ["tracking_clustered.webm", "tracking_clustered.mp4", "tracking_clustered.avi"])
    trajectory_path = task_trajectory_path(task_id)
    return {
        "task_id": task_id,
        "task": task_info,
        "summary": build_metric_summary(metrics),
        "metrics": metrics,
        "video_url": file_url_for_task(task_id, result_dir, video_filename),
        "charts": {
            "speed": file_url_for_task(task_id, result_dir, "speed_hist.png"),
            "displacement": file_url_for_task(task_id, result_dir, "displacement_hist.png"),
            "directionality": file_url_for_task(task_id, result_dir, "directionality_hist.png"),
            "trap": file_url_for_task(task_id, result_dir, "trap_hist.png"),
            "fpt": file_url_for_task(task_id, result_dir, "fpt_hist.png"),
            "rates": file_url_for_task(task_id, result_dir, "rates_bar.png"),
            "classification": file_url_for_task(task_id, result_dir, "classification.png"),
        },
        "paths": {
            "video": relpath_for_ui(os.path.join(result_dir, video_filename)) if video_filename else "",
            "json": relpath_for_ui(metrics_path) if os.path.exists(metrics_path) else "",
            "csv": relpath_for_ui(trajectory_path) if os.path.exists(trajectory_path) else "",
            "track_metrics_csv": relpath_for_ui(os.path.join(result_dir, "track_metrics.csv")) if os.path.exists(os.path.join(result_dir, "track_metrics.csv")) else "",
            "charts_dir": relpath_for_ui(result_dir),
        },
    }
def build_history_entry(task):
    payload = build_result_payload(task["task_id"])
    task_info = payload["task"]
    return {
        "serial_number": 0,
        "task_id": task["task_id"],
        "original_filename": task_info["original_filename"],
        "created_at": task_info["created_at"],
        "finished_at": task_info["finished_at"],
        "processing_seconds": task_info["processing_seconds"],
        "note": task_info["note"],
        "tags": task_info["tags"],
        "starred": task_info["starred"],
        "summary": payload["summary"],
    }
def list_history_tasks(username=None):
    ensure_directories()
    if username:
        user_task_ids = set(get_user_task_ids(username))
    else:
        user_task_ids = None
    task_ids = set()
    with TASKS_LOCK:
        task_ids.update(TASKS.keys())
    for name in os.listdir(RESULTS_DIR):
        result_dir = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(result_dir) and os.path.exists(os.path.join(result_dir, "metrics.json")):
            task_ids.add(name)
    if user_task_ids is not None:
        task_ids = task_ids & user_task_ids
    history = []
    for task_id in task_ids:
        task = get_task(task_id)
        if task and task.get("status") == "done":
            history.append(build_history_entry(task))
    history.sort(key=lambda item: as_float(item.get("finished_at"), as_float(item.get("created_at"), 0.0)), reverse=True)
    for index, item in enumerate(history, start=1):
        item["serial_number"] = index
    return history
def process_task(task_id):
    task = get_task(task_id)
    if not task:
        return
    pause_event = threading.Event()
    pause_event.set()
    with PAUSE_EVENTS_LOCK:
        PAUSE_EVENTS[task_id] = pause_event
    started_at = time.time()
    update_task(task_id, status="processing", message="正在分析视频，请稍候", error="", started_at=started_at, finished_at=0.0, processing_seconds=0.0, progress=0, progress_message="", is_paused=False)
    def progress_callback(pct, message):
        update_task(task_id, progress=int(pct), progress_message=message)
    try:
        result = run_analysis(task["input_path"], dict(task["config"]), task_id, progress_callback=progress_callback, pause_event=pause_event)
        if not isinstance(result, dict):
            raise RuntimeError("算法返回结果格式不正确")
        if result.get("error"):
            raise RuntimeError(result["error"])
        finished_at = time.time()
        update_task(task_id, finished_at=finished_at, processing_seconds=max(0.0, finished_at - started_at), progress=100, progress_message="处理完成")
        persist_task_metadata(task_id)
        update_task(task_id, status="done", message="处理完成", result=build_result_payload(task_id))
    except Exception:
        finished_at = time.time()
        update_task(task_id, status="failed", message="处理失败", error=traceback.format_exc(), finished_at=finished_at, processing_seconds=max(0.0, finished_at - started_at))
    finally:
        with PAUSE_EVENTS_LOCK:
            PAUSE_EVENTS.pop(task_id, None)
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.accept_mimetypes.best == "application/json":
                return json_error("请先登录", 401)
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated
@app.route("/")
def splash():
    if session.get("logged_in"):
        return redirect(url_for("main_page"))
    return render_template("splash.html")
@app.route("/login")
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("main_page"))
    return render_template("login.html")
@app.route("/api/login", methods=["POST"])
def do_login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if not username or not password:
        return jsonify({"ok": False, "error": "请输入用户名和密码"})
    if not verify_user(username, password):
        return jsonify({"ok": False, "error": "用户名或密码错误"})
    session["logged_in"] = True
    session["username"] = username
    return jsonify({"ok": True})
@app.route("/register")
def register_page():
    if session.get("logged_in"):
        return redirect(url_for("main_page"))
    return render_template("register.html")
@app.route("/api/register", methods=["POST"])
def do_register():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    confirm = str(payload.get("confirm", ""))
    if not username or not password:
        return jsonify({"ok": False, "error": "请输入用户名和密码"})
    if len(username) < 2 or len(username) > 20:
        return jsonify({"ok": False, "error": "用户名长度需为 2-20 个字符"})
    if len(password) < 6:
        return jsonify({"ok": False, "error": "密码至少 6 个字符"})
    if password != confirm:
        return jsonify({"ok": False, "error": "两次密码输入不一致"})
    ok, err = create_user(username, password)
    if not ok:
        return jsonify({"ok": False, "error": err})
    session["logged_in"] = True
    session["username"] = username
    return jsonify({"ok": True})
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("splash"))
@app.route("/main")
@login_required
def main_page():
    return render_template("index.html", username=session.get("username", ""))
@app.route("/upload", methods=["POST"])
@login_required
def upload():
    ensure_directories()
    if "video" not in request.files:
        return json_error("请选择视频文件", 400)
    file = request.files["video"]
    if not file or file.filename == "":
        return json_error("请选择视频文件", 400)
    if not allowed_file(file.filename):
        return json_error("仅支持 mp4/avi/mov/mkv/mpeg/mpg 视频文件", 400)
    task_id = uuid.uuid4().hex
    original_name = (os.path.basename(file.filename) if file.filename else "").strip() or f"{task_id}.mp4"
    safe_name = secure_filename(original_name) or f"{task_id}.mp4"
    _, ext = os.path.splitext(safe_name)
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}{(ext or '.mp4').lower()}")
    file.save(input_path)
    create_task_record(task_id, input_path, original_name, dict(CURRENT_CONFIG))
    username = session.get("username", "")
    if username:
        link_task_to_user(username, task_id)
    threading.Thread(target=process_task, args=(task_id,), daemon=True).start()
    return jsonify({"status": "processing", "task_id": task_id, "message": "上传成功，后台已开始处理"})
@app.route("/tasks/history")
@login_required
def task_history():
    return jsonify({"tasks": list_history_tasks(session.get("username"))})
@app.route("/tasks/<task_id>/status")
@login_required
def task_status(task_id):
    task, error_response = require_task(task_id)
    if error_response:
        return error_response
    return jsonify({
        "task_id": task_id,
        "status": task["status"],
        "message": task["message"],
        "error": task["error"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "started_at": task.get("started_at", 0.0),
        "finished_at": task.get("finished_at", 0.0),
        "processing_seconds": task.get("processing_seconds", 0.0),
        "original_filename": task.get("original_filename", ""),
        "progress": task.get("progress", 0),
        "progress_message": task.get("progress_message", ""),
        "is_paused": task.get("is_paused", False),
    })
@app.route("/tasks/<task_id>/pause", methods=["POST"])
@login_required
def task_pause(task_id):
    task, error_response = require_task(task_id)
    if error_response:
        return error_response
    if task["status"] != "processing":
        return json_error("任务当前不在处理中", 409)
    with PAUSE_EVENTS_LOCK:
        event = PAUSE_EVENTS.get(task_id)
        if event:
            event.clear()
    update_task(task_id, is_paused=True, message="已暂停处理")
    return jsonify({"status": "paused"})
@app.route("/tasks/<task_id>/resume", methods=["POST"])
@login_required
def task_resume(task_id):
    task, error_response = require_task(task_id)
    if error_response:
        return error_response
    with PAUSE_EVENTS_LOCK:
        event = PAUSE_EVENTS.get(task_id)
        if event:
            event.set()
    update_task(task_id, is_paused=False, message="正在分析视频，请稍候")
    return jsonify({"status": "resumed"})
@app.route("/tasks/<task_id>/meta", methods=["POST"])
@login_required
def task_meta_update(task_id):
    task, error_response = require_task(task_id)
    if error_response:
        return error_response
    payload = request.get_json(silent=True) or {}
    note = sanitize_note(payload.get("note", task.get("note", "")))
    tags = sanitize_tags(payload.get("tags", task.get("tags", [])))
    starred = as_bool(payload.get("starred", task.get("starred", False)))
    update_task(task_id, note=note, tags=tags, starred=starred)
    if task.get("status") == "done":
        persist_task_metadata(task_id)
        update_task(task_id, result=build_result_payload(task_id))
    updated_task = get_task(task_id) or {}
    return jsonify({"task_id": task_id, "note": updated_task.get("note", ""), "tags": updated_task.get("tags", []), "starred": updated_task.get("starred", False)})
@app.route("/tasks/<task_id>/result")
@login_required
def task_result(task_id):
    task, error_response = require_task(task_id)
    if error_response:
        return error_response
    if task["status"] != "done":
        return json_error("任务尚未完成", 409)
    return jsonify(task["result"] or build_result_payload(task_id))
@app.route("/tasks/<task_id>/files/<path:filename>")
@login_required
def task_file(task_id, filename):
    task, _ = require_task(task_id)
    if not task:
        abort(404)
    full_path = os.path.join(task["result_dir"], filename)
    if not os.path.exists(full_path):
        abort(404)
    return send_from_directory(task["result_dir"], filename)
@app.route("/update_params", methods=["POST"])
@login_required
def update_params():
    global CURRENT_CONFIG
    CURRENT_CONFIG = sanitize_config(request.get_json(silent=True) or {})
    return jsonify({"status": "ok", "params": CURRENT_CONFIG})
if __name__ == "__main__":
    ensure_directories()
    _ensure_default_admin()
    app.run(debug=False, use_reloader=False, threaded=True, port=5005)
