"""Web-compatible V3.4.2 bacteria detection and tracking engine.

The public ``run_analysis`` contract stays compatible with the original web
application while detection, classification and clustered artifact filtering
follow the accepted V3.4.2 batch pipeline.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import motion_detector as detector


BASE_DIR = Path(__file__).resolve().parent

# Accepted V3.4/V3.4.1/V3.4.2 parameters in 1280 x 960 coordinates.
V342_DIFF_THRESHOLD = 8
V342_HIGHPASS_THRESHOLD = 3
V342_MIN_COMPONENT_AREA = 3
V342_MAX_COMPONENT_AREA = 140
V342_MAX_COMPONENT_SIDE = 32
V342_BUBBLE_MIN_DURATION = 10
V342_BUBBLE_MAX_MEDIAN_SPEED = 1.00
V342_BUBBLE_MIN_MEDIAN_BBOX_AREA = 160.0
V342_BUBBLE_MAX_STATIONARY_RADIUS = 12.0
V342_LOW_MOTION_MIN_DURATION = 10
V342_LOW_MOTION_MAX_MEDIAN_SPEED = 0.35
V342_NEAR_STATIC_MAX_NET_DISPLACEMENT = 6.0
V342_NEAR_STATIC_MAX_RADIUS = 4.0
V342_CANDIDATE_MAX_MEDIAN_SPEED = 0.90
V342_CANDIDATE_MAX_STATIONARY_RADIUS = 10.0
V342_CANDIDATE_MAX_NET_DISPLACEMENT = 15.0
V342_CANDIDATE_MIN_DURATION = 8
V342_CLUSTER_DISTANCE = 18.0
V342_CLUSTER_MIN_TRACKS = 3
V342_CLUSTER_MIN_SIMULTANEOUS_TRACKS = 3
V342_CLUSTER_MAX_P90_RADIUS = 20.0
V342_CLUSTER_MIN_SPAN = 20.0
V342_CLUSTER_MAX_SPAN = 60.0
V342_CLUSTER_MIN_TEMPORAL_SPAN = 20
V342_CLUSTER_MAX_MEDIAN_SPEED = 0.75


def _cfg(config, name, default, cast=float):
    try:
        return cast(config.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


def _wait(pause_event):
    if pause_event is not None:
        pause_event.wait()


def _progress(callback, value, message):
    if callback is not None:
        callback(max(0, min(100, int(value))), message)


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _color(track_id):
    hsv = np.uint8([[[int(track_id * 43 % 180), 215, 235]]])
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


class SpatialPredictiveTracker:
    """V3.2 prediction tracker with sparse, spatial Hungarian assignment."""

    def __init__(self, match_distance=18.0, max_miss=12):
        self.match_distance = float(match_distance)
        self.max_miss = int(max_miss)
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def predicted_center(track, frame_no):
        history = track["history"]
        if len(history) < 2:
            return np.asarray(track["center"], np.float32)
        sample = history[-5:]
        frames = np.asarray([item[0] for item in sample], np.float32)
        xs = np.asarray([item[1] for item in sample], np.float32)
        ys = np.asarray([item[2] for item in sample], np.float32)
        if frames[-1] <= frames[0]:
            return np.asarray(track["center"], np.float32)
        velocity = np.asarray([
            np.polyfit(frames, xs, 1)[0], np.polyfit(frames, ys, 1)[0]
        ], np.float32)
        speed = float(np.linalg.norm(velocity))
        if speed > 8.0:
            velocity *= 8.0 / speed
        return np.asarray([xs[-1], ys[-1]], np.float32) + velocity * (frame_no - frames[-1])

    def _apply(self, track_id, detection, frame_no):
        track = self.tracks[track_id]
        cx, cy = detection["center"]
        track["center"] = detection["center"]
        track["last_frame"] = frame_no
        track["hits"] += 1
        track["history"].append((frame_no, float(cx), float(cy), tuple(detection["bbox"])))

    def update(self, detections, frame_no):
        active_ids = [
            track_id for track_id, track in self.tracks.items()
            if frame_no - track["last_frame"] <= self.max_miss
        ]
        matched = set()
        if active_ids and detections:
            points = np.asarray([item["center"] for item in detections], np.float32)
            tree = cKDTree(points)
            row_edges, col_edges, predictions, gap_by_row = {}, defaultdict(list), {}, {}
            for row, track_id in enumerate(active_ids):
                track = self.tracks[track_id]
                predicted = self.predicted_center(track, frame_no)
                gap = frame_no - track["last_frame"]
                allowed = self.match_distance + min(8.0, gap * 1.5)
                columns = sorted(tree.query_ball_point(predicted, allowed))
                if not columns:
                    continue
                row_edges[row] = columns
                predictions[row] = predicted
                gap_by_row[row] = gap
                for column in columns:
                    col_edges[column].append(row)
            unseen = set(row_edges)
            while unseen:
                seed = min(unseen)
                rows_set, cols_set, stack = set(), set(), [seed]
                while stack:
                    row = stack.pop()
                    if row in rows_set:
                        continue
                    rows_set.add(row)
                    unseen.discard(row)
                    for column in row_edges[row]:
                        if column not in cols_set:
                            cols_set.add(column)
                            stack.extend(col_edges[column])
                rows, columns = sorted(rows_set), sorted(cols_set)
                local_col = {column: index for index, column in enumerate(columns)}
                costs = np.full((len(rows), len(columns)), 1e6, np.float32)
                for local_row, row in enumerate(rows):
                    for column in row_edges[row]:
                        distance = float(np.linalg.norm(points[column] - predictions[row]))
                        costs[local_row, local_col[column]] = distance + gap_by_row[row] * 0.25
                assigned_rows, assigned_cols = linear_sum_assignment(costs)
                for local_row, local_column in zip(assigned_rows, assigned_cols):
                    if costs[local_row, local_column] >= 1e5:
                        continue
                    row, column = rows[local_row], columns[local_column]
                    self._apply(active_ids[row], detections[column], frame_no)
                    matched.add(column)
        for index, detection in enumerate(detections):
            if index in matched:
                continue
            track_id = self.next_id
            self.next_id += 1
            cx, cy = detection["center"]
            self.tracks[track_id] = {
                "center": detection["center"], "last_frame": frame_no, "hits": 1,
                "history": [(frame_no, float(cx), float(cy), tuple(detection["bbox"]))],
            }


def _velocity(history, from_end=True):
    sample = history[-5:] if from_end else history[:5]
    if len(sample) < 2:
        return np.zeros(2, np.float32)
    frames = np.asarray([item[0] for item in sample], np.float32)
    xs = np.asarray([item[1] for item in sample], np.float32)
    ys = np.asarray([item[2] for item in sample], np.float32)
    if frames[-1] <= frames[0]:
        return np.zeros(2, np.float32)
    return np.asarray([np.polyfit(frames, xs, 1)[0], np.polyfit(frames, ys, 1)[0]], np.float32)


def _stitch(tracks, max_gap, max_error):
    starts = defaultdict(list)
    for track_id, track in tracks.items():
        if len(track["history"]) >= 3:
            starts[int(track["history"][0][0])].append(track_id)
    candidates = []
    for previous_id, previous in tracks.items():
        if len(previous["history"]) < 3:
            continue
        end = previous["history"][-1]
        velocity = _velocity(previous["history"], True)
        speed = float(np.linalg.norm(velocity))
        for start_frame in range(int(end[0]) + 1, int(end[0]) + max_gap + 1):
            gap = start_frame - int(end[0])
            predicted = np.asarray([end[1], end[2]], np.float32) + velocity * gap
            allowed = min(max_error, 10.0 + speed * gap * 0.45)
            for next_id in starts.get(start_frame, []):
                following = tracks[next_id]
                first = following["history"][0]
                error = float(np.linalg.norm(np.asarray([first[1], first[2]]) - predicted))
                if error > allowed:
                    continue
                next_velocity = _velocity(following["history"], False)
                next_speed = float(np.linalg.norm(next_velocity))
                penalty = 0.0
                if speed >= 0.5 and next_speed >= 0.5:
                    cosine = float(np.dot(velocity, next_velocity) / (speed * next_speed))
                    if cosine < -0.2:
                        continue
                    penalty = (1.0 - cosine) * 0.15
                score = error / max(allowed, 1.0) + gap / max_gap * 0.20 + penalty
                candidates.append((score, previous_id, next_id))
    candidates.sort()
    by_previous, by_next = defaultdict(list), defaultdict(list)
    for candidate in candidates:
        by_previous[candidate[1]].append(candidate)
        by_next[candidate[2]].append(candidate)
    predecessor, successor = {}, {}
    for score, previous_id, next_id in candidates:
        if previous_id in successor or next_id in predecessor:
            continue
        if len(by_previous[previous_id]) > 1 and by_previous[previous_id][1][0] - score < 0.08:
            continue
        if len(by_next[next_id]) > 1 and by_next[next_id][1][0] - score < 0.08:
            continue
        successor[previous_id], predecessor[next_id] = next_id, previous_id
    root_map = {}
    for track_id in tracks:
        root, seen = track_id, set()
        while root in predecessor and root not in seen:
            seen.add(root)
            root = predecessor[root]
        root_map[track_id] = root
    merged = {}
    for track_id, track in tracks.items():
        root = root_map[track_id]
        merged.setdefault(root, {"center": track["center"], "last_frame": track["last_frame"], "hits": 0, "history": []})
        merged[root]["history"].extend(track["history"])
        merged[root]["hits"] += track["hits"]
        if track["last_frame"] >= merged[root]["last_frame"]:
            merged[root]["center"], merged[root]["last_frame"] = track["center"], track["last_frame"]
    for track in merged.values():
        unique = {item[0]: item for item in track["history"]}
        track["history"] = sorted(unique.values(), key=lambda item: item[0])
    return merged, sum(track_id != root for track_id, root in root_map.items())


def _track_record(track_id, track, config):
    history = track["history"]
    frames = np.asarray([item[0] for item in history], np.float32)
    positions = np.asarray([[item[1], item[2]] for item in history], np.float32)
    duration = int(frames[-1] - frames[0] + 1)
    hits = len(history)
    coverage = hits / max(duration, 1)
    net = float(np.linalg.norm(positions[-1] - positions[0]))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1) if hits >= 2 else np.asarray([])
    gaps = np.maximum(1.0, np.diff(frames)) if hits >= 2 else np.asarray([])
    normalized = steps / gaps if hits >= 2 else np.asarray([])
    path_length = float(np.sum(steps))
    directionality = net / path_length if path_length > 0 else 0.0
    median_speed = float(np.median(normalized)) if normalized.size else 0.0
    step_p90 = float(np.percentile(normalized, 90)) if normalized.size else 0.0
    center = np.median(positions, axis=0)
    radius = float(np.percentile(np.linalg.norm(positions - center, axis=1), 90))
    bbox_areas = np.asarray([
        max(0.0, float(item[3][2] - item[3][0]))
        * max(0.0, float(item[3][3] - item[3][1]))
        for item in history
    ], np.float32)
    median_bbox_area = float(np.median(bbox_areas)) if bbox_areas.size else 0.0
    if hits < _cfg(config, "MIN_CONFIRM_HITS", 5, int):
        classification = "unconfirmed"
    else:
        slow_large_artifact = (
            duration >= _cfg(config, "BUBBLE_MIN_DURATION", V342_BUBBLE_MIN_DURATION, int)
            and median_speed <= _cfg(config, "BUBBLE_MAX_MEDIAN_SPEED", V342_BUBBLE_MAX_MEDIAN_SPEED)
            and median_bbox_area >= _cfg(config, "BUBBLE_MIN_MEDIAN_BBOX_AREA", V342_BUBBLE_MIN_MEDIAN_BBOX_AREA)
            and radius <= _cfg(config, "BUBBLE_MAX_STATIONARY_RADIUS", V342_BUBBLE_MAX_STATIONARY_RADIUS)
        )
        very_low_motion = (
            duration >= _cfg(config, "LOW_MOTION_MIN_DURATION", V342_LOW_MOTION_MIN_DURATION, int)
            and median_speed <= _cfg(config, "LOW_MOTION_MAX_MEDIAN_SPEED", V342_LOW_MOTION_MAX_MEDIAN_SPEED)
        )
        nearly_stationary = (
            duration >= _cfg(config, "LOW_MOTION_MIN_DURATION", V342_LOW_MOTION_MIN_DURATION, int)
            and net <= _cfg(config, "NEAR_STATIC_MAX_NET_DISPLACEMENT", V342_NEAR_STATIC_MAX_NET_DISPLACEMENT)
            and radius <= _cfg(config, "NEAR_STATIC_MAX_RADIUS", V342_NEAR_STATIC_MAX_RADIUS)
        )
        if slow_large_artifact:
            classification = "slow_bubble"
        elif very_low_motion or nearly_stationary:
            classification = "low_motion"
        else:
            classification = "bacterium"
    eligible = (
        classification == "bacterium"
        and duration >= _cfg(config, "QUALITY_MIN_DURATION", 30, int)
        and hits >= _cfg(config, "QUALITY_MIN_HITS", 15, int)
        and coverage >= _cfg(config, "QUALITY_MIN_COVERAGE", 0.35)
        and net >= _cfg(config, "QUALITY_MIN_DISPLACEMENT", 80.0)
        and step_p90 <= _cfg(config, "QUALITY_MAX_STEP_P90", 30.0)
    )
    straight = eligible and directionality >= _cfg(config, "STRAIGHT_THRESHOLD", 0.75)
    return {
        "track_id": track_id, "classification": classification,
        "start_frame": int(frames[0]), "end_frame": int(frames[-1]),
        "duration_frames": duration, "observed_points": hits, "coverage": coverage,
        "net_displacement_px": net, "path_length_px": path_length,
        "directionality": directionality, "median_speed_px_frame": median_speed,
        "step_p90_px_frame": step_p90, "stationary_radius_px": radius,
        "quality_eligible": eligible, "straight": straight,
    }


class _UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _cluster_v342_bubble_regions(records, trajectories, config, width, height):
    """Return V3.4.2 clustered artifact regions and their member track IDs."""
    record_map = {int(row["track_id"]): row for row in records}
    observations = defaultdict(list)
    for row in trajectories:
        observations[int(row["track_id"])].append(row)
    candidates = []
    for track_id, rows in observations.items():
        record = record_map[track_id]
        if record["classification"] == "unconfirmed":
            continue
        if not (
            int(record["duration_frames"]) >= _cfg(config, "V342_CANDIDATE_MIN_DURATION", V342_CANDIDATE_MIN_DURATION, int)
            and float(record["median_speed_px_frame"]) <= _cfg(config, "V342_CANDIDATE_MAX_MEDIAN_SPEED", V342_CANDIDATE_MAX_MEDIAN_SPEED)
            and float(record["stationary_radius_px"]) <= _cfg(config, "V342_CANDIDATE_MAX_STATIONARY_RADIUS", V342_CANDIDATE_MAX_STATIONARY_RADIUS)
            and float(record["net_displacement_px"]) <= _cfg(config, "V342_CANDIDATE_MAX_NET_DISPLACEMENT", V342_CANDIDATE_MAX_NET_DISPLACEMENT)
        ):
            continue
        candidates.append({
            "track_id": track_id,
            "classification": record["classification"],
            "center": np.asarray([
                np.median([float(row["x"]) for row in rows]),
                np.median([float(row["y"]) for row in rows]),
            ], np.float32),
            "median_bbox": [
                float(np.median([float(row[key]) for row in rows]))
                for key in ("x1", "y1", "x2", "y2")
            ],
            "start_frame": int(record["start_frame"]),
            "end_frame": int(record["end_frame"]),
            "median_speed": float(record["median_speed_px_frame"]),
        })
    if not candidates:
        return [], set(), 0
    centers = np.asarray([item["center"] for item in candidates], np.float32)
    union_find = _UnionFind(len(candidates))
    distance = _cfg(config, "V342_CLUSTER_DISTANCE", V342_CLUSTER_DISTANCE)
    for left, right in cKDTree(centers).query_pairs(distance):
        union_find.union(left, right)
    components = defaultdict(list)
    for index in range(len(candidates)):
        components[union_find.find(index)].append(index)
    regions, member_ids, rejected_large = [], set(), 0
    for indexes in components.values():
        if len(indexes) < _cfg(config, "V342_CLUSTER_MIN_TRACKS", V342_CLUSTER_MIN_TRACKS, int):
            continue
        items = [candidates[index] for index in indexes]
        centers_here = np.asarray([item["center"] for item in items])
        center = np.median(centers_here, axis=0)
        p90_radius = float(np.percentile(np.linalg.norm(centers_here - center, axis=1), 90))
        x1 = min(item["median_bbox"][0] for item in items)
        y1 = min(item["median_bbox"][1] for item in items)
        x2 = max(item["median_bbox"][2] for item in items)
        y2 = max(item["median_bbox"][3] for item in items)
        span_width, span_height = x2 - x1, y2 - y1
        start_frame = min(item["start_frame"] for item in items)
        end_frame = max(item["end_frame"] for item in items)
        temporal_span = end_frame - start_frame + 1
        median_speed = float(np.median([item["median_speed"] for item in items]))
        max_simultaneous = max(
            sum(item["start_frame"] <= frame <= item["end_frame"] for item in items)
            for frame in range(start_frame, end_frame + 1)
        )
        has_anchor = any(item["classification"] in {"slow_bubble", "low_motion"} for item in items)
        if (
            span_width > _cfg(config, "V342_CLUSTER_MAX_SPAN", V342_CLUSTER_MAX_SPAN)
            or span_height > _cfg(config, "V342_CLUSTER_MAX_SPAN", V342_CLUSTER_MAX_SPAN)
            or p90_radius > _cfg(config, "V342_CLUSTER_MAX_P90_RADIUS", V342_CLUSTER_MAX_P90_RADIUS)
        ):
            rejected_large += 1
            continue
        if max(span_width, span_height) < _cfg(config, "V342_CLUSTER_MIN_SPAN", V342_CLUSTER_MIN_SPAN):
            continue
        if (
            temporal_span < _cfg(config, "V342_CLUSTER_MIN_TEMPORAL_SPAN", V342_CLUSTER_MIN_TEMPORAL_SPAN, int)
            or median_speed > _cfg(config, "V342_CLUSTER_MAX_MEDIAN_SPEED", V342_CLUSTER_MAX_MEDIAN_SPEED)
            or max_simultaneous < _cfg(config, "V342_CLUSTER_MIN_SIMULTANEOUS_TRACKS", V342_CLUSTER_MIN_SIMULTANEOUS_TRACKS, int)
            or not has_anchor
        ):
            continue
        track_ids = sorted(item["track_id"] for item in items)
        member_ids.update(track_ids)
        regions.append({
            "region_id": len(regions) + 1,
            "x1": max(0.0, x1 - 8.0), "y1": max(0.0, y1 - 8.0),
            "x2": min(float(width - 1), x2 + 8.0), "y2": min(float(height - 1), y2 + 8.0),
            "start_frame": start_frame, "end_frame": end_frame,
            "track_count": len(items), "track_ids": track_ids,
            "median_speed_px_frame": median_speed,
            "p90_center_radius_px": p90_radius,
            "max_simultaneous_tracks": max_simultaneous,
            "source_classifications": dict(Counter(item["classification"] for item in items)),
        })
    return regions, member_ids, rejected_large


def _write_dict_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _font():
    candidates = [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyh.ttc"]
    for candidate in candidates:
        if os.path.exists(candidate):
            return font_manager.FontProperties(fname=candidate)
    return None


def _hist(values, xlabel, title, path, density=False):
    font = _font()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if values:
        bins = min(20, max(5, int(math.sqrt(len(values)) * 2)))
        ax.hist(values, bins=bins, density=density, color="#3B82C4", edgecolor="white", alpha=0.88)
    else:
        ax.text(0.5, 0.5, "暂无合格轨迹数据", ha="center", va="center", transform=ax.transAxes, fontproperties=font)
    ax.set_xlabel(xlabel, fontproperties=font)
    ax.set_ylabel("概率密度" if density else "轨迹数", fontproperties=font)
    ax.set_title(title, fontproperties=font, fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_charts(out_dir, chart_data):
    _hist(chart_data["speeds"], "速度 (μm/s)", "细菌速度分布", out_dir / "speed_hist.png")
    _hist(chart_data["displacements"], "净位移 (μm)", "细菌位移分布", out_dir / "displacement_hist.png")
    _hist(chart_data["directionalities"], "方向性", "细菌方向性分布", out_dir / "directionality_hist.png")
    _hist(chart_data["trap_ratios"], "低速帧占比", "低速占比分布", out_dir / "trap_hist.png")
    _hist(chart_data["fpt"], "首次到达时间 (s)", "首次到达时间分布", out_dir / "fpt_hist.png", density=True)
    font = _font()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["直线率", "穿透率", "目标到达率"]
    values = [chart_data["straight_rate"], chart_data["penetration_rate"], chart_data["target_rate"]]
    bars = ax.bar(labels, values, color=["#2563EB", "#14B8A6", "#F59E0B"])
    ax.set_xticks(range(3), labels, fontproperties=font)
    ax.set_ylim(0, 1)
    ax.set_ylabel("比例", fontproperties=font)
    ax.set_title("迁移效率指标", fontproperties=font, fontsize=14)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "rates_bar.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["直线运动", "非直线运动", "已排除气泡", "未确认目标"]
    counts = chart_data["class_counts"]
    bars = ax.bar(labels, counts, color=["#2563EB", "#93C5FD", "#EF4444", "#9CA3AF"])
    ax.set_xticks(range(4), labels, fontproperties=font, rotation=8)
    ax.set_ylabel("轨迹数", fontproperties=font)
    ax.set_title("行为分类与过滤结果", fontproperties=font, fontsize=14)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts + [1]) * 0.02, str(count), ha="center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "classification.png", dpi=170)
    plt.close(fig)


def _analysis_size(width, height, config):
    max_width = _cfg(config, "WORKING_MAX_WIDTH", 1280, int)
    max_height = _cfg(config, "WORKING_MAX_HEIGHT", 960, int)
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))


def _render(video_path, output_path, tracks, selected_ids, fps, size, total_frames, config, callback, pause_event):
    visual_gap = _cfg(config, "VISUAL_GAP_FILL", 15, int)
    trail_frames = max(2, int(round(_cfg(config, "DISPLAY_TRAIL_SECONDS", 3.0) * fps)))
    per_frame = defaultdict(list)
    for track_id in selected_ids:
        history = tracks[track_id]["history"]
        for index, item in enumerate(history):
            frame_no, x, y, bbox = item
            per_frame[int(frame_no)].append((track_id, float(x), float(y), bbox, True))
            if index + 1 >= len(history):
                continue
            next_item = history[index + 1]
            gap = int(next_item[0] - frame_no)
            if gap <= 1 or gap > visual_gap:
                continue
            width = max(8, int(bbox[2] - bbox[0]))
            height = max(8, int(bbox[3] - bbox[1]))
            for offset in range(1, gap):
                ratio = offset / gap
                ix = x + (next_item[1] - x) * ratio
                iy = y + (next_item[2] - y) * ratio
                ibox = (int(ix - width / 2), int(iy - height / 2), int(ix + width / 2), int(iy + height / 2))
                per_frame[int(frame_no + offset)].append((track_id, ix, iy, ibox, False))
    cap = cv2.VideoCapture(str(video_path))
    mp4_writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    browser_path = output_path.with_suffix(".webm")
    browser_writer = cv2.VideoWriter(str(browser_path), cv2.VideoWriter_fourcc(*"VP80"), fps, size)
    if not mp4_writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建结果视频，请检查视频编码器")
    histories = {track_id: deque(maxlen=trail_frames) for track_id in selected_ids}
    frame_no = 0
    while True:
        _wait(pause_event)
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        active = 0
        for track_id, x, y, bbox, _observed in per_frame.get(frame_no, []):
            active += 1
            color = _color(track_id)
            histories[track_id].append((int(round(x)), int(round(y))))
            points = list(histories[track_id])
            if len(points) >= 2:
                cv2.polylines(frame, [np.asarray(points, np.int32).reshape((-1, 1, 2))], False, color, 2, cv2.LINE_AA)
            x1, y1, x2, y2 = [int(value) for value in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(size[0] - 1, x2), min(size[1] - 1, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            cv2.putText(frame, str(track_id), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (8, 8), (540, 42), (0, 0, 0), -1)
        cv2.putText(frame, f"V3.4.2 | selected: {len(selected_ids)} | active: {active}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        mp4_writer.write(frame)
        if browser_writer.isOpened():
            browser_writer.write(frame)
        if frame_no % 10 == 0:
            _progress(callback, 62 + 30 * frame_no / max(total_frames, 1), f"第二阶段：生成轨迹视频 {frame_no}/{total_frames} 帧")
    cap.release()
    mp4_writer.release()
    if browser_writer.isOpened():
        browser_writer.release()
    elif browser_path.exists():
        browser_path.unlink()


def run_analysis(video_path, config, task_id, progress_callback=None, pause_event=None):
    """Analyze one uploaded video and write web-compatible artifacts."""
    out_dir = BASE_DIR / "results" / str(task_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "无法打开上传的视频"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 960)
    size = _analysis_size(source_width, source_height, config)
    detector.DIFF_THRESHOLD = _cfg(config, "DIFF_THRESHOLD", V342_DIFF_THRESHOLD, int)
    detector.HIGHPASS_THRESHOLD = _cfg(config, "HIGHPASS_THRESHOLD", V342_HIGHPASS_THRESHOLD, int)
    detector.MIN_COMPONENT_AREA = _cfg(config, "MIN_COMPONENT_AREA", V342_MIN_COMPONENT_AREA, int)
    detector.MAX_COMPONENT_AREA = _cfg(config, "MAX_COMPONENT_AREA", V342_MAX_COMPONENT_AREA, int)
    detector.MAX_COMPONENT_SIDE = _cfg(config, "MAX_COMPONENT_SIDE", V342_MAX_COMPONENT_SIDE, int)
    tracker = SpatialPredictiveTracker(
        _cfg(config, "MATCH_DISTANCE", 18.0), _cfg(config, "TRACK_MAX_MISS", 12, int)
    )
    background = None
    frame_no = 0
    _progress(progress_callback, 1, "第一阶段：检测并连接细菌轨迹")
    while True:
        _wait(pause_event)
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if background is None:
            background = gray.astype(np.float32)
            continue
        detections = detector.detect_motion_targets(gray, background, [])
        cv2.accumulateWeighted(gray, background, 0.015)
        tracker.update(detections, frame_no)
        if frame_no % 10 == 0:
            _progress(progress_callback, 2 + 53 * frame_no / max(total_frames, 1), f"第一阶段：轨迹识别 {frame_no}/{total_frames} 帧")
    cap.release()
    total_frames = frame_no
    if total_frames < 2:
        return {"error": "视频帧数不足，无法分析"}
    _progress(progress_callback, 56, "正在连接断开的轨迹片段")
    tracks, stitched = _stitch(
        tracker.tracks,
        _cfg(config, "TRACK_STITCH_MAX_GAP", 15, int),
        _cfg(config, "TRACK_STITCH_MAX_ERROR", 22.0),
    )
    records = [_track_record(track_id, track, config) for track_id, track in tracks.items()]
    fields = list(records[0]) if records else [
        "track_id", "classification", "start_frame", "end_frame", "duration_frames",
        "observed_points", "coverage", "net_displacement_px", "path_length_px",
        "directionality", "median_speed_px_frame", "step_p90_px_frame",
        "stationary_radius_px", "quality_eligible", "straight",
    ]
    _write_dict_csv(out_dir / "track_metrics.csv", records, fields)
    trajectory_rows = []
    for track_id, track in tracks.items():
        for frame, x, y, bbox in track["history"]:
            trajectory_rows.append({
                "track_id": track_id, "frame": frame, "x": x, "y": y,
                "x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3],
            })
    trajectory_rows.sort(key=lambda row: (int(row["frame"]), int(row["track_id"])))
    _write_dict_csv(out_dir / "trajectory_observed.csv", trajectory_rows, ["track_id", "frame", "x", "y", "x1", "y1", "x2", "y2"])
    bubble_regions, clustered_bubble_ids, rejected_large_clusters = _cluster_v342_bubble_regions(
        records, trajectory_rows, config, size[0], size[1]
    )
    for row in records:
        if int(row["track_id"]) in clustered_bubble_ids:
            row["classification"] = "slow_bubble"
            row["quality_eligible"] = False
            row["straight"] = False
    _write_dict_csv(out_dir / "track_metrics.csv", records, fields)
    with (out_dir / "v342_bubble_regions.json").open("w", encoding="utf-8") as handle:
        json.dump(bubble_regions, handle, ensure_ascii=False, indent=2)
    record_map = {row["track_id"]: row for row in records}
    bacteria = [row for row in records if row["classification"] == "bacterium"]
    bubbles = [row for row in records if row["classification"] == "slow_bubble"]
    unconfirmed = [row for row in records if row["classification"] == "unconfirmed"]
    low_motion = [row for row in records if row["classification"] == "low_motion"]
    eligible = [row for row in records if row["quality_eligible"]]
    straight = [row for row in eligible if row["straight"]]
    display_min_directionality = _cfg(config, "DISPLAY_MIN_DIRECTIONALITY", 0.75)
    display_percent = min(1.0, max(0.01, _cfg(config, "DISPLAY_TRACK_PERCENT", 0.10)))
    display_limit = max(1, int(math.ceil(len(eligible) * display_percent))) if eligible else 0
    ranking = sorted(eligible, key=lambda row: (
        row["directionality"], row["coverage"], row["duration_frames"], row["net_displacement_px"]
    ), reverse=True)
    preferred = [row for row in ranking if row["directionality"] >= display_min_directionality]
    selected = preferred[:display_limit]
    if len(selected) < display_limit:
        used = {row["track_id"] for row in selected}
        selected.extend(row for row in ranking if row["track_id"] not in used)
        selected = selected[:display_limit]
    selected_ids = [row["track_id"] for row in selected]
    _progress(progress_callback, 61, f"已筛出 {len(eligible)} 条合格轨迹，视频展示 {len(selected_ids)} 条")
    _render(video_path, out_dir / "tracking_clustered.mp4", tracks, selected_ids, fps, size, total_frames, config, progress_callback, pause_event)
    pixel_to_micron = _cfg(config, "PIXEL_TO_MICRON", 1.0)
    speed_min = _cfg(config, "speed_min", 5.0)
    displacements, directionalities, speeds, trap_ratios, fpt = [], [], [], [], []
    penetration_count = target_count = target_eligible = 0
    interface_y = size[1] * _cfg(config, "INTERFACE_RATIO", 0.5)
    target_y = _cfg(config, "TARGET_THRESHOLD_MICRON", 500.0) / max(pixel_to_micron, 1e-9)
    for row in eligible:
        track = tracks[row["track_id"]]
        history = track["history"]
        frames = np.asarray([item[0] for item in history], np.float32)
        positions = np.asarray([[item[1], item[2]] for item in history], np.float32)
        gaps = np.maximum(1.0, np.diff(frames))
        step_speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / gaps * fps * pixel_to_micron
        displacements.append(row["net_displacement_px"] * pixel_to_micron)
        directionalities.append(row["directionality"])
        speeds.append(float(np.median(step_speeds)) if step_speeds.size else 0.0)
        trap_ratios.append(float(np.mean(step_speeds <= speed_min)) if step_speeds.size else 0.0)
        ys = positions[:, 1]
        if any(ys[index - 1] < interface_y <= ys[index] for index in range(1, len(ys))):
            penetration_count += 1
        if ys[0] < target_y:
            target_eligible += 1
            reached = np.flatnonzero(ys >= target_y)
            if reached.size:
                target_count += 1
                fpt.append(float((frames[reached[0]] - frames[0]) / fps))
    straight_rate = len(straight) / len(eligible) if eligible else 0.0
    penetration_rate = penetration_count / len(eligible) if eligible else 0.0
    target_rate = target_count / target_eligible if target_eligible else 0.0
    chart_data = {
        "speeds": speeds, "displacements": displacements, "directionalities": directionalities,
        "trap_ratios": trap_ratios, "fpt": fpt, "straight_rate": straight_rate,
        "penetration_rate": penetration_rate, "target_rate": target_rate,
        "class_counts": [len(straight), len(eligible) - len(straight), len(bubbles), len(unconfirmed)],
    }
    _progress(progress_callback, 94, "正在生成统计表和图表")
    _save_charts(out_dir, chart_data)
    metrics = {
        "algorithm_version": "V3.4.2 网页版",
        "penetration": straight_rate,
        "straight_rate": straight_rate,
        "migration_penetration_rate": penetration_rate,
        "target_rate": target_rate,
        "trapped": _mean(trap_ratios),
        "first_time": _mean(fpt),
        "avg_displacement": _mean(displacements),
        "avg_directionality": _mean(directionalities),
        "avg_speed": _mean(speeds),
        "straight_tracks": len(straight),
        "quality_eligible_tracks": len(eligible),
        "target_tracks": target_count,
        "target_eligible_tracks": target_eligible,
        "penetration_tracks": penetration_count,
        "bacterium_tracks": len(bacteria),
        "slow_bubble_tracks": len(bubbles),
        "unconfirmed_tracks": len(unconfirmed),
        "low_motion_tracks": len(low_motion),
        "v342_bubble_regions": len(bubble_regions),
        "v342_clustered_bubble_tracks": len(clustered_bubble_ids),
        "v342_rejected_large_clusters": rejected_large_clusters,
        "stitched_fragments": stitched,
        "displayed_tracks": len(selected_ids),
        "display_track_percent": display_percent,
        "displayed_track_ids": selected_ids,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
        "analysis_width": size[0],
        "analysis_height": size[1],
        "note": "气泡和未确认目标不计入细菌统计；轨迹视频仅展示方向性优先的合格轨迹。",
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    _progress(progress_callback, 100, "V3.4.2 分析完成")
    return metrics
