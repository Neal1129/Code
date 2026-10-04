"""V3 preview: motion-gated bacteria detection for the first 10 seconds."""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(r"D:\大创\软件")
NAME = "MG-空载质粒-诱导后-无蓝光"
VIDEO = ROOT / "Microscope-movement-mp4" / f"{NAME}.mp4"
V1_DIR = ROOT / "轨迹优化版_20260813" / "蓝光直线率完整识别_20260824" / NAME
OUT_DIR = ROOT / "轨迹优化版_20260813" / "运动检测V3_扩展召回_前10秒_20260824"
OUT_VIDEO = OUT_DIR / f"{NAME}_V3扩展召回前10秒.mp4"
OUT_CSV = OUT_DIR / f"{NAME}_V3扩展召回轨迹.csv"

FRAME_LIMIT_SECONDS = 10
DIFF_THRESHOLD = 10
HIGHPASS_THRESHOLD = 5
MIN_COMPONENT_AREA = 5
MAX_COMPONENT_AREA = 100
MAX_COMPONENT_SIDE = 26
MATCH_DISTANCE = 15.0
MAX_MISS = 3
MIN_CONFIRM_HITS = 5
TRAIL_FRAMES = 35


def load_bubble_boxes():
    path = V1_DIR / "bubbles.csv"
    if not path.exists():
        return []
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[int(row["bubble_id"])].append([
                float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])
            ])
    return [np.median(np.asarray(values), axis=0).tolist() for values in groups.values()]


def inside_bubble(x, y, boxes, margin=10):
    return any(x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin for x1, y1, x2, y2 in boxes)


def detect_motion_targets(gray, background, bubble_boxes):
    bg_u8 = cv2.convertScaleAbs(background)
    diff = cv2.absdiff(gray, bg_u8)
    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    highpass = cv2.absdiff(gray, smooth)
    mask = ((diff >= DIFF_THRESHOLD) & (highpass >= HIGHPASS_THRESHOLD)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    detections = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if not (MIN_COMPONENT_AREA <= area <= MAX_COMPONENT_AREA):
            continue
        if width > MAX_COMPONENT_SIDE or height > MAX_COMPONENT_SIDE or width < 2 or height < 2:
            continue
        cx, cy = centroids[label]
        if inside_bubble(cx, cy, bubble_boxes):
            continue
        component_values = highpass[labels == label]
        if component_values.size == 0 or float(np.percentile(component_values, 75)) < HIGHPASS_THRESHOLD:
            continue
        pad = 3
        detections.append({
            "center": (float(cx), float(cy)),
            "bbox": (max(0, x - pad), max(0, y - pad), min(gray.shape[1] - 1, x + width + pad), min(gray.shape[0] - 1, y + height + pad)),
        })
    return detections


class MotionTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    def update(self, detections, frame_no):
        active_ids = [track_id for track_id, track in self.tracks.items() if frame_no - track["last_frame"] <= MAX_MISS]
        matched_tracks, matched_detections = set(), set()
        if active_ids and detections:
            costs = np.full((len(active_ids), len(detections)), 1e6, np.float32)
            for row, track_id in enumerate(active_ids):
                tx, ty = self.tracks[track_id]["center"]
                for column, detection in enumerate(detections):
                    dx, dy = detection["center"]
                    distance = float(np.hypot(dx - tx, dy - ty))
                    if distance <= MATCH_DISTANCE:
                        costs[row, column] = distance
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] >= 1e5:
                    continue
                track_id = active_ids[row]
                self._apply(track_id, detections[column], frame_no)
                matched_tracks.add(track_id)
                matched_detections.add(column)
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "center": detection["center"], "bbox": detection["bbox"], "last_frame": frame_no,
                "hits": 1, "history": deque([(frame_no, *detection["center"])])
            }
        stale = [track_id for track_id, track in self.tracks.items() if frame_no - track["last_frame"] > MAX_MISS]
        for track_id in stale:
            del self.tracks[track_id]

    def _apply(self, track_id, detection, frame_no):
        track = self.tracks[track_id]
        track["center"] = detection["center"]
        track["bbox"] = detection["bbox"]
        track["last_frame"] = frame_no
        track["hits"] += 1
        track["history"].append((frame_no, *detection["center"]))
        while track["history"] and track["history"][0][0] < frame_no - TRAIL_FRAMES:
            track["history"].popleft()


def color_for_id(track_id):
    hsv = np.uint8([[[int(track_id * 43 % 180), 215, 235]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_limit = int(round(FRAME_LIMIT_SECONDS * fps))
    writer = cv2.VideoWriter(str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 960))
    bubble_boxes = load_bubble_boxes()
    tracker = MotionTracker()
    background = None
    trajectory_rows = []
    confirmed_ids = set()
    active_counts = []
    frame_no = 0
    while frame_no < frame_limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        frame = cv2.resize(frame, (1280, 960), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if background is None:
            background = gray.astype(np.float32)
            writer.write(frame)
            continue
        detections = detect_motion_targets(gray, background, bubble_boxes)
        # Slow background update: moving objects remain different long enough to be confirmed.
        cv2.accumulateWeighted(gray, background, 0.015)
        tracker.update(detections, frame_no)
        overlay = frame.copy()
        active_count = 0
        for track_id, track in tracker.tracks.items():
            if track["hits"] < MIN_CONFIRM_HITS or frame_no - track["last_frame"] > 1:
                continue
            confirmed_ids.add(track_id)
            active_count += 1
            color = color_for_id(track_id)
            trail = list(track["history"])
            if len(trail) >= 2:
                points = np.asarray([(int(x), int(y)) for _, x, y in trail], np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [points], False, color, 2, cv2.LINE_AA)
            x1, y1, x2, y2 = track["bbox"]
            cx, cy = int(round(track["center"][0])), int(round(track["center"][1]))
            inner_half = 5
            ix1, iy1 = max(0, cx - inner_half), max(0, cy - inner_half)
            ix2, iy2 = min(1280, cx + inner_half + 1), min(960, cy + inner_half + 1)
            overlay[iy1:iy2, ix1:ix2] = frame[iy1:iy2, ix1:ix2]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            trajectory_rows.append([track_id, frame_no, cx, cy, x1, y1, x2, y2])
        active_counts.append(active_count)
        frame = cv2.addWeighted(overlay, 0.90, frame, 0.10, 0)
        cv2.rectangle(frame, (8, 8), (625, 42), (0, 0, 0), -1)
        cv2.putText(frame, f"V3 motion-confirmed | active: {active_count} | frame {frame_no}/{frame_limit}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        csv_writer = csv.writer(handle)
        csv_writer.writerow(["track_id", "frame", "x", "y", "x1", "y1", "x2", "y2"])
        csv_writer.writerows(trajectory_rows)
    print(f"video={OUT_VIDEO}")
    print(f"confirmed_tracks={len(confirmed_ids)}")
    print(f"trajectory_points={len(trajectory_rows)}")
    print(f"mean_active={float(np.mean(active_counts)):.2f}")
    print(f"p90_active={float(np.percentile(active_counts, 90)):.2f}")


if __name__ == "__main__":
    main()
