from pathlib import Path
import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_FILE = PROJECT_ROOT / "data" / "external" / "Phoenix-2014T.test"
TEST_FILE = Path(os.environ.get("PHOENIX_TEST_FILE", str(DEFAULT_TEST_FILE))).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "generated" / "keyposes" / "sample_3258"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAME = "test/22September_2010_Wednesday_tagesschau-3258"
TARGET_SUFFIX = "22September_2010_Wednesday_tagesschau-3258"
NUMBER_OF_SNAPSHOTS = 6
CLIP_LEN = 400

FACE_POINTS = [
    23, 26, 29, 33, 36, 39, 41, 43, 46, 48, 53, 56, 59,
    62, 65, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81,
]
UPPER_BODY_POINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LEFT_HAND_POINTS = list(range(91, 112))
RIGHT_HAND_POINTS = list(range(112, 133))
MSKA_VIS_POINTS = sorted(
    set(UPPER_BODY_POINTS + FACE_POINTS + LEFT_HAND_POINTS + RIGHT_HAND_POINTS)
)

BODY_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
]
HAND_LOCAL_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
LEFT_HAND_EDGES = [(91 + start, 91 + end) for start, end in HAND_LOCAL_EDGES]
RIGHT_HAND_EDGES = [(112 + start, 112 + end) for start, end in HAND_LOCAL_EDGES]
WRIST_CONNECTIONS = [(9, 91), (10, 112)]
ALL_EDGES = BODY_EDGES + LEFT_HAND_EDGES + RIGHT_HAND_EDGES + WRIST_CONNECTIONS


if not TEST_FILE.exists():
    raise FileNotFoundError(
        "Cannot find the PHOENIX test pickle. Set the PHOENIX_TEST_FILE "
        f"environment variable or place the file at {DEFAULT_TEST_FILE}. "
        f"Resolved path: {TEST_FILE}"
    )

print(f"Loading:\n{TEST_FILE}\n")
with open(TEST_FILE, "rb") as file:
    raw_data = pickle.load(file)
print(f"Loaded {len(raw_data)} test samples.")

sample = None
sample_key = None
for key, candidate in raw_data.items():
    candidate_name = str(candidate.get("name", ""))
    if candidate_name == TARGET_NAME:
        sample = candidate
        sample_key = key
        break

if sample is None:
    matches = []
    for key, candidate in raw_data.items():
        candidate_name = str(candidate.get("name", ""))
        if candidate_name.endswith(TARGET_SUFFIX):
            matches.append((key, candidate))

    if len(matches) == 1:
        sample_key, sample = matches[0]
    elif len(matches) > 1:
        raise RuntimeError(f"Multiple samples matched {TARGET_SUFFIX}")

if sample is None:
    raise KeyError(f"Could not locate Sample 3258. Target name: {TARGET_NAME}")

print("\nFound target sample")
print("-" * 80)
print(f"Dictionary key: {sample_key}")
print(f"Name: {sample['name']}")
print(f"Gloss: {sample.get('gloss', '')}")
print(f"num_frames: {sample.get('num_frames', '')}")

keypoints = sample["keypoint"]
if torch.is_tensor(keypoints):
    keypoints = keypoints.detach().cpu().numpy()
else:
    keypoints = np.asarray(keypoints)

print(f"Raw keypoint shape: {keypoints.shape}")
if keypoints.ndim != 3:
    raise ValueError(f"Expected a 3-D keypoint tensor, but received {keypoints.shape}")
if keypoints.shape[1] < 133:
    raise ValueError(f"Expected at least 133 keypoints, but received shape {keypoints.shape}")
if keypoints.shape[2] < 2:
    raise ValueError("Keypoint tensor does not contain x/y coordinates.")

number_of_raw_frames = keypoints.shape[0]
if number_of_raw_frames <= CLIP_LEN:
    model_frame_indices = np.arange(number_of_raw_frames)
else:
    start = (number_of_raw_frames - CLIP_LEN) // 2
    model_frame_indices = np.arange(start, start + CLIP_LEN)

valid_length = len(model_frame_indices)
valid_length -= valid_length % 4
model_frame_indices = model_frame_indices[:valid_length]
if valid_length <= 0:
    raise ValueError("No valid frames available.")

print(f"Model-valid input frames: {valid_length}")
snapshot_positions = np.rint(
    np.linspace(0, valid_length - 1, NUMBER_OF_SNAPSHOTS)
).astype(int)
snapshot_frame_indices = model_frame_indices[snapshot_positions]
print(f"Selected raw frame indices: {snapshot_frame_indices.tolist()}")


def point_is_valid(point):
    x = float(point[0])
    y = float(point[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    return not (abs(x) < 1e-8 and abs(y) < 1e-8)


all_x = []
all_y = []
for frame_index in snapshot_frame_indices:
    frame = keypoints[frame_index]
    for point_index in MSKA_VIS_POINTS:
        point = frame[point_index]
        if point_is_valid(point):
            all_x.append(float(point[0]))
            all_y.append(float(point[1]))

if len(all_x) < 10 or len(all_y) < 10:
    raise ValueError("Too few valid keypoints to create the visualisation.")

all_x = np.asarray(all_x)
all_y = np.asarray(all_y)
x_min = np.percentile(all_x, 1)
x_max = np.percentile(all_x, 99)
y_min = np.percentile(all_y, 1)
y_max = np.percentile(all_y, 99)
x_range = max(x_max - x_min, 1.0)
y_range = max(y_max - y_min, 1.0)
COMMON_XLIM = (x_min - x_range * 0.10, x_max + x_range * 0.10)
COMMON_YLIM = (y_min - y_range * 0.10, y_max + y_range * 0.10)

print(f"Shared crop x = {COMMON_XLIM}")
print(f"Shared crop y = {COMMON_YLIM}")


def draw_pose(axis, frame):
    for start_index, end_index in ALL_EDGES:
        start_point = frame[start_index]
        end_point = frame[end_index]
        if not (point_is_valid(start_point) and point_is_valid(end_point)):
            continue
        axis.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            linewidth=1.7,
            solid_capstyle="round",
            zorder=1,
        )

    body_xy = [frame[index][:2] for index in UPPER_BODY_POINTS if point_is_valid(frame[index])]
    if body_xy:
        body_xy = np.asarray(body_xy)
        axis.scatter(body_xy[:, 0], body_xy[:, 1], s=15, zorder=3)

    hand_xy = [
        frame[index][:2]
        for index in LEFT_HAND_POINTS + RIGHT_HAND_POINTS
        if point_is_valid(frame[index])
    ]
    if hand_xy:
        hand_xy = np.asarray(hand_xy)
        axis.scatter(hand_xy[:, 0], hand_xy[:, 1], s=9, zorder=3)

    face_xy = [frame[index][:2] for index in FACE_POINTS if point_is_valid(frame[index])]
    if face_xy:
        face_xy = np.asarray(face_xy)
        axis.scatter(face_xy[:, 0], face_xy[:, 1], s=5, alpha=0.70, zorder=2)

    axis.set_xlim(COMMON_XLIM)
    axis.set_ylim(COMMON_YLIM[1], COMMON_YLIM[0])
    axis.set_aspect("equal")
    axis.axis("off")


for sequence_number, frame_index in enumerate(snapshot_frame_indices, start=1):
    frame = keypoints[frame_index]
    fig, ax = plt.subplots(figsize=(2.3, 3.0))
    draw_pose(ax, frame)

    svg_path = OUTPUT_DIR / f"sample3258_pose_{sequence_number:02d}_frame{frame_index}.svg"
    png_path = OUTPUT_DIR / f"sample3258_pose_{sequence_number:02d}_frame{frame_index}.png"
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)

fig, axes = plt.subplots(1, NUMBER_OF_SNAPSHOTS, figsize=(13.8, 3.0))
for axis, frame_index in zip(axes, snapshot_frame_indices):
    frame = keypoints[frame_index]
    draw_pose(axis, frame)
    axis.text(
        0.5,
        -0.025,
        f"f={frame_index}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

fig.tight_layout(w_pad=0.10)
strip_png = OUTPUT_DIR / "sample3258_keypose_strip.png"
strip_pdf = OUTPUT_DIR / "sample3258_keypose_strip.pdf"
strip_svg = OUTPUT_DIR / "sample3258_keypose_strip.svg"
fig.savefig(strip_png, dpi=300, bbox_inches="tight", transparent=True)
fig.savefig(strip_pdf, bbox_inches="tight", transparent=True)
fig.savefig(strip_svg, bbox_inches="tight", transparent=True)
plt.show()
plt.close(fig)

metadata_file = OUTPUT_DIR / "sample3258_keypose_metadata.csv"
with open(metadata_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["snapshot_number", "raw_frame_index", "model_valid_position"])
    for snapshot_number, (raw_frame_index, model_position) in enumerate(
        zip(snapshot_frame_indices, snapshot_positions),
        start=1,
    ):
        writer.writerow([snapshot_number, int(raw_frame_index), int(model_position)])

print("\nExport complete")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Combined PNG: {strip_png}")
print(f"Combined PDF: {strip_pdf}")
print(f"Combined SVG: {strip_svg}")
print(f"Metadata: {metadata_file}")
