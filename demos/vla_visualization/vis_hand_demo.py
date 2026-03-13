#!/usr/bin/env python
"""Demo: Visualize MANO hand mesh with optional camera poses.

Render modes:
  - opencv: Render hand mesh wireframe/solid overlay on video frames (OpenCV).
            When camera poses are available, also renders a world-space
            wrist trajectory bird's-eye view.
  - aitviewer: 3D visualization with camera poses (aitviewer, requires
               display or xvfb)

Usage:
    # OpenCV overlay (auto-detects camera poses, adds trajectory)
    python vis_hand_demo.py \
        --data_path /path/to/data.pkl (jsonl)\
        --save_dir ./vis_hand \
        --renderer opencv --render_mode both

    # aitviewer 3D with VLA pipeline camera poses
    xvfb-run -a python vis_hand_demo.py \
        --data_path /path/to/data.pkl (jsonl)\
        --save_dir ./vis_hand \
        --renderer aitviewer --vis_mode cam
"""

import argparse
import os
import pickle
import json
import sys

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch

# ---- HaWoR imports ----
from hawor_utils.common_utils import prepare_hawor_and_add_to_path
prepare_hawor_and_add_to_path()

from hawor_utils.patches.process import get_mano_faces, run_mano, run_mano_left

from lib.models.mano_wrapper import MANO

# ---------------------------------------------------------------
# Data loading & format detection
# ---------------------------------------------------------------
def load_image(image_input):
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ValueError(f"Error: File not found at {image_input}")
            img = cv2.imread(image_input)
        else:
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image.")

        return img
    
    else:
        raise TypeError("Input must be a file path (str) or image bytes (bytes).")


def load_data(data_path, sample_idx=0, video_idx=0):
    """Load data, detect format, compute hand vertices and joints.

    Auto-detects camera poses from ``video_camera_pose_tags`` if present.

    Returns:
        results: dict with 'right' and 'left' keys, each containing:
            {'vertices': (T, V, 3), 'joints': (T, J, 3), 'faces': (F, 3),
             'frame_ids': list}
            or None if hand not detected
        img_focal: float
        frame_paths: list of frame image paths
        cam_c2w: (N, 4, 4) numpy array of camera-to-world transforms,
                 or None if not available
    """
    with open(data_path, 'rb') as f:
        if data_path.endswith('.jsonl'):
            samples = [json.loads(line) for line in f.readlines()]
        elif data_path.endswith('.pkl'):
            samples = pickle.load(f)
        elif data_path.endswith('.parquet'):
            table = pq.read_table(f)
            samples = table.to_pylist()

    tgt_sample = samples[sample_idx]
    meta = tgt_sample['__dj__meta__']
    if isinstance(meta, bytes):
        meta = pickle.loads(meta)
    hawor = meta['hand_reconstruction_hawor_tags'][video_idx]
    frames = tgt_sample['video_frames'][video_idx]

    # Auto-detect camera poses
    cam_c2w = None
    cam_pose_tags = meta.get('video_camera_pose_tags')
    if cam_pose_tags and len(cam_pose_tags) > video_idx:
        cam_pose = cam_pose_tags[video_idx]
        if 'cam_c2w' in cam_pose:
            cam_c2w = np.array(cam_pose['cam_c2w'])
            print(f"  Camera poses detected: {cam_c2w.shape}")

    # Focal length
    if 'img_focal' in hawor:
        img_focal = float(hawor['img_focal'])
    else:
        fov_x = float(hawor['fov_x'])
        img0 = load_image(frames[0])
        W = img0.shape[1]
        img_focal = W / (2.0 * np.tan(fov_x / 2.0))

    # Faces
    faces = get_mano_faces()
    faces_new = np.array([
        [92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
        [120, 108, 78], [78, 108, 79],
    ])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    results = {}

    print("Detected NEW format (axis-angle)")
    for hand_type, mano_fn, hand_faces in [
        ("right", run_mano, faces_right),
        ("left", run_mano_left, faces_left),
    ]:
        hand = hawor[hand_type]
        if not hand['frame_ids']:
            results[hand_type] = None
            continue

        transl = torch.tensor(
            hand['transl'], dtype=torch.float32).unsqueeze(0)
        rot = torch.tensor(
            hand['global_orient'], dtype=torch.float32).unsqueeze(0)
        pose = torch.tensor(
            hand['hand_pose'], dtype=torch.float32).unsqueeze(0)
        betas = torch.tensor(
            hand['betas'], dtype=torch.float32).unsqueeze(0)

        mano_out = mano_fn(transl, rot, pose, betas=betas)
        results[hand_type] = {
            "vertices": mano_out['vertices'][0].cpu().numpy(),
            "joints": mano_out['joints'][0].cpu().numpy(),
            "faces": hand_faces,
            "frame_ids": hand['frame_ids'],
        }
        print(f"  {hand_type}: vertices {results[hand_type]['vertices'].shape}")


    n_frames = len(frames)
    print(f"  Frames: {n_frames}, focal: {img_focal:.1f}")
    return results, img_focal, frames, cam_c2w


# ---------------------------------------------------------------
# OpenCV rendering helpers
# ---------------------------------------------------------------

def project_to_2d(points_3d, focal, width, height):
    """Project 3D points to 2D pixel coords using pinhole model."""
    cx, cy = width / 2.0, height / 2.0
    z = np.where(np.abs(points_3d[..., 2]) < 1e-6, 1e-6, points_3d[..., 2])
    u = focal * points_3d[..., 0] / z + cx
    v = focal * points_3d[..., 1] / z + cy
    return np.stack([u, v], axis=-1)


def draw_mesh_wireframe(frame, verts_2d, faces, color, alpha=0.4,
                        thickness=1):
    """Draw mesh wireframe on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for face in faces:
        pts = verts_2d[face].astype(np.int32)
        if np.any(pts[:, 0] < -w) or np.any(pts[:, 0] > 2 * w):
            continue
        if np.any(pts[:, 1] < -h) or np.any(pts[:, 1] > 2 * h):
            continue
        cv2.polylines(overlay, [pts], isClosed=True, color=color,
                      thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_mesh_filled(frame, verts_2d, faces, color, alpha=0.3):
    """Draw filled semi-transparent mesh on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for face in faces:
        pts = verts_2d[face].astype(np.int32)
        if np.any(pts[:, 0] < -w) or np.any(pts[:, 0] > 2 * w):
            continue
        if np.any(pts[:, 1] < -h) or np.any(pts[:, 1] > 2 * h):
            continue
        cv2.fillPoly(overlay, [pts], color=color, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_mesh_solid(frame, verts_3d, verts_2d, faces, base_color,
                    alpha=0.85, light_dir=None):
    """Draw solid shaded mesh with per-face Lambertian lighting."""
    if light_dir is None:
        light_dir = np.array([0.0, 0.0, -1.0])
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)

    h, w = frame.shape[:2]
    base_color = np.array(base_color, dtype=np.float64)

    face_verts = verts_3d[faces]
    face_depths = face_verts[:, :, 2].mean(axis=1)
    sorted_indices = np.argsort(-face_depths)

    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms < 1e-8, 1.0, norms)

    dots = np.clip(np.dot(normals, -light_dir), 0.0, 1.0)
    intensities = 0.35 + 0.65 * dots

    overlay = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    for idx in sorted_indices:
        pts = verts_2d[faces[idx]].astype(np.int32)
        if np.any(pts[:, 0] < -w) or np.any(pts[:, 0] > 2 * w):
            continue
        if np.any(pts[:, 1] < -h) or np.any(pts[:, 1] > 2 * h):
            continue
        face_color = tuple(
            np.clip(base_color * intensities[idx], 0, 255).astype(int).tolist()
        )
        cv2.fillPoly(overlay, [pts], color=face_color, lineType=cv2.LINE_AA)
        cv2.fillPoly(mask, [pts], color=255, lineType=cv2.LINE_AA)

    mask_bool = mask > 0
    frame[mask_bool] = cv2.addWeighted(
        overlay, alpha, frame, 1 - alpha, 0)[mask_bool]


def draw_joints(frame, joints_2d, color, radius=4):
    """Draw joint points and finger connections on frame."""
    for j in range(joints_2d.shape[0]):
        x, y = int(joints_2d[j, 0]), int(joints_2d[j, 1])
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        if j == 0:
            cv2.circle(frame, (x, y), radius + 3, color, 2, cv2.LINE_AA)

    connections = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    for chain in connections:
        for i in range(len(chain) - 1):
            j1, j2 = chain[i], chain[i + 1]
            if j1 < joints_2d.shape[0] and j2 < joints_2d.shape[0]:
                pt1 = (int(joints_2d[j1, 0]), int(joints_2d[j1, 1]))
                pt2 = (int(joints_2d[j2, 0]), int(joints_2d[j2, 1]))
                cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)


def render_opencv_frame(frame, right_data, left_data, img_focal, frame_idx,
                        render_mode="wireframe"):
    """Render hand mesh overlay on a single frame using OpenCV."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    right_mesh_color = (180, 120, 200)
    right_joint_color = (200, 100, 255)
    left_mesh_color = (200, 150, 50)
    left_joint_color = (255, 170, 50)

    for data, mesh_color, joint_color, label in [
        (right_data, right_mesh_color, right_joint_color, "R"),
        (left_data, left_mesh_color, left_joint_color, "L"),
    ]:
        if data is None:
            continue

        verts = data["vertices"][frame_idx]
        joints = data["joints"][frame_idx]
        faces = data["faces"]

        verts_2d = project_to_2d(verts, img_focal, w, h)
        joints_2d = project_to_2d(joints, img_focal, w, h)

        if render_mode == "solid":
            draw_mesh_solid(annotated, verts, verts_2d, faces, mesh_color,
                            alpha=0.85)
        else:
            draw_mesh_filled(annotated, verts_2d, faces, mesh_color,
                             alpha=0.25)
            draw_mesh_wireframe(annotated, verts_2d, faces, mesh_color,
                                alpha=0.5, thickness=1)
            draw_joints(annotated, joints_2d, joint_color, radius=3)

        wrist = joints_2d[0].astype(int)
        cv2.putText(annotated, label, (wrist[0] + 8, wrist[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, joint_color, 2,
                    cv2.LINE_AA)

    cv2.putText(annotated, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    return annotated


# ---------------------------------------------------------------
# World-space trajectory helpers (OpenCV, requires camera poses)
# ---------------------------------------------------------------

def world_to_camera(point_world, cam_c2w):
    """Transform a world-space 3D point to camera-space using c2w matrix.

    Args:
        point_world: (3,) world-space position
        cam_c2w: (4, 4) camera-to-world transform

    Returns:
        (3,) camera-space position
    """
    R = cam_c2w[:3, :3]
    t = cam_c2w[:3, 3]
    # w2c: R_w2c = R_c2w^T, t_w2c = -R_c2w^T @ t_c2w
    return R.T @ (point_world - t)


def compute_world_wrist_positions(results, cam_c2w):
    """Compute wrist positions in world space using camera poses.

    For each hand, transforms the MANO wrist joint (joints[frame, 0])
    from camera space to world space.

    Returns:
        dict mapping hand_type -> list of (frame_idx, world_pos) tuples
    """
    world_wrists = {}
    for hand_type in ("right", "left"):
        data = results.get(hand_type)
        if data is None:
            continue
        frame_ids = data["frame_ids"]
        joints = data["joints"]  # (T, J, 3)
        positions = []
        for i, fid in enumerate(frame_ids):
            if fid >= cam_c2w.shape[0]:
                continue
            wrist_cam = joints[i, 0, :]  # camera-space wrist
            c2w = cam_c2w[fid]
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            wrist_world = R @ wrist_cam + t
            positions.append((fid, wrist_world))
        if positions:
            world_wrists[hand_type] = positions
    return world_wrists


def draw_trajectory_birdseye(world_wrists, cam_c2w, canvas_size=300,
                              margin=30):
    """Draw bird's-eye view (XZ plane) of camera path and wrist trajectories.

    Args:
        world_wrists: dict from compute_world_wrist_positions
        cam_c2w: (N, 4, 4) camera-to-world transforms
        canvas_size: output image size (square)
        margin: pixel margin

    Returns:
        (canvas_size, canvas_size, 3) BGR image
    """
    # Collect all points for axis range
    all_points = []
    cam_positions = cam_c2w[:, :3, 3]  # (N, 3)
    all_points.append(cam_positions[:, [0, 2]])  # XZ
    for positions in world_wrists.values():
        pts = np.array([p for _, p in positions])
        all_points.append(pts[:, [0, 2]])
    all_pts = np.concatenate(all_points, axis=0)

    # Compute bounds
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    span = mx - mn
    span = np.where(span < 1e-6, 1.0, span)
    scale = (canvas_size - 2 * margin) / span.max()

    def to_px(xz):
        centered = (xz - (mn + mx) / 2) * scale
        return (centered + canvas_size / 2).astype(np.int32)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    canvas[:] = 30  # dark background

    # Draw camera path
    cam_xz = cam_positions[:, [0, 2]]
    cam_px = to_px(cam_xz)
    for i in range(len(cam_px) - 1):
        cv2.line(canvas, tuple(cam_px[i]), tuple(cam_px[i + 1]),
                 (100, 100, 100), 1, cv2.LINE_AA)
    # Camera start/end markers
    if len(cam_px) > 0:
        cv2.circle(canvas, tuple(cam_px[0]), 4, (255, 255, 255), -1)
        cv2.circle(canvas, tuple(cam_px[-1]), 4, (150, 150, 150), -1)

    # Draw wrist trajectories
    hand_colors = {"right": (200, 100, 255), "left": (255, 170, 50)}
    for hand_type, positions in world_wrists.items():
        color = hand_colors.get(hand_type, (200, 200, 200))
        pts_xz = np.array([p[1][[0, 2]] for p in positions])
        pts_px = to_px(pts_xz)
        for i in range(len(pts_px) - 1):
            cv2.line(canvas, tuple(pts_px[i]), tuple(pts_px[i + 1]),
                     color, 2, cv2.LINE_AA)
        # Start circle
        if len(pts_px) > 0:
            cv2.circle(canvas, tuple(pts_px[0]), 5, color, -1)

    # Label
    cv2.putText(canvas, "Bird's-eye (XZ)", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                cv2.LINE_AA)
    return canvas


def overlay_trajectory_on_frame(frame, world_wrists, cam_c2w, current_fid,
                                 img_focal):
    """Project world-space wrist trajectory onto the current camera frame.

    Draws historical wrist positions as a fading trail on the frame.
    """
    h, w = frame.shape[:2]
    if current_fid >= cam_c2w.shape[0]:
        return

    c2w = cam_c2w[current_fid]
    hand_colors = {"right": (200, 100, 255), "left": (255, 170, 50)}

    for hand_type, positions in world_wrists.items():
        color = hand_colors.get(hand_type, (200, 200, 200))
        # Only draw positions up to current frame
        trail = [(fid, pos) for fid, pos in positions if fid <= current_fid]
        if len(trail) < 2:
            continue

        pts_2d = []
        for fid, wrist_world in trail:
            wrist_cam = world_to_camera(wrist_world, c2w)
            if wrist_cam[2] <= 0:
                pts_2d.append(None)
                continue
            cx, cy = w / 2.0, h / 2.0
            u = img_focal * wrist_cam[0] / wrist_cam[2] + cx
            v = img_focal * wrist_cam[1] / wrist_cam[2] + cy
            pts_2d.append((int(u), int(v)))

        # Draw trail with fading alpha
        n = len(pts_2d)
        for i in range(n - 1):
            if pts_2d[i] is None or pts_2d[i + 1] is None:
                continue
            alpha = 0.3 + 0.7 * (i / max(n - 1, 1))
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts_2d[i], pts_2d[i + 1], fade_color, 2,
                     cv2.LINE_AA)

        # Draw current wrist position
        if pts_2d and pts_2d[-1] is not None:
            cv2.circle(frame, pts_2d[-1], 5, color, -1, cv2.LINE_AA)


# ---------------------------------------------------------------
# OpenCV renderer
# ---------------------------------------------------------------

def run_opencv_renderer(results, img_focal, frames, save_dir,
                        render_mode="both", fps=2.0, cam_c2w=None):
    """Render hand mesh overlay on video frames using OpenCV.

    When *cam_c2w* is provided, also renders:
      - World-space wrist trajectory projected onto each frame
      - A bird's-eye (XZ plane) mini-map showing camera path and wrist tracks
    """
    n_frames = len(frames)
    has_cam = cam_c2w is not None

    right_id_map = {}
    left_id_map = {}
    if results.get("right"):
        right_id_map = {fid: i for i, fid in
                        enumerate(results["right"]["frame_ids"])}
    if results.get("left"):
        left_id_map = {fid: i for i, fid in
                       enumerate(results["left"]["frame_ids"])}

    # Pre-compute world-space wrist positions (if camera poses available)
    world_wrists = {}
    birdseye_img = None
    if has_cam:
        world_wrists = compute_world_wrist_positions(results, cam_c2w)
        if world_wrists:
            birdseye_img = draw_trajectory_birdseye(
                world_wrists, cam_c2w, canvas_size=250, margin=25)
            print(f"  World-space trajectory: "
                  f"{', '.join(f'{k}={len(v)} pts' for k, v in world_wrists.items())}")

    render_modes = (["wireframe", "solid"] if render_mode == "both"
                    else [render_mode])

    for mode in render_modes:
        print(f"\n{'='*50}")
        print(f"  Rendering mode: {mode}"
              f"{' + trajectory' if has_cam else ''}")
        print(f"{'='*50}")

        frames_dir = os.path.join(save_dir, f"frames_{mode}")
        os.makedirs(frames_dir, exist_ok=True)

        print(f"Rendering {n_frames} frames...")
        annotated_frames = []
        for frame_idx in range(n_frames):
            frame = load_image(frames[frame_idx])
            if frame is None:
                print(f"  [WARN] Cannot read: {frames[frame_idx]}")
                continue

            r_frame_data = None
            if frame_idx in right_id_map:
                r_frame_data = {
                    "vertices": results["right"]["vertices"],
                    "joints": results["right"]["joints"],
                    "faces": results["right"]["faces"],
                }

            l_frame_data = None
            if frame_idx in left_id_map:
                l_frame_data = {
                    "vertices": results["left"]["vertices"],
                    "joints": results["left"]["joints"],
                    "faces": results["left"]["faces"],
                }

            annotated = render_opencv_frame(
                frame, r_frame_data, l_frame_data,
                img_focal, frame_idx, render_mode=mode,
            )

            # Overlay world-space trajectory on frame
            if has_cam and world_wrists:
                overlay_trajectory_on_frame(
                    annotated, world_wrists, cam_c2w, frame_idx, img_focal)

            # Overlay bird's-eye mini-map in bottom-right corner
            if birdseye_img is not None:
                bh, bw = birdseye_img.shape[:2]
                ah, aw = annotated.shape[:2]
                y0, x0 = ah - bh - 10, aw - bw - 10
                if y0 >= 0 and x0 >= 0:
                    roi = annotated[y0:y0 + bh, x0:x0 + bw]
                    cv2.addWeighted(birdseye_img, 0.8, roi, 0.2, 0, roi)
                    annotated[y0:y0 + bh, x0:x0 + bw] = roi

            out_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(out_path, annotated)
            annotated_frames.append(annotated)

        # Save video
        if annotated_frames:
            import fractions
            import av

            video_name = f"hand_mesh_{mode}.mp4"
            video_path = os.path.join(save_dir, video_name)
            h, w = annotated_frames[0].shape[:2]
            fps_frac = fractions.Fraction(fps).limit_denominator(10000)

            container = av.open(video_path, mode="w")
            stream = container.add_stream("libx264", rate=fps_frac)
            stream.width = w
            stream.height = h
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}

            for frame_bgr in annotated_frames:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)
            container.close()

            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  Video: {video_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------
# Aitviewer renderer
# ---------------------------------------------------------------

def cam_c2w_to_Rt(cam_c2w, num_frames):
    """Extract (R_c2w, t_c2w) torch tensors from (N, 4, 4) numpy array."""
    R_c2w = torch.tensor(
        cam_c2w[:num_frames, :3, :3], dtype=torch.float32)
    t_c2w = torch.tensor(
        cam_c2w[:num_frames, :3, 3], dtype=torch.float32)
    return R_c2w, t_c2w


def load_camera_from_hawor_slam(slam_npz_path, num_frames):
    """Load camera poses from HaWoR SLAM .npz file."""
    from lib.eval_utils.custom_utils import load_slam_cam
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_npz_path)
    R_c2w = R_c2w[:num_frames].float()
    t_c2w = t_c2w[:num_frames].float()
    print(f"Loaded HaWoR SLAM camera poses: R_c2w {R_c2w.shape}")
    return R_c2w, t_c2w


def run_aitviewer_renderer(results, img_focal, frames, save_dir,
                           cam_c2w=None, slam_npz=None, vis_mode='world',
                           interactive=False):
    """Visualize hands with camera poses using aitviewer."""
    from lib.vis.run_vis2 import run_vis2_on_video
    from hawor_utils.patches.run_vis2 import run_vis2_on_video_cam

    faces_right = results["right"]["faces"] if results.get("right") else None
    faces_left = results["left"]["faces"] if results.get("left") else None

    # Build vertex tensors in (1, T, V, 3) format for aitviewer
    right_dict = None
    left_dict = None

    if results.get("right"):
        right_dict = {
            'vertices': torch.tensor(
                results["right"]["vertices"],
                dtype=torch.float32).unsqueeze(0),
            'faces': faces_right,
        }
    if results.get("left"):
        left_dict = {
            'vertices': torch.tensor(
                results["left"]["vertices"],
                dtype=torch.float32).unsqueeze(0),
            'faces': faces_left,
        }

    # Use the shorter sequence length
    T_frames = 0
    if right_dict is not None:
        T_frames = max(T_frames, right_dict['vertices'].shape[1])
    if left_dict is not None:
        T_frames = max(T_frames, left_dict['vertices'].shape[1])

    # R_x transform (HaWoR convention: flip Y/Z for aitviewer)
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()

    # Load camera poses
    if cam_c2w is not None:
        R_c2w_sla_all, t_c2w_sla_all = cam_c2w_to_Rt(cam_c2w, T_frames)
        print("Camera source: VLA pipeline")
    elif slam_npz:
        R_c2w_sla_all, t_c2w_sla_all = load_camera_from_hawor_slam(
            slam_npz, T_frames)
        print("Camera source: HaWoR SLAM (.npz)")
    else:
        R_c2w_sla_all = torch.eye(3).unsqueeze(0).expand(
            T_frames, -1, -1).clone()
        t_c2w_sla_all = torch.zeros(T_frames, 3)
        print("Camera source: identity")

    # Apply R_x transform
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum(
        "bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    if left_dict is not None:
        left_dict['vertices'] = torch.einsum(
            'ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    if right_dict is not None:
        right_dict['vertices'] = torch.einsum(
            'ij,btnj->btni', R_x, right_dict['vertices'].cpu())

    # Provide empty dict if None (aitviewer needs both)
    if left_dict is None:
        left_dict = {
            'vertices': torch.zeros(1, T_frames, 778, 3),
            'faces': get_mano_faces(),
        }
    if right_dict is None:
        right_dict = {
            'vertices': torch.zeros(1, T_frames, 778, 3),
            'faces': get_mano_faces(),
        }

    images = frames[:T_frames]
    print(f"Visualizing frames 0 to {T_frames}")

    if vis_mode == 'world':
        run_vis2_on_video(
            left_dict, right_dict, save_dir, img_focal, images,
            R_c2w=R_c2w_sla_all, t_c2w=t_c2w_sla_all,
            interactive=interactive)
    elif vis_mode == 'cam':
        run_vis2_on_video_cam(
            left_dict, right_dict, save_dir, img_focal, images,
            R_w2c=R_w2c_sla_all, t_w2c=t_w2c_sla_all, interactive=interactive)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize MANO hand mesh with optional camera poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to data. The data format  Camera poses are auto-detected if present.")
    parser.add_argument("--save_dir", type=str, default="./vis_hand")
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--video_idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2.0)

    # Renderer selection
    parser.add_argument("--renderer", type=str, default="opencv",
                        choices=["opencv", "aitviewer"],
                        help="Rendering backend: opencv (2D overlay) or "
                             "aitviewer (3D viewer)")

    # OpenCV-specific options
    parser.add_argument("--render_mode", type=str, default="both",
                        choices=["wireframe", "solid", "both"],
                        help="OpenCV render mode (wireframe/solid/both)")

    # Aitviewer-specific options
    parser.add_argument("--slam_npz", type=str, default=None,
                        help="Path to HaWoR SLAM .npz file (overrides "
                             "auto-detected camera poses)")
    parser.add_argument("--vis_mode", type=str, default='world',
                        choices=["cam", "world"],
                        help="Aitviewer visualization mode")
    parser.add_argument("--interactive", action='store_true',
                        help="Interactive aitviewer (requires display)")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data (camera poses auto-detected from data)
    print(f"Loading: {args.data_path}")
    results, img_focal, frame_paths, cam_c2w = load_data(
        args.data_path, args.sample_idx, args.video_idx)

    if cam_c2w is not None:
        print("Camera poses: available (auto-detected from data)")
    else:
        print("Camera poses: not available")

    # Render
    if args.renderer == "opencv":
        run_opencv_renderer(
            results, img_focal, frame_paths, args.save_dir,
            render_mode=args.render_mode, fps=args.fps,
            cam_c2w=cam_c2w)
    elif args.renderer == "aitviewer":
        run_aitviewer_renderer(
            results, img_focal, frame_paths, args.save_dir,
            cam_c2w=cam_c2w, slam_npz=args.slam_npz,
            vis_mode=args.vis_mode, interactive=args.interactive)

    print("\nDone!")


if __name__ == "__main__":
    main()
