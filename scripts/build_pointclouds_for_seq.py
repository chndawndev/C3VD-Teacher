#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_pointclouds_for_seq.py

Given a C3VDv2 sequence name seq:
  1. Build full GT point cloud directly from GT (fisheye depth + pose.txt)
       -> web-viewer/models/<seq>_gt_full.ply
  2. Build NeRF point cloud from Nerfstudio rendered raw-depth (train+test)
       -> web-viewer/models/<seq>_nerf_rawdepth_alpha0.10m.ply   (alpha can be changed)

Usage example:
  python build_pointclouds_for_seq.py c1_descending_t2_v2
  python build_pointclouds_for_seq.py c1_descending_t2_v2 --alpha 0.1
"""

import os
import json
import gzip
import argparse
import numpy as np
from PIL import Image

import torch


# ================= Global paths & constants =================

BASE_ROOT = "/data1_ycao/chua/projects/cdTeacher"
DATA_RAW_ROOT = os.path.join(BASE_ROOT, "data_raw")
STAGE_A_ROOT = os.path.join(BASE_ROOT, "outputs", "stage_A")
STAGE_D_ROOT = os.path.join(BASE_ROOT, "outputs", "stage_D")
MODELS_ROOT = os.path.join(BASE_ROOT, "web-viewer", "models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Using device:", device)

# Scaramuzza fisheye intrinsics provided by C3VDv2 (shared across sequence)
OMNI_INTRINSICS = {
    "width": 1350,
    "height": 1080,
    "cx": 677.739464094188,
    "cy": 543.057997844875,
    "a0": 767.733695862103,
    "a1": 0.0,
    "a2": -0.000592506426558248,
    "a3": -2.69440266600040e-07,
    "a4": -2.16380341010063e-10,
    "c": 0.9999,
    "d": 1.10e-4,
    "e": -1.83e-4,
}

# Scale from NeRF raw-depth units to meters (tunable via CLI)
DEFAULT_ALPHA_M_PER_UNIT = 0.1

# GT point cloud sampling control
MAX_POINTS_PER_FRAME_GT = 10000   # Cap per-frame samples to avoid GPU/CPU OOM
MAX_POINTS_TOTAL_GT = 500000      # Global cap to keep viewer responsive

# NeRF point cloud sampling control
MAX_POINTS_PER_FRAME_NERF = 10000
MAX_POINTS_TOTAL_NERF = 300000


# ================= Small utilities =================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_ply(path: str, points: np.ndarray):
    """Lightweight PLY writer for XYZ vertices only. points: (N,3) float32."""
    N = points.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
    print(f"[WRITE] Saved PLY with {N} points to {path}")


# ================= Fisheye camera model (OmniCamera) =================

class OmniCamera(torch.nn.Module):
    """
    Scaramuzza-style omnidirectional camera (pixel -> ray in camera frame).

    Coordinate convention:
      - Pixel: u to the right, v downward
      - Camera: +x right, +y down, +z along the viewing direction
    """
    def __init__(self, intrinsics):
        super().__init__()
        self.width = intrinsics["width"]
        self.height = intrinsics["height"]
        self.cx = intrinsics["cx"]
        self.cy = intrinsics["cy"]
        self.c = intrinsics["c"]
        self.d = intrinsics["d"]
        self.e = intrinsics["e"]

        self.register_buffer(
            "pol",
            torch.tensor([
                intrinsics["a0"],
                intrinsics["a1"],
                intrinsics["a2"],
                intrinsics["a3"],
                intrinsics["a4"],
            ], dtype=torch.float32),
        )

        A = torch.tensor([[self.c, self.d],
                          [self.e, 1.0]], dtype=torch.float32)
        A_inv = torch.inverse(A)
        self.register_buffer("A_inv", A_inv)

    def forward(self, u, v):
        """
        u, v: (...,) pixel coordinates, float32
        Returns: (..., 3) unit direction vector (x, y, z) in camera frame
        """
        u = torch.as_tensor(u, dtype=torch.float32, device=self.pol.device)
        v = torch.as_tensor(v, dtype=torch.float32, device=self.pol.device)

        x_img = u - self.cx
        y_img = v - self.cy

        x_prime = self.A_inv[0, 0] * x_img + self.A_inv[0, 1] * y_img
        y_prime = self.A_inv[1, 0] * x_img + self.A_inv[1, 1] * y_img

        r = torch.sqrt(x_prime**2 + y_prime**2)

        powers = torch.stack([r**i for i in range(self.pol.shape[0])], dim=0)
        z = (self.pol.view(-1, *([1] * (powers.ndim - 1))) * powers).sum(dim=0)

        dir_cam = torch.stack([x_prime, y_prime, z], dim=-1)
        dir_norm = dir_cam / torch.linalg.norm(dir_cam, dim=-1, keepdim=True).clamp(min=1e-9)
        return dir_norm


# ================= Load cam2world poses from pose.txt =================

def load_cam2world_poses(raw_root: str) -> torch.Tensor:
    """
    Read 4x4 cam2world matrices (meters) from raw_root/pose.txt.
    C3VDv2: each line has 16 values, flatten then reshape(4,4), followed by a transpose.
    Translation values are in millimeters and must be divided by 1000.
    """
    pose_path = os.path.join(raw_root, "pose.txt")
    if not os.path.isfile(pose_path):
        raise FileNotFoundError(f"[GT] pose.txt not found at: {pose_path}")

    poses_list = []
    with open(pose_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) != 16:
                raise ValueError(
                    f"[GT] Line {line_idx} in pose.txt has {len(parts)} values, expected 16.\n"
                    f"Line content: {line}"
                )
            vals = np.array(parts, dtype=np.float32)
            mat_raw = vals.reshape(4, 4)
            T = mat_raw.T  # Transpose to standard cam2world

            # Translation: mm -> m
            T[0:3, 3] /= 1000.0
            poses_list.append(T)

    poses_cam2world = torch.from_numpy(np.stack(poses_list, axis=0))  # (N,4,4)
    print("[GT] Loaded poses:", poses_cam2world.shape)
    return poses_cam2world


# ================= Build GT point cloud from depth + occlusion + fisheye =================

def load_depth_and_mask(raw_root: str, frame_id: int, device: torch.device):
    """
    Load one frame of depth and occlusion, returning:
      depth_m: (H,W) float32, in meters
      valid:   (H,W) bool, valid pixel mask
    """
    fname = f"{frame_id:04d}"
    depth_path = os.path.join(raw_root, "depth", f"{fname}_depth.tiff")
    occ_path   = os.path.join(raw_root, "occlusions", f"{fname}_occlusion.png")

    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"[GT] Depth file not found: {depth_path}")
    if not os.path.isfile(occ_path):
        raise FileNotFoundError(f"[GT] Occlusion file not found: {occ_path}")

    depth_img = np.array(Image.open(depth_path))  # uint16
    occ_img   = np.array(Image.open(occ_path))    # uint8

    # torch does not support uint16; convert to int32
    depth_raw = torch.from_numpy(depth_img.astype(np.int32)).to(device)
    occ_u8    = torch.from_numpy(occ_img.astype(np.uint8)).to(device)

    # 0~65535 -> 0~0.1 m (0~100mm)
    depth_m = depth_raw.to(torch.float32) / 65535.0 * 0.1

    # Valid pixels: has depth & no occlusion
    valid = (depth_raw > 0) & (occ_u8 == 0)

    return depth_m, valid


def build_gt_pointcloud_for_seq(seq_name: str) -> str:
    """
    Build a full GT point cloud using all frames of depth + occlusion + pose + fisheye model,
    saved to web-viewer/models/<seq>_gt_full.ply.
    """
    ensure_dir(MODELS_ROOT)

    raw_root = os.path.join(DATA_RAW_ROOT, seq_name)
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"[GT] Raw seq dir not found: {raw_root}")

    # Build fisheye camera model
    omni_cam = OmniCamera(OMNI_INTRINSICS).to(device)

    H = OMNI_INTRINSICS["height"]
    W = OMNI_INTRINSICS["width"]

    # Precompute per-pixel ray directions (camera frame)
    u_coords = torch.arange(W, device=device).view(1, -1).expand(H, W)
    v_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W)
    with torch.no_grad():
        dirs_cam = omni_cam(u_coords, v_coords)  # (H,W,3)
    print("[GT] dirs_cam shape:", dirs_cam.shape)

    # Load cam2world poses
    poses_cam2world = load_cam2world_poses(raw_root).to(device)  # (N,4,4)
    num_frames = poses_cam2world.shape[0]
    print(f"[GT] Total frames in pose.txt: {num_frames}")

    all_points = []

    for fid in range(num_frames):
        depth_m, valid = load_depth_and_mask(raw_root, fid, device=device)  # (H,W)
        valid_mask = valid & (depth_m > 0)
        if valid_mask.sum().item() == 0:
            continue

        depth_z_valid = depth_m[valid_mask]       # (Nv,)
        dirs_valid = dirs_cam[valid_mask]         # (Nv,3)

        dz = dirs_valid[:, 2].clamp(min=1e-6)     # z-axis component
        scale = depth_z_valid / dz                # (Nv,)
        pts_cam = dirs_valid * scale.unsqueeze(-1)  # (Nv,3)

        T = poses_cam2world[fid]                  # (4,4)
        R = T[:3, :3]
        t = T[:3, 3]

        pts_world = (R @ pts_cam.T + t.view(3, 1)).T  # (Nv,3)

        if MAX_POINTS_PER_FRAME_GT is not None and pts_world.shape[0] > MAX_POINTS_PER_FRAME_GT:
            idx = torch.randperm(pts_world.shape[0], device=device)[:MAX_POINTS_PER_FRAME_GT]
            pts_world = pts_world[idx]

        all_points.append(pts_world.cpu())
        if (fid + 1) % 50 == 0:
            print(f"[GT] Frame {fid+1}/{num_frames} processed")

    if not all_points:
        raise RuntimeError(f"[GT] No GT points collected for seq '{seq_name}'. "
                           f"Check depth / occlusions / pose.")

    all_points_tensor = torch.cat(all_points, dim=0)  # (N_total,3)
    pts_np = all_points_tensor.numpy().astype(np.float32)
    print(f"[GT] Total GT points before global subsample: {pts_np.shape[0]}")

    if MAX_POINTS_TOTAL_GT is not None and pts_np.shape[0] > MAX_POINTS_TOTAL_GT:
        sel = np.random.choice(pts_np.shape[0], MAX_POINTS_TOTAL_GT, replace=False)
        pts_np = pts_np[sel]
        print(f"[GT] Subsampled GT points to {pts_np.shape[0]}.")

    out_path = os.path.join(MODELS_ROOT, f"{seq_name}_gt_full.ply")
    write_ply(out_path, pts_np)
    return out_path


# ================= NeRF raw-depth → point cloud =================

def build_nerf_pointcloud_for_seq(
    seq_name: str,
    alpha_m_per_unit: float = DEFAULT_ALPHA_M_PER_UNIT,
    max_points_per_frame: int = MAX_POINTS_PER_FRAME_NERF,
    max_points_total: int = MAX_POINTS_TOTAL_NERF,
) -> str:
    """
    Build a NeRF point cloud using ns-render dataset --split train+test raw-depth outputs.

    Assumes:
      STAGE_A_ROOT/<seq>/transforms.json
      STAGE_D_ROOT/<seq>/train/raw-depth/*.npy.gz
      STAGE_D_ROOT/<seq>/test/raw-depth/*.npy.gz
    """

    ensure_dir(MODELS_ROOT)

    stage_a_seq_root = os.path.join(STAGE_A_ROOT, seq_name)
    stage_d_seq_root = os.path.join(STAGE_D_ROOT, seq_name)

    transforms_path = os.path.join(stage_a_seq_root, "transforms.json")
    if not os.path.isfile(transforms_path):
        raise FileNotFoundError(
            f"[NeRF] transforms.json not found for seq '{seq_name}'. "
            f"Expected at: {transforms_path}"
        )

    with open(transforms_path, "r") as f:
        meta = json.load(f)

    W_p = int(meta["w"])
    H_p = int(meta["h"])
    fx = float(meta["fl_x"])
    fy = float(meta["fl_y"])
    cx = float(meta["cx"])
    cy = float(meta["cy"])
    frames_meta = meta["frames"]

    print(f"[NeRF] transforms: W={W_p}, H={H_p}, fx={fx:.4f}, fy={fy:.4f}, "
          f"cx={cx:.2f}, cy={cy:.2f}, #frames={len(frames_meta)}")

    # Precompute pinhole camera ray directions
    u_grid, v_grid = np.meshgrid(np.arange(W_p), np.arange(H_p))
    u_grid = u_grid.astype(np.float32)
    v_grid = v_grid.astype(np.float32)
    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fy
    z = np.ones_like(x, dtype=np.float32)
    ray_dir = np.stack([x, y, z], axis=-1)
    ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1, keepdims=True).clip(min=1e-6)

    all_pts = []

    for split in ["train", "test"]:
        rd_dir = os.path.join(stage_d_seq_root, split, "raw-depth")
        if not os.path.isdir(rd_dir):
            print(f"[NeRF] Skip split '{split}' (no dir: {rd_dir})")
            continue

        files = sorted(f for f in os.listdir(rd_dir) if f.endswith(".npy.gz"))
        if not files:
            print(f"[NeRF] No raw-depth files under: {rd_dir}")
            continue

        print(f"[NeRF] Using {len(files)} raw-depth frames from split '{split}'")

        for idx, fname in enumerate(files):
            stem = os.path.splitext(os.path.splitext(fname)[0])[0]  # 0010.npy.gz -> 0010
            try:
                frame_id = int(stem)
            except ValueError:
                print(f"[NeRF] WARNING: cannot parse frame id from {fname}, skip.")
                continue

            raw_path = os.path.join(rd_dir, fname)
            with gzip.open(raw_path, "rb") as f:
                d = np.load(f, allow_pickle=True)  # (H,W,1) or (H,W)

            if d.ndim == 3 and d.shape[2] == 1:
                d = d[..., 0]

            if d.shape[0] != H_p or d.shape[1] != W_p:
                raise ValueError(
                    f"[NeRF] raw-depth shape {d.shape} vs transforms w/h ({W_p},{H_p}) "
                    f"for file {fname} in split {split}"
                )

            d_valid = d.astype(np.float32)
            valid_mask = d_valid > 0
            if not np.any(valid_mask):
                continue

            depth_vals = (alpha_m_per_unit * d_valid[valid_mask]).astype(np.float32)  # (Nv,)
            dirs_valid = ray_dir[valid_mask]  # (Nv,3)
            pts_cam = dirs_valid * depth_vals[:, None]  # (Nv,3)

            if frame_id >= len(frames_meta):
                raise IndexError(
                    f"[NeRF] frame_id={frame_id} out of range in transforms.json "
                    f"(len={len(frames_meta)}). Check ns-render / transforms mapping."
                )

            T_c2w = np.array(frames_meta[frame_id]["transform_matrix"], dtype=np.float32)
            R = T_c2w[:3, :3]
            t = T_c2w[:3, 3]

            pts_world = (R @ pts_cam.T + t.reshape(3, 1)).T  # (Nv,3)

            if max_points_per_frame is not None and pts_world.shape[0] > max_points_per_frame:
                sel = np.random.choice(pts_world.shape[0], max_points_per_frame, replace=False)
                pts_world = pts_world[sel]

            all_pts.append(pts_world.astype(np.float32))

            if (idx + 1) % 20 == 0:
                print(f"[NeRF] [{split}] frame {idx+1}/{len(files)} processed")

    if not all_pts:
        raise RuntimeError(f"[NeRF] No NeRF points collected for seq '{seq_name}'. "
                           f"Check raw-depth rendering.")

    pts_cat = np.concatenate(all_pts, axis=0).astype(np.float32)
    print(f"[NeRF] Total NeRF points before global subsample: {pts_cat.shape[0]}")

    if max_points_total is not None and pts_cat.shape[0] > max_points_total:
        sel = np.random.choice(pts_cat.shape[0], max_points_total, replace=False)
        pts_cat = pts_cat[sel]
        print(f"[NeRF] Subsampled NeRF points to {pts_cat.shape[0]}.")

    out_name = f"{seq_name}_nerf_rawdepth_alpha{alpha_m_per_unit:.2f}m.ply"
    out_path = os.path.join(MODELS_ROOT, out_name)
    write_ply(out_path, pts_cat)
    return out_path


# ================= CLI entrypoint =================

def main():
    parser = argparse.ArgumentParser(
        description="Build GT & NeRF pointclouds for a given C3VDv2 sequence."
    )
    parser.add_argument(
        "seq_name",
        type=str,
        help="Sequence name, e.g. c1_descending_t2_v2",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA_M_PER_UNIT,
        help=f"Scale factor (meters per raw-depth unit). Default: {DEFAULT_ALPHA_M_PER_UNIT}",
    )
    parser.add_argument(
        "--skip-gt",
        action="store_true",
        help="Skip building GT pointcloud (only build NeRF).",
    )
    parser.add_argument(
        "--skip-nerf",
        action="store_true",
        help="Skip building NeRF pointcloud (only build GT).",
    )
    args = parser.parse_args()

    seq = args.seq_name
    alpha = args.alpha

    print("===================================================")
    print(f"[SEQ]     {seq}")
    print(f"[ALPHA]   {alpha} m / raw-depth-unit")
    print(f"[DEVICE]  {device}")
    print("===================================================")

    ensure_dir(MODELS_ROOT)

    if not args.skip_gt:
        gt_path = build_gt_pointcloud_for_seq(seq)
        print(f"[DONE] GT pointcloud:   {gt_path}")

    if not args.skip_nerf:
        nerf_path = build_nerf_pointcloud_for_seq(
            seq,
            alpha_m_per_unit=alpha,
            max_points_per_frame=MAX_POINTS_PER_FRAME_NERF,
            max_points_total=MAX_POINTS_TOTAL_NERF,
        )
        print(f"[DONE] NeRF pointcloud: {nerf_path}")


if __name__ == "__main__":
    main()
