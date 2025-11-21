#!/usr/bin/env python
# stage_A_build_gt_for_eval.py
#
# Build "aligned" GT point cloud for Stage D evaluation using frames from ns-render raw-depth.
#
# Default path convention:
#   RAW_ROOT      = /data1_ycao/chua/projects/cdTeacher/data_raw/<SEQ>
#   STAGE_D_ROOT  = /data1_ycao/chua/projects/cdTeacher/outputs/stage_D/<SEQ>
#
# Output:
#   <STAGE_D_ROOT>/gt_pointcloud_eval_frames.ply

import os
import re
import argparse
import numpy as np
import torch
from PIL import Image

# -------------------------
# Camera intrinsics (C3VDv2 / Olympus CF-HQ190L)
# -------------------------
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


class OmniCamera(torch.nn.Module):
    """
    Scaramuzza-style omnidirectional camera (pixel -> ray in camera frame).

    Coordinate system convention:
      - Pixel: u right, v down
      - Camera frame: +x right, +y down, +z along view direction
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
            torch.tensor(
                [
                    intrinsics["a0"],
                    intrinsics["a1"],
                    intrinsics["a2"],
                    intrinsics["a3"],
                    intrinsics["a4"],
                ],
                dtype=torch.float32,
            ),
        )

        A = torch.tensor([[self.c, self.d], [self.e, 1.0]], dtype=torch.float32)
        A_inv = torch.inverse(A)
        self.register_buffer("A_inv", A_inv)

    def forward(self, u, v):
        u = torch.as_tensor(u, dtype=torch.float32, device=self.pol.device)
        v = torch.as_tensor(v, dtype=torch.float32, device=self.pol.device)

        x_img = u - self.cx
        y_img = v - self.cy

        x_prime = self.A_inv[0, 0] * x_img + self.A_inv[0, 1] * y_img
        y_prime = self.A_inv[1, 0] * x_img + self.A_inv[1, 1] * y_img

        r = torch.sqrt(x_prime**2 + y_prime**2)

        powers = torch.stack(
            [r**i for i in range(self.pol.shape[0])], dim=0
        )  # (deg+1, ...)
        z = (self.pol.view(-1, *([1] * (powers.ndim - 1))) * powers).sum(dim=0)

        dir_cam = torch.stack([x_prime, y_prime, z], dim=-1)
        dir_norm = dir_cam / torch.linalg.norm(
            dir_cam, dim=-1, keepdim=True
        ).clamp(min=1e-9)
        return dir_norm


def load_poses_cam2world(pose_path: str) -> torch.Tensor:
    """读取 pose.txt，返回 (N,4,4) 的 cam2world（单位：米）"""
    poses_list = []
    with open(pose_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) != 16:
                raise ValueError(
                    f"Line {line_idx} in pose.txt has {len(parts)} values, expected 16. Line content: {line}"
                )
            vals = np.array(parts, dtype=np.float32)
            mat_raw = vals.reshape(4, 4)
            T = mat_raw.T  # 转置到标准 cam2world
            T[0:3, 3] /= 1000.0  # mm -> m
            poses_list.append(T)
    poses = torch.from_numpy(np.stack(poses_list, axis=0))  # (N,4,4)
    return poses


def load_depth_and_mask(raw_root: str, frame_id: int, device: torch.device):
    """加载单帧 GT 深度 & occlusion，返回 depth_m (H,W) 和 valid mask (H,W)"""
    fname = f"{frame_id:04d}"
    depth_path = os.path.join(raw_root, "depth", f"{fname}_depth.tiff")
    occ_path = os.path.join(raw_root, "occlusions", f"{fname}_occlusion.png")

    if not os.path.exists(depth_path) or not os.path.exists(occ_path):
        print(f"[WARN] depth/occ not found for frame {frame_id:04d}, skip.")
        return None, None

    depth_img = np.array(Image.open(depth_path))  # uint16
    occ_img = np.array(Image.open(occ_path))  # uint8

    depth_raw = torch.from_numpy(depth_img.astype(np.int32)).to(device)
    occ_u8 = torch.from_numpy(occ_img.astype(np.uint8)).to(device)

    depth_m = depth_raw.to(torch.float32) / 65535.0 * 0.1  # 0~0.1 m
    valid = (depth_raw > 0) & (occ_u8 == 0)
    return depth_m, valid


def frame_points_world(
    frame_id: int,
    raw_root: str,
    device: torch.device,
    dirs_cam: torch.Tensor,
    poses_cam2world: torch.Tensor,
    max_points: int = None,
):
    """返回某帧的世界系点云 (N,3)"""
    depth_m, valid = load_depth_and_mask(raw_root, frame_id, device)
    if depth_m is None:
        return None

    valid_mask = valid & (depth_m > 0)
    if valid_mask.sum() == 0:
        return None

    depth_z_valid = depth_m[valid_mask]  # (Nv,)
    dirs_valid = dirs_cam[valid_mask]  # (Nv,3)

    dz = dirs_valid[:, 2].clamp(min=1e-6)
    scale = depth_z_valid / dz
    pts_cam = dirs_valid * scale.unsqueeze(-1)  # (Nv,3)

    T = poses_cam2world[frame_id].to(device)
    R = T[:3, :3]
    t = T[:3, 3]
    pts_world = (R @ pts_cam.T + t.view(3, 1)).T  # (Nv,3)

    if max_points is not None and pts_world.shape[0] > max_points:
        idx = torch.randperm(pts_world.shape[0], device=device)[:max_points]
        pts_world = pts_world[idx]
    return pts_world


def write_ply(path, points: np.ndarray):
    N = points.shape[0]
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
    print(f"[INFO] Saved PLY with {N} points to {path}")


def parse_frame_ids_from_rawdepth(rawdepth_dir: str):
    """从 test/raw-depth/*.npy.gz 提取帧号"""
    files = [
        f for f in os.listdir(rawdepth_dir) if f.endswith(".npy.gz") and not f.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"No .npy.gz files found in {rawdepth_dir}")
    files = sorted(files)
    frame_ids = []
    for f in files:
        # 假设名字形如 0010.npy.gz
        m = re.search(r"(\d+)", f)
        if m:
            frame_ids.append(int(m.group(1)))
        else:
            print(f"[WARN] Cannot parse frame id from filename: {f}, skip.")
    frame_ids = sorted(set(frame_ids))
    return frame_ids


def main():
    parser = argparse.ArgumentParser(
        description="Build GT point cloud only on frames used by ns-render raw-depth."
    )
    parser.add_argument("--seq", type=str, required=True, help="Sequence name, e.g. c1_descending_t2_v2")
    parser.add_argument(
        "--raw-root-base",
        type=str,
        default="/data1_ycao/chua/projects/cdTeacher/data_raw",
        help="Base dir for raw data.",
    )
    parser.add_argument(
        "--stage-d-base",
        type=str,
        default="/data1_ycao/chua/projects/cdTeacher/outputs/stage_D",
        help="Base dir for Stage D outputs.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=10000,
        help="Subsample per frame to at most this many GT points.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for Omni camera + depth loading.",
    )

    args = parser.parse_args()

    raw_root = os.path.join(args.raw_root_base, args.seq)
    stage_d_root = os.path.join(args.stage_d_base, args.seq)
    rawdepth_dir = os.path.join(stage_d_root, "test", "raw-depth")

    assert os.path.isdir(raw_root), f"RAW_ROOT not found: {raw_root}"
    assert os.path.isdir(rawdepth_dir), f"raw-depth dir not found: {rawdepth_dir}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    pose_path = os.path.join(raw_root, "pose.txt")
    poses_cam2world = load_poses_cam2world(pose_path)  # (N,4,4)
    num_frames = poses_cam2world.shape[0]
    print(f"[INFO] Loaded poses: {poses_cam2world.shape}")

    # OmniCamera + dirs_cam
    H = OMNI_INTRINSICS["height"]
    W = OMNI_INTRINSICS["width"]
    omni_cam = OmniCamera(OMNI_INTRINSICS).to(device)
    u_coords = torch.arange(W, device=device).view(1, -1).expand(H, W)
    v_coords = torch.arange(H, device=device).view(-1, 1).expand(H, W)
    with torch.no_grad():
        dirs_cam = omni_cam(u_coords, v_coords)  # (H,W,3)
    print("[INFO] dirs_cam:", dirs_cam.shape)

    # 从 raw-depth 名字解析出真正参与评估的帧
    frame_ids = parse_frame_ids_from_rawdepth(rawdepth_dir)
    # 安全裁剪一下，避免超出 pose 范围
    frame_ids = [fid for fid in frame_ids if fid < num_frames]
    print(f"[INFO] Using {len(frame_ids)} eval frames:", frame_ids[:10], "..." if len(frame_ids) > 10 else "")

    all_points = []
    for i, fid in enumerate(frame_ids, 1):
        pts_w = frame_points_world(
            fid,
            raw_root=raw_root,
            device=device,
            dirs_cam=dirs_cam,
            poses_cam2world=poses_cam2world,
            max_points=args.max_points_per_frame,
        )
        if pts_w is None:
            continue
        all_points.append(pts_w.cpu())
        if i % 5 == 0 or i == len(frame_ids):
            print(f"[{i}/{len(frame_ids)}] Frame {fid:04d}: {pts_w.shape[0]} pts")

    if not all_points:
        print("[WARN] No GT points collected, please check depth/masks.")
        return

    all_points_tensor = torch.cat(all_points, dim=0)
    pts_np = all_points_tensor.numpy().astype(np.float32)
    print(f"[INFO] GT points total: {pts_np.shape}")

    ply_path = os.path.join(stage_d_root, "gt_pointcloud_eval_frames.ply")
    write_ply(ply_path, pts_np)


if __name__ == "__main__":
    main()
