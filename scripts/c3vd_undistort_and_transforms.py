#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
c3vd_stageA_undistort_and_transforms.py

Usage example:
    python c3vd_stageA_undistort_and_transforms.py --seq c1_descending_t2_v2
"""

import os
import json
import math
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# -------------------------
# Camera intrinsics (consistent with notebook)
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

        # 多项式系数 pol(r) = a0 + a1*r + ...
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

        # 预算 affine 矩阵的逆: [x';y'] = A^{-1} * ([u-cx; v-cy])
        A = torch.tensor([[self.c, self.d], [self.e, 1.0]], dtype=torch.float32)
        A_inv = torch.inverse(A)
        self.register_buffer("A_inv", A_inv)

    def forward(self, u, v):
        """
        u, v: (...,) 像素坐标 (float32)，可以是任意 shape
        返回: (..., 3) 单位方向向量 (x,y,z) in camera frame
        """
        u = torch.as_tensor(u, dtype=torch.float32, device=self.pol.device)
        v = torch.as_tensor(v, dtype=torch.float32, device=self.pol.device)

        # 平移到主点
        x_img = u - self.cx
        y_img = v - self.cy

        # 逆 affine 校正
        x_prime = self.A_inv[0, 0] * x_img + self.A_inv[0, 1] * y_img
        y_prime = self.A_inv[1, 0] * x_img + self.A_inv[1, 1] * y_img

        r = torch.sqrt(x_prime**2 + y_prime**2)

        # pol(r) = a0 + a1*r + a2*r^2 + ...
        powers = torch.stack(
            [r**i for i in range(self.pol.shape[0])], dim=0
        )  # (deg+1, ...)
        z = (self.pol.view(-1, *([1] * (powers.ndim - 1))) * powers).sum(dim=0)

        dir_cam = torch.stack([x_prime, y_prime, z], dim=-1)
        dir_norm = dir_cam / torch.linalg.norm(dir_cam, dim=-1, keepdim=True).clamp(
            min=1e-9
        )
        return dir_norm


# -------------------------
# pose.txt -> (N,4,4) T_cam2world  (mm → m)
# -------------------------

def load_poses_cam2world(raw_root: str) -> torch.Tensor:
    pose_path = os.path.join(raw_root, "pose.txt")
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
            mat_raw = vals.reshape(4, 4)  # 这是 T^T 的形状
            T = mat_raw.T  # 转置成标准 T_cam2world

            # 把平移从 mm → m
            T[0:3, 3] /= 1000.0

            poses_list.append(T)

    poses_cam2world = torch.from_numpy(np.stack(poses_list, axis=0))  # (N,4,4)
    return poses_cam2world


# -------------------------
# fisheye -> pinhole forward splat
# -------------------------

def build_pinhole_intrinsics(W_p=960, H_p=720, fov_deg=90.0):
    fov_rad = math.radians(fov_deg)
    fx = (W_p / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx_p = W_p / 2.0
    cy_p = H_p / 2.0
    return {
        "width": W_p,
        "height": H_p,
        "fx": fx,
        "fy": fy,
        "cx": cx_p,
        "cy": cy_p,
        "fov_deg": fov_deg,
    }


def precompute_fisheye_to_pinhole_maps(omni_cam: OmniCamera, pinhole_intrinsics, device):
    """生成 fisheye 像素 → pinhole 像素坐标 (u_p_map, v_p_map)"""
    H_f = omni_cam.height
    W_f = omni_cam.width
    fx = pinhole_intrinsics["fx"]
    fy = pinhole_intrinsics["fy"]
    cx_p = pinhole_intrinsics["cx"]
    cy_p = pinhole_intrinsics["cy"]

    u_coords = torch.arange(W_f, device=device).view(1, -1).expand(H_f, W_f)
    v_coords = torch.arange(H_f, device=device).view(-1, 1).expand(H_f, W_f)

    with torch.no_grad():
        dirs_cam = omni_cam(u_coords, v_coords)  # (H_f, W_f, 3)

    x_f = dirs_cam[..., 0]
    y_f = dirs_cam[..., 1]
    z_f = dirs_cam[..., 2].clamp(min=1e-6)

    u_p_map = fx * (x_f / z_f) + cx_p
    v_p_map = fy * (y_f / z_f) + cy_p

    return u_p_map, v_p_map


def fisheye_to_pinhole_rgb(
    rgb_fisheye: torch.Tensor,
    u_p_map: torch.Tensor,
    v_p_map: torch.Tensor,
    W_p: int,
    H_p: int,
):
    """
    rgb_fisheye: (H_f, W_f, 3), float32, [0,1]
    u_p_map, v_p_map: (H_f, W_f)
    返回:
      rgb_pinhole: (H_p, W_p, 3)
    """
    device = rgb_fisheye.device
    H_f, W_f, _ = rgb_fisheye.shape

    u_p_map = u_p_map.to(device)
    v_p_map = v_p_map.to(device)

    # 展平成一维
    u_flat = u_p_map.reshape(-1)
    v_flat = v_p_map.reshape(-1)
    rgb_flat = rgb_fisheye.reshape(-1, 3)

    # 只保留落在 pinhole 范围附近的
    valid = (u_flat >= -1) & (u_flat <= W_p) & (v_flat >= -1) & (v_flat <= H_p)
    u_flat = u_flat[valid]
    v_flat = v_flat[valid]
    rgb_flat = rgb_flat[valid]

    u0 = torch.floor(u_flat).to(torch.long)
    v0 = torch.floor(v_flat).to(torch.long)
    du = u_flat - u0.to(torch.float32)
    dv = v_flat - v0.to(torch.float32)

    u1 = u0 + 1
    v1 = v0 + 1

    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv

    rgb_out = torch.zeros((H_p, W_p, 3), dtype=torch.float32, device=device)
    w_out = torch.zeros((H_p, W_p), dtype=torch.float32, device=device)

    def accumulate(u, v, w):
        mask = (u >= 0) & (u < W_p) & (v >= 0) & (v < H_p) & (w > 0)
        if mask.sum() == 0:
            return
        u_sel = u[mask]
        v_sel = v[mask]
        w_sel = w[mask]
        rgb_sel = rgb_flat[mask]

        idx = v_sel * W_p + u_sel
        rgb_out_flat = rgb_out.reshape(-1, 3)
        w_out_flat = w_out.reshape(-1)

        rgb_out_flat.index_add_(0, idx, rgb_sel * w_sel.unsqueeze(-1))
        w_out_flat.index_add_(0, idx, w_sel)

    accumulate(u0, v0, w00)
    accumulate(u1, v0, w10)
    accumulate(u0, v1, w01)
    accumulate(u1, v1, w11)

    w_out_clamped = w_out.clamp(min=1e-6).unsqueeze(-1)
    rgb_norm = rgb_out / w_out_clamped
    return rgb_norm


def load_rgb_frame(raw_root: str, frame_id: int, device):
    fname = f"{frame_id:04d}"
    rgb_path = os.path.join(raw_root, "rgb", f"{fname}.png")
    img = Image.open(rgb_path).convert("RGB")
    arr = np.array(img)
    rgb = torch.from_numpy(arr).to(device=device, dtype=torch.float32) / 255.0
    return rgb


# -------------------------
# 写 transforms.json （nerfstudio-data）
# -------------------------

def write_transforms_json(poses_cam2world: torch.Tensor, out_root: str, pinhole_intrinsics):
    poses_c2w = poses_cam2world.to(torch.float32).cpu().numpy()
    W_p = pinhole_intrinsics["width"]
    H_p = pinhole_intrinsics["height"]
    fx = pinhole_intrinsics["fx"]
    fy = pinhole_intrinsics["fy"]
    cx_p = pinhole_intrinsics["cx"]
    cy_p = pinhole_intrinsics["cy"]

    ns_transform = {
        "camera_model": "PINHOLE",
        "w": W_p,
        "h": H_p,
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(cx_p),
        "cy": float(cy_p),
        "frames": [],
    }

    num_frames = poses_c2w.shape[0]
    for fid in range(num_frames):
        T_c2w = poses_c2w[fid]
        frame = {
            "file_path": f"./undistorted/rgb/{fid:04d}.png",
            "transform_matrix": T_c2w.tolist(),
        }
        ns_transform["frames"].append(frame)

    out_path = os.path.join(out_root, "transforms.json")
    with open(out_path, "w") as f:
        json.dump(ns_transform, f, indent=2)
    print("[StageA] Wrote Nerfstudio transforms to", out_path)


# -------------------------
# main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True,
                        help="序列名，例如 c1_descending_t2_v2")
    parser.add_argument(
        "--data-raw-root",
        type=str,
        default="/data1_ycao/chua/projects/cdTeacher/data_raw",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="/data1_ycao/chua/projects/cdTeacher/outputs/stage_A",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fov-deg", type=float, default=90.0)

    args = parser.parse_args()

    raw_root = os.path.join(args.data_raw_root, args.seq)
    out_root = os.path.join(args.out_root, args.seq)
    undist_root = os.path.join(out_root, "undistorted")
    undist_rgb_root = os.path.join(undist_root, "rgb")

    os.makedirs(out_root, exist_ok=True)
    os.makedirs(undist_rgb_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[StageA] Using device:", device)
    print("[StageA] RAW_ROOT:", raw_root)
    print("[StageA] OUT_ROOT:", out_root)

    # 保存 omni intrinsics
    with open(os.path.join(out_root, "camera_omni.json"), "w") as f:
        json.dump(OMNI_INTRINSICS, f, indent=2)

    # 1. 加载 pose
    poses_cam2world = load_poses_cam2world(raw_root)
    num_frames = poses_cam2world.shape[0]
    print("[StageA] Loaded poses:", poses_cam2world.shape)

    # 2. 初始化相机 + 预计算投影映射
    omni_cam = OmniCamera(OMNI_INTRINSICS).to(device)
    pinhole_intrinsics = build_pinhole_intrinsics(
        W_p=args.width, H_p=args.height, fov_deg=args.fov_deg
    )
    with open(os.path.join(undist_root, "camera_pinhole.json"), "w") as f:
        json.dump(pinhole_intrinsics, f, indent=2)
    print("[StageA] Pinhole intrinsics:", pinhole_intrinsics)

    u_p_map, v_p_map = precompute_fisheye_to_pinhole_maps(
        omni_cam, pinhole_intrinsics, device
    )
    print(
        "[StageA] u_p range:",
        u_p_map.min().item(),
        u_p_map.max().item(),
        "| v_p range:",
        v_p_map.min().item(),
        v_p_map.max().item(),
    )

    # 3. 展平所有 RGB 帧
    for fid in tqdm(range(num_frames), desc="[StageA] Undistorting frames"):
        rgb_fisheye = load_rgb_frame(raw_root, fid, device)
        rgb_pinhole = fisheye_to_pinhole_rgb(
            rgb_fisheye, u_p_map, v_p_map, pinhole_intrinsics["width"], pinhole_intrinsics["height"]
        )
        rgb_np = (rgb_pinhole.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(
            np.uint8
        )
        img_out = Image.fromarray(rgb_np, mode="RGB")
        out_path = os.path.join(undist_rgb_root, f"{fid:04d}.png")
        img_out.save(out_path)

    # 4. 写 transforms.json
    write_transforms_json(poses_cam2world, out_root, pinhole_intrinsics)

    print("[StageA] Done for sequence:", args.seq)


if __name__ == "__main__":
    main()
