#!/usr/bin/env python
# stage_D_eval_rawdepth_vs_gt.py
#
# 使用 Stage A 的 transforms + Stage D 的 raw-depth 和对齐版 GT 点云，
# 在 CPU 上计算 NeRF raw-depth vs GT 的 Chamfer 距离（支持多组 alpha）。

import os
import re
import argparse
import gzip
import json
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def load_gt_pointcloud(ply_path: str, max_points: int = 300000):
    print(f"[INFO] Loading GT point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    print(f"[INFO] GT points total: {pts.shape}")
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        print(f"[INFO] GT subsampled: {pts.shape}")
    else:
        print("[INFO] GT subsampled: no subsampling (<= max_points)")
    return pts


def load_transforms_json(transforms_path: str):
    with open(transforms_path, "r") as f:
        meta = json.load(f)
    W = meta["w"]
    H = meta["h"]
    fx = meta["fl_x"]
    fy = meta["fl_y"]
    cx = meta["cx"]
    cy = meta["cy"]
    frames = meta["frames"]
    # 按原顺序收集 cam2world
    poses_c2w = []
    for fr in frames:
        T = np.array(fr["transform_matrix"], dtype=np.float32)
        poses_c2w.append(T)
    poses_c2w = np.stack(poses_c2w, axis=0)  # (N,4,4)
    return (W, H, fx, fy, cx, cy), poses_c2w


def build_pinhole_dirs(W, H, fx, fy, cx, cy):
    """构建 pinhole 相机下每个像素的单位方向 (H,W,3)"""
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x, dtype=np.float32)

    dirs = np.stack([x, y, z], axis=-1)  # (H,W,3)
    norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs /= np.clip(norm, 1e-9, None)
    return dirs


def parse_frame_ids_from_rawdepth(rawdepth_dir: str):
    files = [
        f for f in os.listdir(rawdepth_dir) if f.endswith(".npy.gz") and not f.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"No .npy.gz files found in {rawdepth_dir}")
    files = sorted(files)
    frame_ids = []
    fname_map = {}
    for f in files:
        m = re.search(r"(\d+)", f)
        if m:
            fid = int(m.group(1))
            frame_ids.append(fid)
            fname_map[fid] = f
        else:
            print(f"[WARN] Cannot parse frame id from filename: {f}, skip.")
    frame_ids = sorted(set(frame_ids))
    return frame_ids, fname_map


def load_raw_depth_frame(path: str):
    """读取一个 raw-depth npy.gz, 返回 (H,W) float32"""
    with gzip.open(path, "rb") as f:
        d = np.load(f, allow_pickle=True)
    # d: (H,W,1)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]
    return d.astype(np.float32)


def build_nerf_points_from_rawdepth(
    rawdepth_dir: str,
    frame_ids,
    fname_map,
    dirs_cam_pinhole: np.ndarray,
    poses_c2w: np.ndarray,
    alpha: float,
    max_points_per_frame: int = 10000,
):
    """从 raw-depth (NeRF) 构建世界坐标点云 (N,3)"""
    H, W, _ = dirs_cam_pinhole.shape
    all_pts = []

    for idx, fid in enumerate(frame_ids, 1):
        fname = fname_map[fid]
        path = os.path.join(rawdepth_dir, fname)
        depth_raw = load_raw_depth_frame(path)  # (H,W)

        if depth_raw.shape != (H, W):
            raise ValueError(
                f"Depth shape mismatch for frame {fid:04d}: "
                f"got {depth_raw.shape}, expected {(H,W)}"
            )

        # 有效深度：>0 && 非 NaN
        valid = np.isfinite(depth_raw) & (depth_raw > 0)
        if not np.any(valid):
            print(f"[WARN] No valid depth in frame {fid:04d}, skip.")
            continue

        d_valid = depth_raw[valid] * alpha  # 变成米
        dirs_valid = dirs_cam_pinhole[valid]  # (Nv,3)
        pts_cam = dirs_valid * d_valid[:, None]  # (Nv,3)

        T_c2w = poses_c2w[fid]  # (4,4)
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]
        pts_world = (pts_cam @ R.T) + t[None, :]  # (Nv,3)

        if max_points_per_frame is not None and pts_world.shape[0] > max_points_per_frame:
            choice = np.random.choice(pts_world.shape[0], max_points_per_frame, replace=False)
            pts_world = pts_world[choice]

        all_pts.append(pts_world)

        if idx % 5 == 0 or idx == len(frame_ids):
            print(f"[{idx}/{len(frame_ids)}] Frame {fid:04d}: {pts_world.shape[0]} pts")

    if not all_pts:
        print("[WARN] No NeRF points collected.")
        return None

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    print(f"[INFO] NeRF points total: {pts.shape}")
    return pts


def chamfer_stats(pts_a: np.ndarray, pts_b: np.ndarray, label_a="A", label_b="B"):
    """用 KDTree 计算双向最近邻误差并打印统计"""
    print("Building KD-trees...")
    tree_a = KDTree(pts_a)
    tree_b = KDTree(pts_b)

    print(f"Query {label_a} → {label_b} ...")
    d_a2b, _ = tree_b.query(pts_a, k=1)
    d_a2b = d_a2b.squeeze(1)

    print(f"Query {label_b} → {label_a} ...")
    d_b2a, _ = tree_a.query(pts_b, k=1)
    d_b2a = d_b2a.squeeze(1)

    def summarize(d, name):
        d = d.astype(np.float64)
        mean = float(np.mean(d))
        median = float(np.median(d))
        p95 = float(np.percentile(d, 95))
        dmax = float(np.max(d))
        print(f"\n{name} stats (meters):")
        print(f"  mean   : {mean}")
        print(f"  median : {median}")
        print(f"  95%ile : {p95}")
        print(f"  max    : {dmax}")
        return mean

    mean_a2b = summarize(d_a2b, f"{label_a} → {label_b}")
    mean_b2a = summarize(d_b2a, f"{label_b} → {label_a}")
    chamfer = mean_a2b + mean_b2a
    print(f"\nChamfer-like distance (mean sum): {chamfer} (meters)")
    return mean_a2b, mean_b2a, chamfer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NeRF raw-depth vs GT point cloud using aligned eval frames."
    )
    parser.add_argument("--seq", type=str, required=True, help="Sequence name, e.g. c1_descending_t2_v2")
    parser.add_argument(
        "--stage-a-base",
        type=str,
        default="/data1_ycao/chua/projects/cdTeacher/outputs/stage_A",
        help="Base dir for Stage A outputs (transforms.json).",
    )
    parser.add_argument(
        "--stage-d-base",
        type=str,
        default="/data1_ycao/chua/projects/cdTeacher/outputs/stage_D",
        help="Base dir for Stage D outputs (GT ply + raw-depth).",
    )
    parser.add_argument(
        "--max-gt-points",
        type=int,
        default=300000,
        help="Subsample GT point cloud to this many points.",
    )
    parser.add_argument(
        "--max-nerf-points",
        type=int,
        default=300000,
        help="Subsample NeRF point cloud to this many points.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=10000,
        help="Max NeRF points per frame (before global subsample).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1],
        help="List of alpha (meters per unit raw-depth) to evaluate.",
    )

    args = parser.parse_args()

    stage_a_root = os.path.join(args.stage_a_base, args.seq)
    stage_d_root = os.path.join(args.stage_d_base, args.seq)

    transforms_path = os.path.join(stage_a_root, "transforms.json")
    gt_ply_path = os.path.join(stage_d_root, "gt_pointcloud_eval_frames.ply")
    rawdepth_dir = os.path.join(stage_d_root, "test", "raw-depth")

    assert os.path.isfile(transforms_path), f"transforms.json not found: {transforms_path}"
    assert os.path.isfile(gt_ply_path), f"GT PLY not found: {gt_ply_path}"
    assert os.path.isdir(rawdepth_dir), f"raw-depth dir not found: {rawdepth_dir}"

    # 1. 加载 GT 点云
    gt_pts = load_gt_pointcloud(gt_ply_path, max_points=args.max_gt_points)

    # 2. 加载 transforms.json（pinhole 相机参数 + poses）
    (W, H, fx, fy, cx, cy), poses_c2w = load_transforms_json(transforms_path)
    print(f"[INFO] Pinhole intrinsics: W={W}, H={H}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"[INFO] Loaded {poses_c2w.shape[0]} camera poses from transforms.json")

    # 3. 构建 pinhole 射线方向
    dirs_cam_pinhole = build_pinhole_dirs(W, H, fx, fy, cx, cy)  # (H,W,3)

    # 4. raw-depth 使用的帧（确保和 Stage A 的 frame index 对齐）
    frame_ids, fname_map = parse_frame_ids_from_rawdepth(rawdepth_dir)
    frame_ids = [fid for fid in frame_ids if fid < poses_c2w.shape[0]]
    print(f"[INFO] Found {len(frame_ids)} raw-depth frames:", frame_ids[:10], "..." if len(frame_ids) > 10 else "")

    # 对 GT/NeRF 再做一次统一 subsample（避免 KDTree 太大）
    if gt_pts.shape[0] > args.max_gt_points:
        idx_gt = np.random.choice(gt_pts.shape[0], args.max_gt_points, replace=False)
        gt_eval = gt_pts[idx_gt]
    else:
        gt_eval = gt_pts
    print(f"[INFO] GT used for KDTree: {gt_eval.shape}")

    # 5. 对每个 alpha 做一次完整评估
    results = []
    for alpha in args.alphas:
        print("\n" + "=" * 20 + f" α = {alpha} m " + "=" * 20)

        nerf_pts = build_nerf_points_from_rawdepth(
            rawdepth_dir=rawdepth_dir,
            frame_ids=frame_ids,
            fname_map=fname_map,
            dirs_cam_pinhole=dirs_cam_pinhole,
            poses_c2w=poses_c2w,
            alpha=alpha,
            max_points_per_frame=args.max_points_per_frame,
        )
        if nerf_pts is None:
            print("[WARN] No NeRF points, skip this alpha.")
            continue

        if nerf_pts.shape[0] > args.max_nerf_points:
            idx_nf = np.random.choice(nerf_pts.shape[0], args.max_nerf_points, replace=False)
            nerf_eval = nerf_pts[idx_nf]
        else:
            nerf_eval = nerf_pts
        print(f"[INFO] NeRF used for KDTree: {nerf_eval.shape}")

        mean_n2g, mean_g2n, chamfer = chamfer_stats(
            nerf_eval, gt_eval, label_a="NeRF", label_b="GT"
        )
        results.append((alpha, mean_n2g, mean_g2n, chamfer))
        print(f"(Using ALPHA = {alpha} m per unit raw-depth)")

    if results:
        print("\n================ All α results ================")
        for alpha, m_n2g, m_g2n, chamfer in results:
            print(
                f"α = {alpha}: Chamfer = {chamfer:.6f} m "
                f"(NeRF→GT mean = {m_n2g:.6f},  GT→NeRF mean = {m_g2n:.6f})"
            )
    else:
        print("[WARN] No valid results produced.")


if __name__ == "__main__":
    main()
