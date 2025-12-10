#!/usr/bin/env python3
"""Generate paper figures for the geometry-centric colonoscopy study."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402
import imageio.v3 as iio  # noqa: E402


ROOT = Path("/data1_ycao/chua/projects/cdTeacher")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def _load_image(path: Path):
    return iio.imread(path)


def _load_depth_meters(path: Path) -> np.ndarray:
    depth_raw = iio.imread(path).astype(np.float32)
    return depth_raw / 65535.0 * 0.1


def _load_occlusion(path: Path) -> np.ndarray:
    mask = iio.imread(path).astype(np.float32)
    return mask / mask.max()


def _load_ply_points(path: Path, voxel: Optional[float] = 0.0015, max_points: int = 400000) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    if voxel:
        down = cloud.voxel_down_sample(voxel)
        if len(down.points) > 0:
            cloud = down
    pts = np.asarray(cloud.points)
    if len(pts) == 0:
        # Fallback: reload without downsampling
        cloud = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(cloud.points)
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def _row_limits(point_sets: List[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    point_sets = [p for p in point_sets if len(p) > 0]
    if not point_sets:
        return ((-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1))
    anchor = point_sets[0]
    all_pts = np.concatenate(point_sets, axis=0)
    center = anchor.mean(axis=0)
    span = np.abs(all_pts - center).max() * 1.1
    if span == 0:
        span = 0.1
    return (
        (center[0] - span, center[0] + span),
        (center[1] - span, center[1] + span),
        (center[2] - span, center[2] + span),
    )


def _scatter_ax(ax, pts: np.ndarray, color: str, title: str, limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]], size: float = 3.0):
    if len(pts) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=size, c=color, alpha=0.9, linewidths=0)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.view_init(elev=18, azim=120)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10)


def _load_transforms(seq_root: Path) -> Dict[str, np.ndarray]:
    meta = json.loads((seq_root / "transforms.json").read_text())
    frames = {}
    for fr in meta["frames"]:
        stem = Path(fr["file_path"]).stem
        frames[stem] = np.asarray(fr["transform_matrix"], dtype=np.float32)
    intr = {k: meta[k] for k in ["fl_x", "fl_y", "cx", "cy", "w", "h"]}
    return {"frames": frames, "intrinsics": intr}


def _build_nerf_cloud(seq: str, alpha: float, max_points: int = 250000) -> np.ndarray:
    seq_root = ROOT / "outputs/stage_A" / seq
    depth_root = ROOT / "outputs/stage_D" / seq
    meta = _load_transforms(seq_root)
    frames = meta["frames"]
    intr = meta["intrinsics"]
    fx, fy, cx, cy = intr["fl_x"], intr["fl_y"], intr["cx"], intr["cy"]
    h, w = int(intr["h"]), int(intr["w"])
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dirs = np.stack(((grid_x - cx) / fx, (grid_y - cy) / fy, np.ones_like(grid_x)), axis=-1).reshape(-1, 3)

    rng = np.random.default_rng(42)
    all_pts: List[np.ndarray] = []
    for split in ["train", "test"]:
        depth_dir = depth_root / split / "raw-depth"
        if not depth_dir.exists():
            continue
        for depth_file in sorted(depth_dir.glob("*.npy.gz")):
            stem = depth_file.name.split(".")[0]
            if stem.endswith(".npy"):
                stem = stem[:-4]
            if stem not in frames:
                continue
            depth = np.load(gzip.open(depth_file)).reshape(-1)
            depth_m = depth * alpha
            valid = depth_m > 0
            if not np.any(valid):
                continue
            pts_cam = dirs[valid] * depth_m[valid][:, None]
            c2w = frames[stem]
            R = c2w[:3, :3]
            t = c2w[:3, 3]
            pts_world = pts_cam @ R.T + t
            all_pts.append(pts_world)
    if not all_pts:
        return np.empty((0, 3))
    pts = np.concatenate(all_pts, axis=0)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    return pts


def fig_overview():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axis("off")
    gt_color = "#3b8b3b"
    sfm_color = "#4c72b0"
    nerf_color = "#c44e52"

    def box(x, y, w, h, text, color="black", face="#f7f7f7", lw=1.5):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            linewidth=lw,
            edgecolor=color,
            facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5)
        return rect

    def arrow(p1, p2, color="black"):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color=color,
            )
        )

    # Input thumbnail
    img = _load_image(ROOT / "data_raw/c1_descending_t2_v2/rgb/0000.png")
    ax.imshow(img, extent=(-0.02, 0.18, 0.32, 0.68), zorder=3)
    box(0.0, 0.3, 0.2, 0.4, "Fisheye colonoscopy\nvideos (C3VDv2)")

    # Rays
    rays_box = box(0.24, 0.36, 0.2, 0.28, "Calibrated\nomnidirectional rays\n(OCamCalib)")
    arrow((0.2, 0.5), (0.24, 0.5))

    # GT path (top)
    gt1 = box(0.5, 0.6, 0.2, 0.2, "GT depth\n+ occlusion masks", color=gt_color)
    gt2 = box(0.74, 0.6, 0.2, 0.2, "GT 3D point clouds\n(world coords)", color=gt_color)
    arrow((0.44, 0.5), (0.5, 0.7), color=gt_color)
    arrow((0.7, 0.7), (0.74, 0.7), color=gt_color)

    # Rectified node (shared)
    rect_box = box(0.5, 0.32, 0.2, 0.18, "Rectified\npinhole images", color="black")
    arrow((0.44, 0.5), (0.5, 0.41))

    # SfM branch
    sfm1 = box(0.74, 0.42, 0.2, 0.14, "SfM (COLMAP)", color=sfm_color)
    sfm2 = box(0.98, 0.42, 0.2, 0.14, "SfM dense\npoint cloud", color=sfm_color)
    arrow((0.7, 0.41), (0.74, 0.47), color=sfm_color)
    arrow((0.94, 0.49), (0.98, 0.49), color=sfm_color)

    # NeRF branch
    nerf1 = box(0.74, 0.18, 0.2, 0.14, "NeRF (nerfacto)", color=nerf_color)
    nerf2 = box(0.98, 0.18, 0.2, 0.14, "NeRF point cloud\n/ depth", color=nerf_color)
    arrow((0.7, 0.41), (0.74, 0.25), color=nerf_color)
    arrow((0.94, 0.25), (0.98, 0.25), color=nerf_color)

    # Evaluation / teacher setting
    eval_rect = FancyBboxPatch(
        (1.24, 0.14),
        0.42,
        0.68,
        boxstyle="round,pad=0.025",
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(eval_rect)
    ax.text(1.45, 0.85, "Geometry-centric benchmark / teacher setting", ha="center", va="center", fontsize=10.5)
    chamfer_sfm = box(1.3, 0.52, 0.3, 0.14, "Chamfer distance:\nGT vs SfM", face="#fefefe")
    chamfer_nerf = box(1.3, 0.3, 0.3, 0.14, "Chamfer distance:\nGT vs NeRF", face="#fefefe")
    arrow((0.94, 0.7), (1.3, 0.59), color=gt_color)
    arrow((1.18, 0.49), (1.3, 0.59), color=sfm_color)
    arrow((1.18, 0.25), (1.3, 0.37), color=nerf_color)
    arrow((0.94, 0.7), (1.3, 0.37), color=gt_color)

    ax.set_xlim(-0.05, 1.72)
    ax.set_ylim(0.05, 0.95)
    fig.savefig(FIG_DIR / "fig_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_rectification():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    fisheye = _load_image(ROOT / "data_raw/c1_descending_t2_v2/rgb/0000.png")
    rectified = _load_image(ROOT / "outputs/stage_A/c1_descending_t2_v2/undistorted/rgb/0000.png")
    titles = ["Fisheye input", "Ray mapping", "Rectified pinhole"]
    axes[0].imshow(fisheye)
    axes[0].axis("off")
    axes[0].set_title(titles[0], fontsize=11)

    ax = axes[1]
    ax.axis("off")
    ax.set_title(titles[1], fontsize=11)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 0.6)
    # Camera center and rays
    ax.plot(0, 0, "ko")
    for ang in np.linspace(-0.8, 0.8, 7):
        ax.plot([0, np.cos(ang)], [0, np.sin(ang) * 0.6], color="gray", lw=1)
    # Image plane
    ax.add_patch(plt.Rectangle((0.4, -0.4), 0.25, 0.8, fill=False, lw=1.5))
    ax.text(0.525, 0.45, "Virtual pinhole\nimage plane", ha="center", va="center", fontsize=9)
    ax.text(0.6, -0.5, r"Intrinsics: $(f_x, f_y, c_x, c_y)$", ha="center", fontsize=9)
    ax.text(0, 0.1, "Omnidirectional\nrays", ha="center", va="center", fontsize=9)
    ax.arrow(0.15, 0.0, 0.4, 0.0, head_width=0.03, head_length=0.03, fc="k", ec="k", length_includes_head=True)
    ax.text(0.35, 0.05, "Ray mapping", fontsize=9, ha="center")
    ax.annotate("FOV", xy=(0.15, 0.0), xytext=(-0.25, 0.25), arrowprops=dict(arrowstyle="->"))

    axes[2].imshow(rectified)
    axes[2].axis("off")
    axes[2].set_title(titles[2], fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rectification.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_alpha_sweep():
    alphas = [0.05, 0.10, 0.20]
    sequences = {
        "c1 descending t2 v2": [0.023666, 0.021962, 0.022927],
        "c2 ascending t1 v1": [0.025430, 0.022136, 0.024650],
        "c2 rectum t4 v3": [0.025053, 0.039836, 0.073917],
        "c2 transverse1 t3 v2": [0.029808, 0.027120, 0.028640],
    }
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for (label, vals), m in zip(sequences.items(), markers):
        ax.plot(alphas, vals, marker=m, label=label)
    ax.set_xlabel("Depth scale α (m per unit raw-depth)", fontsize=10)
    ax.set_ylabel("Chamfer distance (m)", fontsize=10)
    ax.set_xticks(alphas)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_alpha_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_cloud_compare(rectum_cloud: np.ndarray):
    fig = plt.figure(figsize=(9, 6))
    gs = fig.add_gridspec(2, 3, wspace=0.05, hspace=0.2)

    # Row 1: c1_descending_t2_v2
    c1_gt = _load_ply_points(ROOT / "web-viewer/models/c1_descending_t2_v2_gt_full.ply", voxel=None, max_points=300000)
    c1_sfm = _load_ply_points(ROOT / "outputs/stage_B/c1_descending_t2_v2/colmap_dense/fused.ply", voxel=0.002)
    c1_nerf = _load_ply_points(ROOT / "web-viewer/models/c1_descending_t2_v2_nerf_rawdepth_alpha0.10m.ply", voxel=None, max_points=300000)
    limits_c1 = _row_limits([c1_gt, c1_sfm, c1_nerf])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    _scatter_ax(ax, c1_gt, "gray", "GT point cloud", limits_c1, size=2.0)
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    _scatter_ax(ax, c1_sfm, "#4c72b0", "SfM dense reconstruction", limits_c1, size=1.0)
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    _scatter_ax(ax, c1_nerf, "#c44e52", "NeRF (α=0.10)", limits_c1, size=1.5)
    fig.text(0.5, 0.96, "c1_descending_t2_v2 (easier sequence)", ha="center", fontsize=11)

    # Row 2: c2_rectum_t4_v3
    c2_gt = _load_ply_points(ROOT / "web-viewer/models/c2_rectum_t4_v3_gt_full.ply")
    c2_nerf = rectum_cloud
    limits_c2 = _row_limits([c2_gt, c2_nerf])
    ax = fig.add_subplot(gs[1, 0], projection="3d")
    _scatter_ax(ax, c2_gt, "gray", "GT point cloud", limits_c2)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(0.5, 0.5, "SfM dense reconstruction\n(unavailable / failed)", ha="center", va="center", fontsize=10)
    ax.set_title("SfM dense reconstruction", fontsize=10)
    ax = fig.add_subplot(gs[1, 2], projection="3d")
    _scatter_ax(ax, c2_nerf, "#c44e52", "NeRF (α=0.05)", limits_c2)
    fig.text(0.5, 0.49, "c2_rectum_t4_v3 (noisy rectum sequence)", ha="center", fontsize=11)

    fig.savefig(FIG_DIR / "fig_cloud_compare.pdf", bbox_inches="tight")
    plt.close(fig)


def _crop_cloud_near_pose(cloud: np.ndarray, pose: np.ndarray, radius: float = 0.06, max_points: int = 60000) -> np.ndarray:
    if len(cloud) == 0:
        return cloud
    cam_pos = pose[:3, 3]
    dists = np.linalg.norm(cloud - cam_pos[None, :], axis=1)
    subset = cloud[dists < radius]
    if len(subset) == 0:
        subset = cloud[np.random.default_rng(1).choice(len(cloud), size=min(max_points, len(cloud)), replace=False)]
    elif len(subset) > max_points:
        subset = subset[np.random.default_rng(2).choice(len(subset), size=max_points, replace=False)]
    return subset


def fig_rectum_failure(rectum_cloud: np.ndarray):
    frames = [0, 120, 240]
    cols = ["Fisheye RGB", "Rectified RGB", "GT depth", "Occlusion mask", "NeRF geometry (zoomed)"]
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(len(frames), len(cols), wspace=0.05, hspace=0.2)

    meta = _load_transforms(ROOT / "outputs/stage_A/c2_rectum_t4_v3")
    frames_tf = meta["frames"]

    for r, frame_id in enumerate(frames):
        stem = f"{frame_id:04d}"
        fisheye = _load_image(ROOT / f"data_raw/c2_rectum_t4_v3/rgb/{stem}.png")
        rectified = _load_image(ROOT / f"outputs/stage_A/c2_rectum_t4_v3/undistorted/rgb/{stem}.png")
        depth_m = _load_depth_meters(ROOT / f"data_raw/c2_rectum_t4_v3/depth/{stem}_depth.tiff")
        mask = _load_occlusion(ROOT / f"data_raw/c2_rectum_t4_v3/occlusions/{stem}_occlusion.png")
        cloud_crop = _crop_cloud_near_pose(rectum_cloud, frames_tf[stem], radius=0.07)

        row_imgs = [fisheye, rectified, depth_m, mask]
        cmaps = [None, None, "magma", "gray"]
        for c, (img, cmap) in enumerate(zip(row_imgs, cmaps)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(img, cmap=cmap)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=10)

        ax3d = fig.add_subplot(gs[r, 4], projection="3d")
        lims = _row_limits([cloud_crop])
        _scatter_ax(ax3d, cloud_crop, "#c44e52", cols[4], lims)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rectum_failure.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_overview()
    fig_rectification()
    fig_alpha_sweep()
    rectum_cloud = _build_nerf_cloud("c2_rectum_t4_v3", alpha=0.05)
    fig_cloud_compare(rectum_cloud)
    fig_rectum_failure(rectum_cloud)


if __name__ == "__main__":
    main()
