#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_models_index.py

扫描 models 目录下的 .ply 文件，生成一个 index.json，
用于 web viewer 显示有哪些可选的点云模型。

默认目录：
  /data1_ycao/chua/projects/cdTeacher/web-viewer/models

也可以通过 --models-dir 手动指定。
"""

import os
import re
import json
import argparse


DEFAULT_MODELS_DIR = "/data1_ycao/chua/projects/cdTeacher/web-viewer/models"


def parse_model_filename(filename: str):
    """
    根据文件名推断 seq / kind / alpha 等信息。

    约定：
      <seq>_gt_full.ply
      <seq>_nerf_rawdepth_alpha0.10m.ply
    """
    name, ext = os.path.splitext(filename)
    if ext.lower() != ".ply":
        return None

    # 1) GT: <seq>_gt_full.ply
    m_gt = re.match(r"^(?P<seq>.+)_gt_full$", name)
    if m_gt:
        seq = m_gt.group("seq")
        return {
            "seq": seq,
            "kind": "gt",
            "alpha": None,
            "label": f"{seq} (GT full)"
        }

    # 2) NeRF raw-depth: <seq>_nerf_rawdepth_alpha0.10m.ply
    m_nerf = re.match(
        r"^(?P<seq>.+)_nerf_rawdepth_alpha(?P<alpha>[0-9.]+)m$",
        name
    )
    if m_nerf:
        seq = m_nerf.group("seq")
        alpha = float(m_nerf.group("alpha"))
        return {
            "seq": seq,
            "kind": "nerf_rawdepth",
            "alpha": alpha,
            "label": f"{seq} (NeRF raw-depth, α={alpha:.2f} m)"
        }

    # 3) 其它未知命名，也保留，但打个 "other"
    return {
        "seq": name,
        "kind": "other",
        "alpha": None,
        "label": name,
    }


def build_index(models_dir: str):
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    files = sorted(os.listdir(models_dir))
    models = []

    for fname in files:
        if not fname.lower().endswith(".ply"):
            continue
        meta = parse_model_filename(fname)
        if meta is None:
            continue
        meta["filename"] = fname
        models.append(meta)

    index = {
        "models_dir": models_dir,
        "num_models": len(models),
        "models": models,
    }

    out_path = os.path.join(models_dir, "index.json")
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[WRITE] Wrote index.json with {len(models)} models to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan models dir for .ply files and write index.json"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Directory containing .ply models (default: {DEFAULT_MODELS_DIR})",
    )
    args = parser.parse_args()
    build_index(args.models_dir)


if __name__ == "__main__":
    main()
