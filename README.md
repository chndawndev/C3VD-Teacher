# C3VD-Teacher

This project implements a pipeline for processing C3VD dataset sequences, training NeRF models using Nerfstudio, and evaluating the reconstructed geometry against Ground Truth (GT). It also includes a web-based viewer for visualizing point clouds.

## Project Structure

- **`data_raw/`**: Contains the raw input data for sequences.
- **`outputs/`**: Stores the outputs of various processing stages:
  - `stage_A/`: Preprocessing, camera poses, and undistortion.
  - `stage_B/`: Alignment (COLMAP vs GT).
  - `stage_C/`: Dense reconstruction evaluation.
  - `stage_D/`: NeRF depth rendering and evaluation.
- **`scripts/`**: Python and Shell scripts for the pipeline.
- **`web-viewer/`**: A web-based 3D viewer to visualize GT and NeRF point clouds.
- **`notebooks/`**: Jupyter notebooks for interactive analysis and step-by-step processing.

## Pipeline Stages

The pipeline is organized into stages, often corresponding to the notebooks or scripts:

1.  **Stage A**: Camera setup and Ground Truth preparation.
2.  **Stage B**: Alignment of COLMAP/SfM poses to GT.
3.  **Stage C**: Dense reconstruction vs GT evaluation.
4.  **Stage D**: NeRF training, depth rendering, and evaluation against GT.

## Scripts

Located in the `scripts/` directory:

-   `run_ns_train_seq.sh`: Train a Nerfstudio model (nerfacto) on a sequence.
-   `run_ns_render_raw_depth_seq.sh`: Render raw depth maps from a trained NeRF model.
-   `build_pointclouds_for_seq.py`: Generate PLY point clouds from GT data and NeRF raw depth.
-   `build_models_index_JSON.py`: Index generated PLY files for the web viewer.
-   `eval_rawdepth_vs_gt.py`: Evaluate NeRF raw depth against GT using Chamfer distance.
-   `c3vd_undistort_and_transforms.py`: Undistort fisheye images and generate transforms.json.
-   `build_gt_for_eval.py`: Build aligned GT point clouds for evaluation.

## Web Viewer

The `web-viewer/` directory contains a simple 3D viewer based on Three.js.

To use it:
1.  Generate point clouds using `scripts/build_pointclouds_for_seq.py`.
2.  Update the index using `scripts/build_models_index_JSON.py`.
3.  Start a local HTTP server in the `web-viewer` directory:
    ```bash
    cd web-viewer
    python -m http.server 8000
    ```
4.  Open `http://localhost:8000` in your browser.

## Requirements

-   Python 3.8+
-   Nerfstudio
-   PyTorch
-   NumPy, SciPy, Open3D, Pillow
-   Three.js (for web viewer, loaded via CDN)
