# CDTeacher: A Geometry-Centric 3D Reconstruction Pipeline for Fisheye Colonoscopy Videos (C3VDv2)

Project repo: https://github.com/chndawndev/C3VD-Teacher

CDTeacher is a geometry-centric benchmarking pipeline for **3D reconstruction from fisheye colonoscopy videos**, built on top of the **C3VDv2 dataset**.

The pipeline supports:

- Omnidirectional → Pinhole rectification  
- Ground-truth geometry reconstruction (depth + occlusion + calibrated rays)  
- SfM baseline (COLMAP)  
- NeRF baseline (Nerfstudio / nerfacto)  
- Geometry evaluation using Chamfer distance  
- Optional: Browser-based point cloud viewer  

This README explains:

1. **How to install dependencies**  
2. **How to configure the environment**  
3. **How to run the full pipeline (A→B→C→D)**  

> ⚠️ **Warning**  
> A simple `requirements.txt` cannot reproduce this project.  
> This pipeline depends on *strict CUDA, PyTorch, GCC, and Nerfstudio version matching*.  
> All installation instructions **must be followed exactly**.

---

# 1. Project Structure

```

cdTeacher/
│
├── data_raw/                           # Raw C3VDv2 sequences
│
├── outputs/
│   ├── stage_A/    # Rectification + GT geometry
│   ├── stage_B/    # COLMAP sparse+dense
│   ├── stage_C/    # NeRF training
│   └── stage_D/    # Geometry evaluation
│
├── scripts/
│   ├── run_rectify_and_stageA.py
│   ├── run_ns_train_seq.sh
│   ├── build_pointclouds_for_seq.py
│   ├── stage_D_eval_rawdepth_vs_gt.py
│   └── index_models.py
│
└── web-viewer/                        # Three.js viewer

````

---

# 2. Environment Installation  
(Required — **do not modify versions**)

This project uses:

- **Python 3.8**
- **PyTorch 2.1.2 + CUDA 11.8**
- **Nerfstudio 1.1.5**
- **Conda GCC 11**
- **CUDA toolkit from conda (not system)**  

---

## 2.1 Install Miniforge + Create Conda Environment

```bash
conda create -n nerfstudio python=3.8
conda activate nerfstudio
````

---

## 2.2 Install PyTorch 2.1.2 + CUDA 11.8

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --extra-index-url https://download.pytorch.org/whl/cu118
```

---

## 2.3 Install CUDA Toolkit (inside conda)

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

---

## 2.4 Install GCC 11 (required by tiny-cuda-nn)

```bash
conda install -c conda-forge gcc=11 gxx=11
```

Add stub libraries so tiny-cuda-nn can link:

```bash
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LD_LIBRARY_PATH
```

---

## 2.5 Install Nerfstudio (fixed stable version)

```bash
pip install nerfstudio==1.1.5
ns-install-cli
```

---

## 2.6 Install Remaining Python Dependencies

```bash
pip install numpy open3d matplotlib pillow tqdm scipy scikit-image
```

**These packages do not cause version conflicts.**

---

## 2.7 Install COLMAP

### Option A (Recommended): Use Official Binary

```bash
sudo apt install colmap
```

### Option B: Windows COLMAP GUI

Works fine for dense reconstruction.

---

# 3. Environment Configuration

Every new terminal must run:

```bash
conda activate nerfstudio
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LD_LIBRARY_PATH
```

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Verify Nerfstudio:

```bash
ns-train --help
```

---

# 4. How to Run the Pipeline

The pipeline has **four stages**:

---

## **Stage A — Rectification + GT Point Cloud**

```bash
python scripts/run_rectify_and_stageA.py --seq <SEQ_NAME>
```

Outputs:

```
outputs/stage_A/<SEQ>/undistorted/rgb/*.png
outputs/stage_A/<SEQ>/camera_omni.json
outputs/stage_A/<SEQ>/camera_pinhole.json
outputs/stage_A/<SEQ>/transforms.json
outputs/stage_A/<SEQ>/gt_pointcloud_full.ply
```

---

## **Stage B — SfM via COLMAP**

```bash
colmap automatic_reconstructor \
  --workspace_path outputs/stage_B/<SEQ>/ \
  --image_path outputs/stage_A/<SEQ>/undistorted/rgb
```

Outputs:

```
outputs/stage_B/<SEQ>/colmap_dense/0/fused.ply
```

---

## **Stage C — NeRF Training**

Use the launcher script:

```bash
./scripts/run_ns_train_seq.sh <SEQ_NAME> <GPU_ID> 30000
```

Render raw-depth:

```bash
ns-render dataset \
  --load-config outputs/stage_C/<SEQ>/outputs/nerfacto/.../config.yml \
  --output-path outputs/stage_D/<SEQ>/raw_depth \
  --rendered-output-names raw-depth \
  --split train+test
```

---

## **Stage D — Geometry Evaluation (Chamfer)**

```bash
python scripts/stage_D_eval_rawdepth_vs_gt.py --seq <SEQ_NAME>
```

Outputs:

```
outputs/stage_D/<SEQ>/eval_chamfer.json
outputs/stage_D/<SEQ>/plots/*.png
```

---

# 5. Web Viewer

Start server:

```bash
cd web-viewer
python3 -m http.server 8000
```

Generate PLYs for viewer:

```bash
python scripts/build_pointclouds_for_seq.py --seq <SEQ>
python scripts/index_models.py
```

Open in browser:

```
http://localhost:8000
```

Supports GT / SfM / NeRF point clouds.

---

# 6. Reproducing the Paper Results

Run the full pipeline (A→B→C→D) on the four sequences:

* `c1_descending_t2_v2`
* `c2_ascending_t1_v1`
* `c2_transverse1_t3_v2`
* `c2_rectum_t4_v3`

Then generate figures:

```
fig_overview.pdf
fig_rectification.pdf
fig_cloud_compare.pdf
fig_rectum_failure.pdf
fig_alpha_sweep.pdf
```

---

# 7. Troubleshooting

### ❌ Nerfstudio installation errors

→ You forgot GCC 11 or LD_LIBRARY_PATH

### ❌ tiny-cuda-nn fails to compile

→ GCC version mismatch
→ CUDA toolkit not installed inside conda

### ❌ ns-render cannot find checkpoint

→ Wrong config path; use the latest timestamp directory

### ❌ Web viewer empty

→ Missing index.json
→ Wrong path in viewer.js
