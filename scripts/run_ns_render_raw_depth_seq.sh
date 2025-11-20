#!/usr/bin/env bash
# scripts/run_ns_render_raw_depth_seq.sh
# 用 ns-render dataset 渲染某个序列的 raw-depth（test split）

set -e

if [ $# -lt 1 ]; then
  echo "用法: $0 <SEQ_NAME> [SPLIT]"
  echo "例如: $0 c2_ascending_t1_v1 test"
  exit 1
fi

SEQ="$1"
SPLIT="${2:-test}"   # 默认 test

BASE_PROJ="/data1_ycao/chua/projects/cdTeacher"
STAGE_A_ROOT="${BASE_PROJ}/outputs/stage_A/${SEQ}"
STAGE_D_ROOT="${BASE_PROJ}/outputs/stage_D/${SEQ}"

# 自动找到最新的 nerfstudio 训练目录
NERF_RUN_DIR=$(ls -dt "${STAGE_A_ROOT}"/outputs/nerfacto/* 2>/dev/null | head -n 1 || true)
if [ -z "$NERF_RUN_DIR" ]; then
  echo "❌ 找不到 Nerfstudio 训练目录: ${STAGE_A_ROOT}/outputs/nerfacto/*"
  exit 1
fi

CONFIG="${NERF_RUN_DIR}/config.yml"

echo "======================================="
echo "[NS-RENDER] SEQ        : ${SEQ}"
echo "[NS-RENDER] SPLIT      : ${SPLIT}"
echo "[NS-RENDER] CONFIG     : ${CONFIG}"
echo "[NS-RENDER] OUTPUT     : ${STAGE_D_ROOT}"
echo "======================================="

mkdir -p "${STAGE_D_ROOT}"

ns-render dataset \
  --load-config "${CONFIG}" \
  --output-path "${STAGE_D_ROOT}" \
  --rendered-output-names raw-depth \
  --split "${SPLIT}"
