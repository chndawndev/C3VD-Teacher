#!/usr/bin/env bash
# run_ns_train_seq.sh
#
# 用法示例：
#   ./scripts/run_ns_train_seq.sh c2_ascending_t1_v1 1 30000

set -e

if [ $# -lt 2 ]; then
  echo "Usage: $0 <SEQ_NAME> <GPU_ID> [MAX_ITERS]"
  exit 1
fi

SEQ_NAME="$1"
GPU_ID="$2"
MAX_ITERS="${3:-30000}"

DATA_ROOT="/data1_ycao/chua/projects/cdTeacher/outputs/stage_A"
NS_OUT_ROOT="/data1_ycao/chua/projects/cdTeacher/outputs/nerfstudio_runs"

DATA_PATH="${DATA_ROOT}/${SEQ_NAME}"

EXP_NAME="c3vdv2_multi_seq_${SEQ_NAME}"

echo "======================================="
echo "[NS-TRAIN] SEQ       : ${SEQ_NAME}"
echo "[NS-TRAIN] GPU       : ${GPU_ID}"
echo "[NS-TRAIN] DATA_PATH : ${DATA_PATH}"
echo "[NS-TRAIN] EXP_NAME  : ${EXP_NAME}"
echo "[NS-TRAIN] MAX_ITERS : ${MAX_ITERS}"
echo "======================================="

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

ns-train nerfacto \
  --data "${DATA_PATH}" \
  --experiment-name "${EXP_NAME}" \
  --output-dir "${NS_OUT_ROOT}" \
  --max-num-iterations ${MAX_ITERS}
