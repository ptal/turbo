#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run and hardware information.

MZN_SOLVER="turbo.gpu.profiling"
VERSION="v1.1.3"
NUM_GPUS=1

HARDWARE="\"AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM\""
SHORT_HARDWARE="A100"
NCU_COMMAND="ncu -f --set full --replay-mode application --target-processes all -k gpu_solve_kernel"
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s "
INSTANCE_FILE="short.csv"
OUTPUT_DIR="../campaign/$MZN_SOLVER-$VERSION-$SHORT_HARDWARE"
mkdir -p $OUTPUT_DIR

# II. Run the experiments in parallel (one per available GPUs).

cp $0 $OUTPUT_DIR/ # for replicability.

CUDA_WRAP_PATH=$(pwd)/cuda_wrap.sh
cp $CUDA_WRAP_PATH $OUTPUT_DIR/

parallel --rpl '{} uq()' --jobs $NUM_GPUS -k --colsep ',' --skip-first-line $NCU_COMMAND -o $OUTPUT_DIR/{1} $MZN_COMMAND {2} {3} -cutnodes {4}  ">" {1}.output :::: $INSTANCE_FILE

rm *.lock
rm *.count
rm *.cuda_device

