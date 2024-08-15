#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run and hardware information.

MZN_SOLVER="turbo.gpu.profiling"
VERSION="v1.1.8"
NUM_GPUS=1

HARDWARE="\"Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000\""
SHORT_HARDWARE="A5000"
# --replay-mode application
NCU_COMMAND="ncu -f --set full -k gpu_solve_kernel"
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s "
INSTANCE_FILE="very.csv"
OUTPUT_DIR="../campaign/$MZN_SOLVER-$VERSION-$SHORT_HARDWARE"
mkdir -p $OUTPUT_DIR

# II. Run the experiments in parallel (one per available GPUs).

cp $0 $OUTPUT_DIR/ # for replicability.

parallel --rpl '{} uq()' --jobs $NUM_GPUS -k --colsep ',' --skip-first-line $NCU_COMMAND -o $OUTPUT_DIR/{1} $MZN_COMMAND {2} {3} -cutnodes {4}  ">" {1}.output :::: $INSTANCE_FILE
