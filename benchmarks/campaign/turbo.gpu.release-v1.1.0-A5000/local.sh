#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run and hardware information.

MZN_SOLVER="turbo.gpu.release"
VERSION="v1.1.0"
TIMEOUT=360000
NUM_GPUS=1

HARDWARE="\"Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000\""
SHORT_HARDWARE="A5000"
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream -t $TIMEOUT --output-mode json --output-time --output-objective -hardware $HARDWARE -version $VERSION"
INSTANCE_FILE="short.csv"
OUTPUT_DIR="../campaign/$MZN_SOLVER-$VERSION-$SHORT_HARDWARE"
mkdir -p $OUTPUT_DIR

# II. Compile and install the right version of Turbo.

if [[ "$NOCOMPILE_TURBO" == 1 ]]; then
  echo "Skip Turbo compilation..."
else
  cd ../../
  git checkout $VERSION
  cmake --workflow --preset gpu-release --fresh
  cp benchmarks/minizinc/turbo.* ~/.minizinc/solvers/
  # We replace the path of Turbo inside the configuration files with the right one.
  TURBO_PATH=$(pwd)
  sed -i "s|/home/ptalbot/repositories/lattice-land/turbo|${TURBO_PATH}|g" ~/.minizinc/solvers/turbo.*.msc
  cd benchmarks/benchmarking/
  git checkout main
fi

# III. Run the experiments in parallel (one per available GPUs).

DUMP_PY_PATH=$(pwd)/dump.py

cp $0 $OUTPUT_DIR/ # for replicability.
cp $DUMP_PY_PATH $OUTPUT_DIR/

parallel --rpl '{} uq()' --jobs $NUM_GPUS -k --colsep ',' --skip-first-line $MZN_COMMAND {2} {3} -sub 10 '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER "sub10" :::: $INSTANCE_FILE
