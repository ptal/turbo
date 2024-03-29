#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run and hardware information.

MZN_SOLVER="turbo.gpu.release"
VERSION="v1.1.7"
# This is to avoid MiniZinc to kill Turbo before it can print the statistics.
MZN_TIMEOUT=1260000
REAL_TIMEOUT=1200000
NUM_GPUS=4

HARDWARE="\"AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM\""
SHORT_HARDWARE="A100"
MZN_COMMAND="minizinc --solver $MZN_SOLVER -s --json-stream -t $MZN_TIMEOUT --output-mode json --output-time --output-objective -hardware $HARDWARE -version $VERSION -timeout $REAL_TIMEOUT"
INSTANCE_FILE="mzn2022.csv"
OUTPUT_DIR=$(pwd)"/../campaign/$MZN_SOLVER-$VERSION-$SHORT_HARDWARE"
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

## III. Gather the list of Slurm nodes to run the experiments on many nodes if available.

if [ -n "${SLURM_JOB_NODELIST}" ]; then
  # get host name
  NODES_HOSTNAME="nodes_hostname.txt"
  scontrol show hostname $SLURM_JOB_NODELIST > $NODES_HOSTNAME
  # Collect public key and accept them
  while read -r node; do
      ssh-keyscan "$node" >> ~/.ssh/known_hosts
  done < "$NODES_HOSTNAME"
  MULTINODES_OPTION="--sshloginfile $NODES_HOSTNAME"
  cp $(realpath "$(dirname "$0")")/slurm.sh $OUTPUT_DIR/
fi

# IV. Run the experiments in parallel (one per available GPUs).

DUMP_PY_PATH=$(pwd)/dump.py
CUDA_WRAP_PATH=$(pwd)/cuda_wrap.sh

cp $0 $OUTPUT_DIR/ # for replicability.
cp $DUMP_PY_PATH $OUTPUT_DIR/
cp $CUDA_WRAP_PATH $OUTPUT_DIR/

parallel --no-run-if-empty $MULTINODES_OPTION --rpl '{} uq()' --jobs $NUM_GPUS -k --colsep ',' --skip-first-line $CUDA_WRAP_PATH $MZN_COMMAND {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER :::: $INSTANCE_FILE
# TEST_PATH=$(pwd)/test.sh
# parallel --no-run-if-empty $MULTINODES_OPTION --rpl '{} uq()' --jobs $NUM_GPUS -k --colsep ',' --skip-first-line $CUDA_WRAP_PATH $TEST_PATH {1} {2} {3} {4} :::: $INSTANCE_FILE ::: 10 12

rm *.lock
rm *.count
rm *.cuda_device
