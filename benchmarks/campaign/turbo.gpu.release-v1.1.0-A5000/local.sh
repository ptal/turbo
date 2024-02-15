#!/bin/bash

# Exits when an error occurs.
set -e

# I. Define the campaign to run and hardware information.

MZN_SOLVER="turbo.gpu.release"
VERSION="v1.1.0"
TIMEOUT=310000
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

cp $0 $OUTPUT_DIR/ # for replicability.

DUMP_PY_PATH=$(pwd)/dump.py
parallel $MULTINODES_OPTION --rpl '{} uq()' --process-slot-var=CUDA_VISIBLE_DEVICES --jobs $NUM_GPUS -k --colsep ',' --header : $MZN_COMMAND {2} {3} '|' python3 $DUMP_PY_PATH $OUTPUT_DIR {1} {2} {3} $MZN_SOLVER :::: $INSTANCE_FILE
