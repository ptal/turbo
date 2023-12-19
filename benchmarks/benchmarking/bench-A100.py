from datetime import timedelta
from pathlib import Path

import minizinc

from mzn_bench import Configuration, schedule

subproblems = [10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50]
or_nodes = [1, 2, 4, 8, 27, 54, 81]
and_nodes = [64, 128]
c
sub = 10
or_node = 1
and_node = 1
configs = [Configuration(name=f"TurboGPU_{sub}_{or_node}_{and_node}", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-sub": sub, "-and": and_node, "-or": or_node, "-version": "1.1.0", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"})]

or_node = 108
and_node = 256
for sub in subproblems:
  configs.append(
    Configuration(name=f"TurboGPU_{sub}_{or_node}_{and_node}", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-sub": sub, "-and": and_node, "-or": or_node, "-version": "1.1.0", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"}))

sub = 10
and_node = 256
for or_node in or_nodes:
  configs.append(
    Configuration(name=f"TurboGPU_{sub}_{or_node}_{and_node}", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-sub": sub, "-and": and_node, "-or": or_node, "-version": "1.1.0", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"}))

sub = 10
or_node = 108
for and_node in and_nodes:
  configs.append(
    Configuration(name=f"TurboGPU_{sub}_{or_node}_{and_node}", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-sub": sub, "-and": and_node, "-or": or_node, "-version": "1.1.0", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"}))

print("Scheduling ", len(configs), " configurations...")

schedule(
    instances=Path("short.csv"),
    output_dir=Path("../campaign/turbo-v1.1.0-A100/"),
    timeout=timedelta(seconds=300),
    debug=False,
    configurations=configs,
    sbatch_config="slurm_config.sh"
)
