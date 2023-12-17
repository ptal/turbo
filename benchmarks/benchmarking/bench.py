from datetime import timedelta
from pathlib import Path

import minizinc

from mzn_bench import Configuration, schedule

schedule(
    instances=Path("very-short.csv"),
    output_dir=Path("../campaign/turbo-v1.1.0-A100-debug/"),
    timeout=timedelta(seconds=30),
    debug=True,
    configurations=[
        # Configuration(name="or-tools", solver=minizinc.Solver.lookup("com.google.or-tools")),
        # Configuration(name="or-tools.noglobal", solver=minizinc.Solver.lookup("com.google.or-tools.noglobal")),
        # Configuration(name="choco", solver=minizinc.Solver.lookup("org.choco.choco")),
        # Configuration(name="choco.noglobal", solver=minizinc.Solver.lookup("org.choco.choco.noglobal")),
        # Configuration(name="TurboCPU", solver=minizinc.Solver.lookup("turbo.cpu.release"), other_flags={"-version": "1.1.0", "-hardware" : "Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000"}),
        Configuration(name="TurboGPU", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-sub": 10, "-version": "1.1.0", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"}),
        Configuration(name="TurboGPU", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-sub": 12, "-version": "1.1.0", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"}),
    ],
    sbatch_config="slurm_config.sh"
)
