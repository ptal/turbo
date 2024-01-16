from datetime import timedelta
from pathlib import Path

import minizinc

from mzn_bench import Configuration, schedule

schedule(
    instances=Path("graph-labelling-pos1024.csv"),
    output_dir=Path("../campaign/turbo-v1.2.0-1024-A100/"),
    timeout=timedelta(seconds=600),
    debug=False,
    configurations= [Configuration(name=f"TurboGPU", solver=minizinc.Solver.lookup("turbo.gpu.release"), other_flags={"-version": "1.2.0-1024", "-hardware" : "AMD EPYC 7452 32-Core@2.35GHz; RAM 512GO;NVIDIA A100 40GB HBM"})],
    sbatch_config="slurm_config.sh"
)
