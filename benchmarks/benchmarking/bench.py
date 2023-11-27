from datetime import timedelta
from pathlib import Path

import minizinc

from mzn_bench import Configuration, schedule

schedule(
    instances=Path("short.csv"),
    output_dir=Path("../campaign/turbo-v1.1.0-cpu/"),
    timeout=timedelta(seconds=300),
    configurations=[
        # Configuration(name="or-tools", solver=minizinc.Solver.lookup("com.google.or-tools")),
        # Configuration(name="or-tools.noglobal", solver=minizinc.Solver.lookup("com.google.or-tools.noglobal")),
        # Configuration(name="choco", solver=minizinc.Solver.lookup("org.choco.choco")),
        # Configuration(name="choco.noglobal", solver=minizinc.Solver.lookup("org.choco.choco.noglobal")),
        Configuration(name="TurboCPU", solver=minizinc.Solver.lookup("turbo.cpu.release"), extra_data={"-version": "1.1.0", "-hardware" : "Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000"})
    ],
    nodelist=None
)

schedule(
    instances=Path("short.csv"),
    output_dir=Path("../campaign/turbo-v1.1.0-A5000/"),
    timeout=timedelta(seconds=360),
    configurations=[
        Configuration(name="TurboGPU", solver=minizinc.Solver.lookup("turbo.gpu.release"), extra_data={"-kt": 60000, "-sub": 10, "-version": "1.1.0", "-hardware" : "Intel Core i9-10900X@3.7GHz;24GO DDR4;NVIDIA RTX A5000"})
    ],
    nodelist=None
)
