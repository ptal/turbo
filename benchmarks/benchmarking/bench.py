from datetime import timedelta
from pathlib import Path

import minizinc

from mzn_bench import Configuration, schedule

schedule(
    instances=Path("short.csv"),
    output_dir=Path("../campaign/baseline/"),
    timeout=timedelta(seconds=300),
    configurations=[
        Configuration(name="or-tools", solver=minizinc.Solver.lookup("com.google.or-tools")),
        Configuration(name="or-tools.noglobal", solver=minizinc.Solver.lookup("com.google.or-tools.noglobal")),
        Configuration(name="choco", solver=minizinc.Solver.lookup("org.choco.choco")),
        Configuration(name="choco.noglobal", solver=minizinc.Solver.lookup("org.choco.choco.noglobal")),
        Configuration(name="TurboCPU", solver=minizinc.Solver.lookup("turbo.cpu.release")),
        Configuration(name="TurboGPU", solver=minizinc.Solver.lookup("turbo.gpu.release"))
    ],
    nodelist=None
)