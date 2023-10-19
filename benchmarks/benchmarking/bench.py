from datetime import timedelta
from pathlib import Path

import minizinc

from mzn_bench import Configuration, schedule

schedule(
    instances=Path("short.csv"),
    output_dir=Path("../campaign/baseline/"),
    timeout=timedelta(seconds=300),
    configurations=[
        Configuration(name="OrTools", solver=minizinc.Solver.lookup("ortools")),
        Configuration(name="OrTools", solver=minizinc.Solver.lookup("ortools.noglobal")),
        Configuration(name="Choco", solver=minizinc.Solver.lookup("choco")),
        Configuration(name="Choco", solver=minizinc.Solver.lookup("choco.noglobal")),
        # Configuration(name="TurboCPU", solver=minizinc.Solver.lookup("turbo.cpu.release")),
        # Configuration(name="TurboGPU", solver=minizinc.Solver.lookup("turbo.gpu.release"))
    ],
    nodelist=None
)
