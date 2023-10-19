import minizinc
from minizinc import Instance, Model, Solver
from datetime import datetime, timedelta


model = Model("data/mzn2022/pillars-planks-solution_p-d2.fzn")
mzn_solver = Solver.lookup("turbo.gpu.release")
instance = Instance(mzn_solver, model)
try:
  print("Start the CP solver...")
  res = instance.solve(
    all_solutions = False,
    timeout = timedelta(seconds = 10))
  print("Got a result from the CP solver...")
  print(res)
except minizinc.error.MiniZincError as err:
  print("The solver crashed...")
  print(err)