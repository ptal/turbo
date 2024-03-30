from sys import stdin
from pathlib import Path
import sys
import os
import minizinc
import json
from ruamel.yaml import YAML

yaml = YAML(typ="safe")
yaml.register_class(minizinc.types.ConstrEnum)
yaml.register_class(minizinc.types.AnonEnum)
yaml.default_flow_style = False

if os.environ.get("MZN_DEBUG", "OFF") == "ON":
  import logging
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

if __name__ == "__main__":
  output_dir = sys.argv[1]
  problem = sys.argv[2]
  model = Path(sys.argv[3])
  data = Path(sys.argv[4])
  solver = sys.argv[5]
  extras = []
  for i in range(6, len(sys.argv)):
    arg = sys.argv[i].strip().replace(' ', '-')
    if arg != "" and arg != "-s": # we use "-s" when there are "no special options to be used".
      extras.append(arg)
      # Remove leading "-" from extras (these are used for specifying options)
      if extras[-1].startswith("-"):
        extras[-1] = extras[-1][1:]

  uid = solver.replace('.', '-') + "_" + model.stem + "_" + data.stem
  if len(extras) > 0:
    uid += "_"
    uid += "_".join(extras)

  if(output_dir[-1] == "/"):
    output_dir = output_dir[:-1]
  if(Path(output_dir).exists() == False):
    os.mkdir(output_dir)
  sol_filename = Path(output_dir + "/" + uid + "_sol.yml")
  stats_filename = Path(output_dir + "/" + uid + "_stats.yml")

  stat_base = {
    "configuration": uid,
    "problem": problem,
    "model": str(model),
    "data_file": str(data),
    "mzn_solver": solver,
    "status": str(minizinc.result.Status.UNKNOWN)
  }

  statistics = stat_base.copy()

  solutions = []

  for line in stdin:
    output = json.loads(line)
    if(output["type"] == "statistics"):
      statistics.update(output["statistics"])
    elif(output["type"] == "status"):
      statistics["status"] = output["status"]
    elif(output["type"] == "solution"):
      sol = stat_base.copy()
      sol["status"] = str(minizinc.result.Status.SATISFIED)
      if(statistics["status"] == str(minizinc.result.Status.UNKNOWN)):
        statistics["status"] = str(minizinc.result.Status.SATISFIED)
      sol["solution"] = output["output"]["json"]
      if("_objective" in sol["solution"] and "objective" not in sol["solution"]):
        sol["solution"]["objective"] = sol["solution"]["_objective"]
      if("_objective" in sol["solution"]):
        del sol["solution"]["_objective"]
      sol["time"] = float(output["time"]) / 1000.0
      solutions.append(sol)
    elif(output["type"] == "error"):
      statistics["status"] = str(minizinc.result.Status.ERROR)
      statistics["error"] = str(output)
      print("Error: " + str(output), file=sys.stderr)
      statistics["rest_of_output"] = str(stdin.read())
      break
    else:
      print("Unknown output type: " + output["type"], file=sys.stderr)

  if solutions != []:
    with open(sol_filename, "w") as file:
      yaml.dump(solutions, file)
  with open(stats_filename, "w") as file:
    yaml.dump(statistics, file)
