import csv
import sys
import os
import re
import subprocess
import pandas

turbo_git = "https://github.com/ptal/turbo.git"
build_dir = "build/"
turbo_repo = build_dir + "turbo/"
turbo_exec = "turbo"
input_dir = "../../kobe-scheduling/data/"
xp_dir = "experiments/"

def usage():
  print("python3 kobe.py <timeout> <machine> <family> <suite> <cpu|gpu> <sub> <or> <and> [<local> <exec_path> | <branch> <commit-hash|tag>] [dryrun]")
  exit()

def natural_sort(l):
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(l, key=alphanum_key)

def exec_command(dryrun, command):
  if dryrun:
    print(command)
    return 0
  else:
    return os.system(command)

def exec_printf(dryrun, text, results_filename):
  exec_command(dryrun, "printf \"%s\" \"" + text + ", \" | tee -a " + results_filename)

def git_version_id(dryrun, branch, commit_hash_tag) -> str:
  if branch == "main":
    return commit_hash_tag
  elif branch == "local":
    return branch
  else:
    return branch + "-" + commit_hash_tag
  # From https://stackoverflow.com/a/14989911
  # res = "dryrun"
  # if dryrun:
  #   print("git describe --always")
  # else:
  #   res = subprocess.check_output(["git", "describe", "--always"], cwd=turbo_repo).decode('ascii').strip()
  # return res

def delete_last_line(dryrun, filename):
  exec_command(dryrun, "tail -n 1 " + filename + " | wc -c | xargs -I {} truncate " + filename + " -s -{}")

def resume_experiment(dryrun, results_filename) -> int:
  # Note that we count the header line, and array starts at 0, so the result of `wc -l` is good.
  start_at = int(subprocess.check_output(["wc", "-l", results_filename]).decode('ascii').strip().split()[0])
  # Suppose the last experiments failed (redo it)
  start_at = max(0, start_at - 1)
  # Delete last line of the failed experiment in the file
  delete_last_line(dryrun, results_filename)
  print("Resuming at experiment", start_at+1, "...")
  return start_at

def build_turbo(dryrun, arch, branch, commit_hash_tag):
  exec_command(dryrun, f"""
    mkdir -p build/
    cd build/
    git clone {turbo_git}
    cd turbo/
    git pull # in case the repository was already cloned.
    git checkout {branch}
    git checkout {commit_hash_tag}
    ./{arch}-release.sh""")

if len(sys.argv) < 11 or (sys.argv[5] != "cpu" and sys.argv[5] != "gpu"):
  usage()

timeout = sys.argv[1]
machine = sys.argv[2]
family = sys.argv[3]
suite = sys.argv[4]
arch = sys.argv[5]
sub_option = sys.argv[6]
or_option = sys.argv[7]
and_option = sys.argv[8]
branch = sys.argv[9]
commit_hash_tag = sys.argv[10]
dryrun = len(sys.argv) > 11 and sys.argv[11] == "dryrun"

if branch == "local":
  turbo_repo = commit_hash_tag
  if turbo_repo[-1] != '/':
    turbo_repo = turbo_repo + '/'
else:
  build_turbo(dryrun, arch, branch, commit_hash_tag)

turbo_command = turbo_repo + turbo_exec + " " + timeout + " -arch " + arch + " -sub " + sub_option + " -or " + or_option + " -and " + and_option + " "
solver_id = "turbo-" + git_version_id(dryrun, branch, commit_hash_tag)
prefix = suite + "-" + solver_id + "-" + arch +  "-" + timeout + "-" + sub_option + "-" + and_option + "-" + or_option
output_dir = xp_dir + machine + "/" + family + "/"
results_filename = output_dir + prefix + ".csv"
error_dir = build_dir + output_dir
tee_command_result = " | tee -a " + results_filename

exec_command(dryrun, f"mkdir -p {error_dir}")
exec_command(dryrun, f"mkdir -p {output_dir}")

start_at = 0
if os.path.isfile(results_filename):
  start_at = resume_experiment(dryrun, results_filename)
else:
  exec_printf(dryrun, "problem", results_filename)
  exec_command(dryrun, turbo_repo + turbo_exec + " -only_csv_header" + tee_command_result)

inputs_set = input_dir + family + "/" + suite + ".xcsp3/"
inputs_set_dir = os.fsencode(inputs_set)
sorted_inputs_set = natural_sort([os.fsdecode(file) for file in os.listdir(inputs_set_dir)])

# os.system("mkdir -p patterson-new/")
# for a, b in zip(sorted_inputs_set, sorted_inputs_set[1:]):
#   os.system("cp " + inputs_set + a + " patterson-new/" + b)

for filename in sorted_inputs_set[start_at:]:
  if filename.endswith(".xml"):
    instance_name = os.path.splitext(filename)[0]
    instance_path = inputs_set + filename
    error_path = error_dir + instance_name + ".err"
    exec_printf(dryrun, instance_name, results_filename)
    command = turbo_command + instance_path + " 2> " + error_path + tee_command_result + " ; [ -s " + error_path + " ] || rm -f " + error_path
    status = exec_command(dryrun, command)
    if status != 0:
      print(f"The solver did not terminate correctly, the command is: `{command}`")
      exit(1)

# Analysis of the soundness of the results.

if dryrun:
  exit()

pure_suite = suite.split(".")[0]
optimum_filename = input_dir + family + "/" + pure_suite + ".optimum.csv"

results_df = pandas.read_csv(results_filename, skipinitialspace = True)
optimum_df = pandas.read_csv(optimum_filename, skipinitialspace = True)
optimums = dict(zip(optimum_df["problem"], optimum_df["optimum"]))

for problem, sat, exhaustive, opt1 in zip(results_df["problem"], results_df["satisfiability"], results_df["exhaustivity"], results_df["optimum"]):
  problem = problem.strip()
  sat = sat.strip()
  opt1 = opt1.strip()
  if problem not in optimums:
    print(f"The result file contains {problem} which could not be found in the optimum file.")
    exit(1)
  opt2 = optimums[problem]
  if sat == "unsat":
    if opt2 != "unsat":
      print(f"{problem} was found unsat, but it is satisfiable.")
    if not exhaustive:
      print(f"{problem} was found unsat, but the search was not exhaustive.")
  elif sat == "sat":
    if not exhaustive:
      print(f"{problem} was found unsat, but the search was not exhaustive.")
    elif opt1 == "none":
      print(f"{problem} was found sat, but no optimum was found.")
    elif opt2 == "unsat":
      print(f"{problem} was found sat, but it is unsatisfiable.")
    else:
      opt_found = int(opt1)
      opt_best = int(opt2)
      if exhaustive and opt_found != opt_best:
        print(f"{problem} was proven optimal with bound {opt_found} but the optimal bound is actually {opt_best}.")
      elif opt_found < opt_best:
        print(f"{problem}: the best bound found is {opt_found}, but the optimal bound is actually {opt_best}. Note that we suppose a minimization problem (analyze of maximization problem is yet unsupported.")
  elif sat == "unknown" and (exhaustive or opt1 != "none"):
    print(f"{problem}: the solver said it does not know if the problem is satisfiable or not, but was exhaustive ({exhaustive}) or found an optimal bound ({opt1})")
