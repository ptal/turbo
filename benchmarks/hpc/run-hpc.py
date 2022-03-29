import os
import sys
import re

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

timeout = sys.argv[1]
sub_option = sys.argv[2]
suite = sys.argv[3]
and_option = str(108*4)
or_option = str(64*4)
if len(sys.argv) > 4:
  and_option = sys.argv[3]
if len(sys.argv) > 5:
  or_option = sys.argv[4]

directory_str = "benchmarks-xml/" + suite + "-xml/"
directory = os.fsencode(directory_str)
prefix = suite + "-" + timeout + "-" + and_option + "-" + or_option + "-" + sub_option
results_filename = "benchmarks-xml/" + prefix + ".csv"
with open(results_filename, 'w') as f:
  f.write("name, nodes, fails, solutions, depthmax, variables, constraints, satisfiability, exhaustivity, time, optimum\n")

error_dir = "benchmarks-xml/error/" + prefix + "-err/"
os.system("mkdir -p " + error_dir)

sorted_dir = natural_sort([os.fsdecode(file) for file in os.listdir(directory)])

for filename in sorted_dir:
  if filename.endswith(".xml"):
    instance_name = os.path.splitext(filename)[0]
    instance_path = directory_str + filename
    error_path = error_dir + instance_name + ".err"
    os.system("printf \"%s\" \"" + instance_name + ", \" | tee -a " + results_filename)
    command = "./turbo " + timeout + " -and " + and_option + " -or " + or_option + " -sub " + sub_option + " " + instance_path + " 2> " + error_path + " | tee -a " + results_filename
    # print(command)
    os.system(command)