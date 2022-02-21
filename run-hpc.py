import os
import sys

sub_option = sys.argv[1]
suite = sys.argv[2]
and_option = str(108*4)
or_option = str(64*4)
if len(sys.argv) > 3:
  and_option = sys.argv[3]
if len(sys.argv) > 4:
  or_option = sys.argv[4]

directory = os.fsencode("benchmarks-xml/" + suite + "-xml/")
results_prefix = "benchmarks-xml/" + suite + "-" + and_option + "-" + or_option + "-" + sub_option
results_filename = results_prefix + ".csv"
with open(results_filename, 'w') as f:
  f.write("name, nodes, fails, solutions, depthmax, variables, constraints, satisfiability, exhaustivity, time, optimum")

error_dir = results_prefix + "-err/"
os.system("mkdir " + error_dir)

for file in os.listdir(directory):
  filename = os.fsdecode(file)
  if filename.endswith(".xml"):
    instance_name = os.path.splitext(filename)[0]
    instance_path = os.fsdecode(os.path.join(directory, file))
    error_path = error_dir + instance_name + ".err"
    os.system("echo \"" + instance_name + ", \" >> " + results_filename)
    command = "./turbo -and " + and_option + " -or " + or_option + " -sub " + sub_option + " " + instance_path + " 2> " + error_path + " | tee -a " + results_filename
    # print(command)
    os.system(command)