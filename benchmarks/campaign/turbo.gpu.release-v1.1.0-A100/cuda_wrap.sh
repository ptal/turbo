#!/bin/bash

counter_file=/project/scratch/p200244/lattice-land/turbo/benchmarks/benchmarking/${PARALLEL_SSHHOST}.count
lock_file=/project/scratch/p200244/lattice-land/turbo/benchmarks/benchmarking/${PARALLEL_SSHHOST}.lock

(
  flock -x 200
  if [ -f "$counter_file" ]; then
    read -r counter < $counter_file
  else
    counter=0
  fi

  cuda_device=$((counter % 4))
  echo $cuda_device > "$PARALLEL_SSHHOST"_"$PARALLEL_JOBSLOT".cuda_device

  ((counter++))
  echo "$counter" > "$counter_file"

) 200>$lock_file

read -r CUDA_VISIBLE_DEVICES_LOCAL < "$PARALLEL_SSHHOST"_"$PARALLEL_JOBSLOT".cuda_device


# Construct the command string while preserving quotes
cmd="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_LOCAL"
for arg in "$@"; do
  # Escape double quotes in the argument
  arg="${arg//\"/\\\"}"
  
  # Wrap the argument in quotes if it contains spaces
  [[ $arg =~ \  ]] && arg="\"$arg\""
  
  cmd="$cmd $arg"
done

# Execute the command
eval "$cmd"

# eval "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_LOCAL $@"
