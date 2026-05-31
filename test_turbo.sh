#!/usr/bin/env bash

# Arrays of variable and value orders
# eps_var_orders=( input_order first_fail anti_first_fail smallest largest )
eps_var_orders=( input_order )
# eps_value_orders=( min max split reverse_split )
eps_value_orders=( min )
simplify_flag=( "" "-disable_simplify" )
# architectures=( "cpu" "barebones" )
architectures=( "barebones" )

CSV="benchmarks/test_list.csv"

# global flag / counters
any_errors=0
passed=0
failed=0
while read -r line; do
    raw_path=$(echo "$line" | cut -d ',' -f 1)
    expected=$(echo "$line" | cut -d ',' -f 2)

    # Strip any surrounding spaces
    problem_path=$(echo "$raw_path" | xargs)
    problem_name=$(basename "$problem_path")
    expected=$(echo "$expected" | xargs)

    # echo "=== Problem: $problem_path (expected bound: $expected) ==="
    mismatch=0

    for var_order in "${eps_var_orders[@]}"; do
    for value_order in "${eps_value_orders[@]}"; do
    for simplify in "${simplify_flag[@]}"; do
    for arch in "${architectures[@]}"; do
      cmd=(
        ./build/gpu-release-local/turbo
        -eps_var_order "$var_order"
        -eps_value_order "$value_order"
        -arch $arch
        $simplify
        -s
        -t 60000
        "$problem_path"
        )
        out=$("${cmd[@]}" < /dev/null 2>&1)

      # only take the first match of each
      obj=$(echo "$out" | grep -m1 -oP 'objective=\K-?[0-9]+(?:\.[0-9]+)?' || :)
      time=$(echo "$out" | grep -m1 -oP 'solveTime=\K[0-9]+(?:\.[0-9]+)?'   || :)

      time_int=${time%.*}

      # build status
      if [[ -z "$obj" ]]; then
          if [[ "$time_int" -lt 60 ]]; then
              status="✗ [no bound, expected=$expected]"
              mismatch=1
          else
              status="✔ [no bound, timeout]"
          fi
      elif [[ "$obj" != "$expected" && "$time_int" -lt 60 ]]; then
          status="✗ [bound=$obj, expected=$expected]"
          mismatch=1
      elif [[ "$time_int" -ge 60 ]]; then
          status="✔ [timeout]"
      else
          status="✔"
      fi

      status="$status [${problem_name}]"

      # tack on the time
      if [[ -n "$time" ]]; then
          status="$status [${time}s]"
      fi
      echo "$status (${cmd[@]})"
    done
    done
    done
    done

    if (( mismatch )); then
        echo ">>> ERROR detected for $problem_path <<<"
        ((failed++))
        any_errors=1
    else
        ((passed++))
    fi
done < "$CSV"

# final summary
echo "======== Final Report ========"
echo "  ✔ Passed: $passed"
echo "  ✗ Failed: $failed"
if (( any_errors )); then
    echo "Wrong bounds detected!"
    exit 1
else
    echo "No error detected!"
    exit 0
fi
