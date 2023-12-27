#!/bin/bash

# Loop through all files with the specified pattern
for file in [0-9]*_TurboGPU_*.yml; do
    # Extract the initial integer from the filename
    initial_number=$(echo "$file" | sed -E 's/^([0-9]+).*/\1/')
    
    # Add 35 to the extracted number
    new_number=$((initial_number + 35))
    
    # Replace the old number with the new number in the filename
    new_filename=$(echo "$file" | sed -E "s/^$initial_number/$new_number/")

    # Rename the file
    mv "$file" "$new_filename"
done

