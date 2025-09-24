#!/bin/bash

# Submits all generated entropy study jobs
for file in ./submit_entropy_*.sbatch; do
    sbatch "$file"
done
