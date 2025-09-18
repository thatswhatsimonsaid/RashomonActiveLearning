#!/bin/bash
for file in ./*.sbatch; do sbatch "$file"; done