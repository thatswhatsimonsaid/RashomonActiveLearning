#!/bin/bash
for f in submit_*.sbatch; do sbatch "$f"; done