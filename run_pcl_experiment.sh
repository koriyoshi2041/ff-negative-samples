#!/bin/bash
# Run PCL-FF experiment with specified parameters
# Usage: ./run_pcl_experiment.sh

cd /Users/parafee41/Desktop/Rios/ff-research

# Activate virtual environment and run experiment
source venv/bin/activate

python experiments/pcl_ff_experiment.py \
    --epochs 500 \
    --batch-size 50000 \
    --output results/pcl_ff_results.json

echo "Experiment complete!"
