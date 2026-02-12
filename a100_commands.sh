#!/bin/bash
cd ~/rds/ff-a100-package

echo "=== Dendritic FF ==="
python experiments/dendritic_ff_experiment.py --epochs 500

echo "=== PCL FF ==="
python experiments/pcl_ff_experiment.py --epochs 500

echo "=== Prospective FF ==="
python experiments/prospective_ff_experiment.py --mode full

echo "=== Layer Collab ==="
python experiments/layer_collab_transfer.py --pretrain-epochs 500

echo "=== DONE ==="
ls results/
