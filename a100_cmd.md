# A100 命令

跳过Dendritic FF（需要55GB，A100只有40GB），运行其他实验：

```
cd ~/rds/ff-a100-package && python experiments/pcl_ff_experiment.py --epochs 500 && python experiments/prospective_ff_experiment.py --mode full && python experiments/layer_collab_transfer.py --pretrain-epochs 500
```
