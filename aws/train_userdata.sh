#!/bin/bash
# EC2 user-data: run training jobs on a GPU instance from S3 bundles, then terminate.
# Launch with --instance-initiated-shutdown-behavior terminate. Placeholders are
# filled by aws/launch.sh: run tag (S3 folder), time limit (minutes), job lines.
TAG=__TAG__
S3=__S3__
shutdown -h +__LIMIT__                            # hard limit no matter what
exec > /var/log/gamd-train.log 2>&1
set -x
date -u
sync_up() { aws s3 sync /opt/gamd/runs $S3/runs/$TAG --exclude "*.pt" --include "*/ckpt_final.pt" --only-show-errors; aws s3 cp /var/log/gamd-train.log $S3/runs/$TAG/train.log --only-show-errors; }
( while true; do sleep 60; sync_up; done ) &
source /opt/pytorch/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
pip install -q rasterio scipy pillow pyproj shapely
mkdir -p /opt/gamd && cd /opt/gamd
aws s3 cp $S3/bundle-v1.tar - --only-show-errors | tar x
aws s3 cp $S3/pre-v1.tar - --only-show-errors | tar x          # precomputed relief bands (train.py --pre)
aws s3 cp $S3/pre-v2-hard.tar - --only-show-errors | tar x     # + 500 hard scenes (bluff edges, corridors)
aws s3 cp $S3/pre-v3-oos.tar - --only-show-errors | tar x      # + out-of-state park mounds (training only)
aws s3 cp $S3/pre-v4-mined.tar - --only-show-errors | tar x    # + reviewed false flags from random Georgia ground
rm -f data/training/v1/scenes/real_* data/training/v1/pre/real_*  # real scenes were rebuilt (18 test mounds)
aws s3 cp $S3/pre-v5-real.tar - --only-show-errors | tar x
aws s3 cp $S3/code-latest.tar - --only-show-errors | tar x      # newer code + eval tiles over the data bundle
date -u
__JOBS__
date -u
sync_up
shutdown -h now
