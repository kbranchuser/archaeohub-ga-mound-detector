#!/bin/bash
# EC2 user-data: scan areas with the final model, save results to S3 (private), terminate.
# Filled by aws/launch.sh (USERDATA=aws/scan_userdata.sh): run tag, time limit, area rows.
TAG=__TAG__
S3=__S3__
shutdown -h +__LIMIT__                            # hard limit no matter what
exec > /var/log/gamd-scan.log 2>&1
set -x
sync_up() { aws s3 sync /opt/gamd/out $S3/scans/$TAG/ --only-show-errors; aws s3 cp /var/log/gamd-scan.log $S3/scans/$TAG/scan.log --only-show-errors; }
( while true; do sleep 60; sync_up; done ) &
source /opt/pytorch/bin/activate
pip install -q rasterio scipy pillow pyproj
mkdir -p /opt/gamd/runs/final/all /opt/gamd/models/davis2021 && cd /opt/gamd
aws s3 cp $S3/code-latest.tar - --only-show-errors | tar x
aws s3 cp $S3/runs/final-v1/final/all/ckpt_final.pt runs/final/all/ckpt_final.pt --only-show-errors
aws s3 cp $S3/models/davis2021/RCNN_2_Output.pth models/davis2021/RCNN_2_Output.pth --only-show-errors   # base architecture weights (load builds on them)
cat > areas.csv <<'CSV'
__JOBS__
CSV
python scripts/area_scan.py areas.csv out
sync_up
shutdown -h now
