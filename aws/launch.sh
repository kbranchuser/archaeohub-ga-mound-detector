#!/bin/bash
# Usage: [TYPE=g6.2xlarge] [USERDATA=aws/scan_userdata.sh] aws/launch.sh TAG LIMIT_MIN JOBS_FILE  (uploads current code, starts one GPU instance; default g6.xlarge)
# Your AWS setup (required): an S3 prefix you own, public subnet(s), a security group with
# no inbound rules, an instance profile that can read/write that prefix, and a GPU Deep
# Learning AMI (PyTorch, /opt/pytorch venv) in your region.
#   export GAMD_S3=s3://your-bucket/ga-mound-detector GAMD_SUBNETS="subnet-aaa subnet-bbb" \
#          GAMD_SG=sg-xxx GAMD_PROFILE=your-instance-profile GAMD_AMI=ami-xxx [GAMD_REGION=us-east-1]
set -e
cd "$(dirname "$0")/.."
TAG=$1; LIMIT=$2; JOBS=$3
: "${GAMD_S3:?set GAMD_S3}" "${GAMD_SUBNETS:?set GAMD_SUBNETS}" "${GAMD_SG:?set GAMD_SG}" "${GAMD_PROFILE:?set GAMD_PROFILE}" "${GAMD_AMI:?set GAMD_AMI}"
REGION=${GAMD_REGION:-us-east-1}
S3=$GAMD_S3
tar cf /tmp/gamd-code.tar src scripts data/eval_tiles_p75 data/baseline_tiles data/ga_mounds_verified.csv
aws s3 cp /tmp/gamd-code.tar $S3/code-latest.tar --only-show-errors
python3 - "$TAG" "$LIMIT" "$JOBS" <<'PY' > /tmp/gamd-ud.sh
import sys
tag, limit, jobs = sys.argv[1:]
import os
s = open(os.environ.get("USERDATA", "aws/train_userdata.sh")).read()
print(s.replace("__S3__", os.environ["GAMD_S3"]).replace("__TAG__", tag).replace("__LIMIT__", limit).replace("__JOBS__", open(jobs).read()), end="")
PY
# GPU capacity moves between zones: try each subnet until one has capacity.
for SUBNET in $GAMD_SUBNETS; do
  if aws ec2 run-instances --region $REGION --image-id $GAMD_AMI --instance-type ${TYPE:-g6.xlarge} \
    --iam-instance-profile Name=$GAMD_PROFILE \
    --network-interfaces "DeviceIndex=0,SubnetId=$SUBNET,Groups=$GAMD_SG,AssociatePublicIpAddress=true" \
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=60,VolumeType=gp3,DeleteOnTermination=true}" \
    --instance-initiated-shutdown-behavior terminate --user-data file:///tmp/gamd-ud.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=gamd-$TAG},{Key=Project,Value=ga-mound-detector}]" \
    --query 'Instances[0].[InstanceId,LaunchTime]' --output text 2>/tmp/gamd-launch.err; then exit 0; fi
  grep -q InsufficientInstanceCapacity /tmp/gamd-launch.err || { cat /tmp/gamd-launch.err; exit 1; }
  echo "no ${TYPE:-g6.xlarge} capacity in $SUBNET, trying next" >&2
done
exit 1
