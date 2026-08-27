#!/bin/bash
# Legacy AWS launch helper for the old Ollama-based path.
# Not reconciled with the current llama-server contract.
# Usage: bash llm/deploy/aws_launch.sh
set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="g4dn.xlarge"  # 1x T4 16GB, 4 vCPUs, 16GB RAM
KEY_NAME="llm-agora"
SG_NAME="llm-agora-sg"
MAX_SPOT_PRICE="0.20"  # T4 spot is ~$0.16/hr

echo "=== Setting up AWS infrastructure ==="

# 1. Create key pair (if not exists)
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "Creating key pair..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' --output text > llm/deploy/${KEY_NAME}.pem
    chmod 600 llm/deploy/${KEY_NAME}.pem
    echo "  Key saved: llm/deploy/${KEY_NAME}.pem"
else
    echo "  Key pair '$KEY_NAME' already exists"
fi

# 2. Create security group (if not exists)
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters Name=isDefault,Values=true \
    --query 'Vpcs[0].VpcId' --output text)

SG_ID=$(aws ec2 describe-security-groups \
    --filters Name=group-name,Values="$SG_NAME" \
    --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "LLM Agora - SSH + Ollama" \
        --vpc-id "$VPC_ID" \
        --region "$REGION" \
        --query 'GroupId' --output text)
    # SSH
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" --protocol tcp --port 22 \
        --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
    # Ollama API
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" --protocol tcp --port 11434 \
        --cidr 0.0.0.0/0 --region "$REGION" > /dev/null
    echo "  Security group: $SG_ID"
else
    echo "  Security group already exists: $SG_ID"
fi

# 3. Find latest Ubuntu 22.04 AMI with NVIDIA drivers
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters \
        "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
        "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)
echo "  AMI: $AMI_ID"

# 4. User data script — installs Ollama + pulls model on boot
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -ex

# Install NVIDIA drivers
apt-get update -qq
apt-get install -y -qq nvidia-driver-535 nvidia-utils-535

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama to listen on all interfaces
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_FLASH_ATTENTION=1"
EOF

systemctl daemon-reload
systemctl enable ollama
systemctl start ollama

# Wait for Ollama to be ready, then pull model
sleep 10
for i in $(seq 1 30); do
    curl -sf http://localhost:11434/api/tags && break
    sleep 5
done

ollama pull gemma4:e2b

# Signal ready
touch /tmp/ollama-ready
USERDATA
)

# 5. Launch spot instance
echo ""
echo "=== Launching spot instance ==="
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$MAX_SPOT_PRICE"'","SpotInstanceType":"one-time"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=llm-agora}]' \
    --query 'Instances[0].InstanceId' --output text)

echo "  Instance: $INSTANCE_ID"
echo "  Waiting for public IP..."

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo ""
echo "=== INSTANCE READY ==="
echo "  Instance ID: $INSTANCE_ID"
echo "  Public IP:   $PUBLIC_IP"
echo "  SSH:         ssh -i llm/deploy/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
echo "  Ollama API:  http://${PUBLIC_IP}:11434"
echo ""
echo "  The instance is installing drivers + pulling the model."
echo "  This takes ~5-10 min. Check progress with:"
echo "    ssh -i llm/deploy/${KEY_NAME}.pem ubuntu@${PUBLIC_IP} 'tail -f /var/log/cloud-init-output.log'"
echo ""
echo "  Once ready, update your client:"
echo "    LLMClient(base_url='http://${PUBLIC_IP}:11434')"
echo ""
echo "  To terminate when done:"
echo "    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"

# Save instance info
cat > llm/deploy/instance.json <<EOF
{
  "instance_id": "$INSTANCE_ID",
  "public_ip": "$PUBLIC_IP",
  "region": "$REGION",
  "type": "$INSTANCE_TYPE",
  "ollama_url": "http://${PUBLIC_IP}:11434",
  "ssh_cmd": "ssh -i llm/deploy/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}",
  "terminate_cmd": "aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
}
EOF
echo "  Instance info saved: llm/deploy/instance.json"
