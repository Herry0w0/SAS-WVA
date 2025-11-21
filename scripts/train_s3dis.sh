#!/bin/bash

# Default values
NPROC_PER_NODE=4
VISIBLE_DEVICES="0,1,2,3"
CONFIG_FILE="./config/s3dis.yaml"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --nproc_per_node N    Number of processes per node (default: 2)"
    echo "  --visible_devices D   CUDA devices to use (default: 1,2)"
    echo "  --config FILE         Config file path"
    echo "  -h, --help           Show this help message"
    exit 0
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nproc_per_node)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --nproc_per_node requires a value"
                exit 1
            fi
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --visible_devices)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --visible_devices requires a value"
                exit 1
            fi
            VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --config)
            if [[ -z "$2" ]] || [[ "$2" == --* ]]; then
                echo "Error: --config requires a value"
                exit 1
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown parameter: $1"
            usage
            ;;
    esac
done

# Validate config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Validate device count
IFS=',' read -ra DEVICES <<< "$VISIBLE_DEVICES"
NUM_DEVICES=${#DEVICES[@]}
if [[ $NPROC_PER_NODE -gt $NUM_DEVICES ]]; then
    echo "Warning: nproc_per_node ($NPROC_PER_NODE) exceeds visible devices ($NUM_DEVICES)"
    echo "Setting nproc_per_node to $NUM_DEVICES"
    NPROC_PER_NODE=$NUM_DEVICES
fi

# Display configuration
echo "Starting distributed training with:"
echo "  Processes: $NPROC_PER_NODE"
echo "  Devices: $VISIBLE_DEVICES"
echo "  Config: $CONFIG_FILE"

# Set the visible CUDA devices
export CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES
export TORCH_CUDA_ARCH_LIST="8.0"
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

# Launch distributed training
torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:29501" \
    train.py --config "$CONFIG_FILE"
