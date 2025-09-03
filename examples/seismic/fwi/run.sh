#!/bin/bash
MAX_ITER=100
START_ITER=0

if [ "$1" != "" ]; then
    START_ITER=$1
    echo "Starting from iteration $START_ITER"
fi

source ~/.bashrc
source /home/andrey/miniconda3/etc/profile.d/conda.sh
conda activate devito

for ((iter=$START_ITER; iter<$MAX_ITER; iter++)); do
    export DEVITO_LOGGING=INFO
    export DEVITO_LANGUAGE=openmp
    export DEVITO_PLATFORM=nvidiaX
    export DEVITO_ARCH=nvc
    
    echo "Starting iteration $iter"
    
    echo "Computing wavefields for iteration $iter..."
    python wavefield_computation.py --config mex_4_5/config.py --iter $iter

    unset DEVITO_LOGGING
    unset DEVITO_LANGUAGE
    unset DEVITO_PLATFORM
    unset DEVITO_ARCH
    
    echo "Computing gradients for iteration $iter..."
    python grad_computation.py --config mex_4_5/config.py --iter $iter --batch-size 8
    
    # echo "Updating image for iteration $iter..."
    python image_update.py --config mex_4_5/config.py --iter $iter
done
echo "FWI workflow completed successfully."
