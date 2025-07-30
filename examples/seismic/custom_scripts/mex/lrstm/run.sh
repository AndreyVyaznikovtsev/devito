#!/bin/bash
MAX_ITER=30

echo "Starting forward wavefield computation..."
export DEVITO_LOGGING=INFO
export DEVITO_LANGUAGE=openacc
export DEVITO_PLATFORM=nvidiaX
export DEVITO_ARCH=nvc

python wavefield_computation.py --mode forward

for ((iter=0; iter<$MAX_ITER; iter++)); do
    export DEVITO_LOGGING=INFO
    export DEVITO_LANGUAGE=openacc
    export DEVITO_PLATFORM=nvidiaX
    export DEVITO_ARCH=nvc
    
    echo "Starting iteration $iter"
    
    echo "Computing adjoint wavefields for iteration $iter..."
    python wavefield_computation.py --mode adjoint --iter $iter

    unset DEVITO_LOGGING
    unset DEVITO_LANGUAGE
    unset DEVITO_PLATFORM
    unset DEVITO_ARCH
    
    echo "Computing gradients for iteration $iter..."
    python grad_computation.py --iter $iter --batch-size 10
    
    echo "Updating image for iteration $iter..."
    python image_update.py --iter $iter
done
echo "LSRTM workflow completed successfully."