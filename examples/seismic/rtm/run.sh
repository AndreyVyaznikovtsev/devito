#!/bin/bash
MAX_ITER=10
START_ITER=0

# List of all pairs
PAIRS=(
    # "lax_80_68/config.py"
    # "lax_80_61/config.py"
    # "mex_4_5/config.py"
    # "mex_4_7/config.py"
    # "mex_6_5/config.py"
    # "mex_7_8/config.py"
    # "mex_8_5/config.py"
    "mex_9_8/config.py"
)

if [ "$1" != "" ]; then
    START_ITER=$1
    echo "Starting from iteration $START_ITER"
fi

source ~/.bashrc
source /home/andrey/miniconda3/etc/profile.d/conda.sh
conda activate devito

# Loop through each pair
for PAIR in "${PAIRS[@]}"; do
    echo "Processing pair: $PAIR"
    DIR_NAME=$(dirname "$PAIR")

    for ((iter=$START_ITER; iter<$MAX_ITER; iter++)); do
        export DEVITO_LOGGING=INFO
        export DEVITO_LANGUAGE=openmp
        export DEVITO_PLATFORM=nvidiaX
        export DEVITO_ARCH=nvc
        
        echo "Starting iteration $iter for $PAIR"
        
        echo "Computing wavefields for iteration $iter..."
        python wavefield_computation.py --config $PAIR --iter $iter

        unset DEVITO_LOGGING
        unset DEVITO_LANGUAGE
        unset DEVITO_PLATFORM
        unset DEVITO_ARCH
        
        echo "Computing gradients for iteration $iter..."
        python grad_computation.py --config $PAIR --iter $iter --batch-size 4
        
        # echo "Updating image for iteration $iter..."
        python image_update.py --config $PAIR --iter $iter
    done
    
    echo "Completed RTM workflow for $PAIR"
    
    # Cleanup: Delete the snapshots folders
    echo "Cleaning up snapshots folders for $DIR_NAME..."
    FORWARD_SNAPS="$DIR_NAME/forward_snaps"
    ADJOINT_SNAPS="$DIR_NAME/adjoint_snaps"
    
    if [ -d "$FORWARD_SNAPS" ]; then
        echo "Deleting $FORWARD_SNAPS"
        rm -rf "$FORWARD_SNAPS"
    else
        echo "$FORWARD_SNAPS does not exist or already deleted"
    fi
    
    if [ -d "$ADJOINT_SNAPS" ]; then
        echo "Deleting $ADJOINT_SNAPS"
        rm -rf "$ADJOINT_SNAPS"
    else
        echo "$ADJOINT_SNAPS does not exist or already deleted"
    fi
    
    echo "Cleanup completed for $DIR_NAME"
done

echo "All RTM workflows completed successfully."