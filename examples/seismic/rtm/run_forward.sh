# List of all pairs
PAIRS=(
    "mex_4_5/config.py"
    "mex_4_7/config.py"
    "mex_6_5/config.py"
    "mex_7_8/config.py"
    "mex_8_5/config.py"
    "mex_9_8/config.py"
)

ITER_RES=(
    "5"
    "5"
    "9"
    "5"
    "6"
    "4"
)

source ~/.bashrc
source /home/andrey/miniconda3/etc/profile.d/conda.sh
conda activate devito

for i in "${!PAIRS[@]}"; do
    PAIR="${PAIRS[i]}"
    iter="${ITER_RES[i]}"
    
    echo "Processing pair: $PAIR with iter: $iter"
    DIR_NAME=$(dirname "$PAIR")

    export DEVITO_LOGGING=INFO
    export DEVITO_LANGUAGE=openmp
    export DEVITO_PLATFORM=nvidiaX
    export DEVITO_ARCH=nvc
    
    echo "Starting iteration $iter for $PAIR"
    
    echo "Computing wavefields for iteration $iter..."
    python wavefield_computation.py --config "$PAIR" --iter "$iter"

    unset DEVITO_LOGGING
    unset DEVITO_LANGUAGE
    unset DEVITO_PLATFORM
    unset DEVITO_ARCH
done