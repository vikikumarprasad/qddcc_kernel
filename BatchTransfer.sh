#!/bin/bash

# Function to convert total seconds into DD-HH:MM:SS format
format_seconds_to_ddhhmmss() {
    local total_seconds=$1
    local days=0
    local hours=0
    local minutes=0
    local seconds=0

    if (( total_seconds < 0 )); then
        echo "Error: Input seconds cannot be negative." >&2
        return 1
    fi

    # Calculate days
    days=$(( total_seconds / (24 * 3600) ))
    remaining_seconds=$(( total_seconds % (24 * 3600) ))

    # Calculate hours
    hours=$(( remaining_seconds / 3600 ))
    remaining_seconds=$(( remaining_seconds % 3600 ))

    # Calculate minutes
    minutes=$(( remaining_seconds / 60 ))
    seconds=$(( remaining_seconds % 60 ))

    # Use printf for zero-padding (e.g., 05 instead of 5)
    printf "%02d-%02d:%02d:%02d\n" "$days" "$hours" "$minutes" "$seconds"
}
STRUCTURES=("Water") #"Ammonia" "Methane" "Ethane" "Formaldehyde" "Methanol" "Ethanol" "Dimethylether" "Benzene")
BASIS_SETS=("STO-3G")
ENCODING_DEGREES=(0)
ENCODING_CIRCUITS=("ParamZ") #"Chebyshev" "ZZ_map" "YZ_CX" "HighDim" "MultiControl" "Hubregtsen")
NUM_LAYERS=(1)
MODEL_TYPES=("qsvr" "qkrr")
export RUN_DATE="04-Jul-2025" #date +”%d-%b-%Y”
export MOLECULE="Water"
export DATA_SETUP="Water-Scaled"
export NUM_FEATURES=5
export HYPERPARAMETERS="OPTIMIZED"
export NUM_QUBITS=5
export QKernel="Projected"
# export ENCODING_DEGREE=0
# export ENCODING_CIRCUIT="HighDim"
# export NUM_LAYERS=1
# export MODEL_TYPE="qsvr"
for STRUCTURE in ${STRUCTURES[@]}
do
    export STRUCTURE=$STRUCTURE
    for BASIS in ${BASIS_SETS[@]}
    do
        for ENCODING_CIRCUIT in ${ENCODING_CIRCUITS[@]}
        do
            export ENCODING_CIRCUIT=$ENCODING_CIRCUIT
            for ENCODING_DEGREE in ${ENCODING_DEGREES[@]}
            do
                export ENCODING_DEGREE=$ENCODING_DEGREE
                for NUM_LAYER in ${NUM_LAYERS[@]}
                do
                    export NUM_LAYER=$NUM_LAYER
                    for MODEL_TYPE in ${MODEL_TYPES[@]}
                    do
                        export MODEL_TYPE=$MODEL_TYPE
                        echo $MODEL_TYPE $ENCODING_CIRCUIT-$ENCODING_DEGREE-$NUM_LAYER
                        SECONDS_TIME=$((3600))
                        MEMORY="8G"
                        JOB_TIME=$(format_seconds_to_ddhhmmss "$SECONDS_TIME")
                        sbatch --time=$JOB_TIME --mem-per-cpu=$MEMORY --job-name=$DATA_SETUP-$HYPERPARAMETERS-$BASIS-$STRUCTURE-$ENCODING_CIRCUIT-$ENCODING_DEGREE-$NUM_LAYERS-$MODEL_TYPE RunTransfer.sh
                    done
                done
            done
        done
    done
done
