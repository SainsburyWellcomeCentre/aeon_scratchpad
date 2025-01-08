#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --study-path           Full path to where the shared RDB should be created (required)"
    echo "  --db-name              Name of SQLite '.db' file (required)"
    echo "  --study-name           Optuna study name (required)"
    echo "  --labels-file          Path to SLEAP labels (required)"
    echo "  --slurm-output-dir     SLURM out & err dir (required)"
    echo "  --model-output-dir     Model output dir (required)"
    echo "  --partition            SLURM partition (default: gpu_branco)"
    echo "  --nodelist             SLURM node (default: gpu-sr675-34)"
    echo "  --n-tasks              Number of parallel SLURM tasks (default: 2)"
    echo "  --n-trials             Number of Optuna trials (default: 75)"
    echo "  --slurm-job-name       SLURM job name (default: par_optuna)"
    echo "  --save-outputs         Save outputs (default: False)"
    exit 1
}

# Parse arguments
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
    usage
fi

PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --study-path|--db-name|--study-name|--labels-file|--slurm-output-dir|--model-output-dir|--partition|--nodelist|--n-tasks|--n-trials|--slurm-job-name|--save-outputs)
            PYTHON_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Infinite loop to rerun the job on failure
while true; do
    echo "Submitting job..."

    # Run the Python command and capture the job ID
    PYTHON_CMD="python main.py ${PYTHON_ARGS[@]}"
    JOB_OUTPUT=$(eval $PYTHON_CMD)
    echo "$JOB_OUTPUT"

    # Extract job ID from the output
    JOB_ID=$(echo "$JOB_OUTPUT" | awk '{for(i=NF;i>=1;i--) if($i ~ /^[0-9]+$/) {print $i; break}}')
    echo "Extracted Job ID: $JOB_ID"

    if [[ -z "$JOB_ID" ]]; then
        echo "Failed to extract Job ID. Exiting."
        exit 1
    fi

    # Wait 20 seconds for .err and .out files to be created
    sleep 20

    # Print paths to all .err and .out files
    for ((i = 1; i <= ${#PYTHON_ARGS[@]}; i++)); do
        if [[ "${PYTHON_ARGS[$i]}" == "--slurm-output-dir" ]]; then
            SLURM_OUTPUT_DIR="${PYTHON_ARGS[$((i + 1))]}"
            break
        fi
    done
    ERR_FILES=(${SLURM_OUTPUT_DIR}/${JOB_ID}_*_log.err)
    OUT_FILES=(${SLURM_OUTPUT_DIR}/${JOB_ID}_*_log.out)

    echo ".err files for $JOB_ID:"
    for ERR_FILE in $ERR_FILES; do
        echo "  $ERR_FILE"
    done

    echo ".out files for $JOB_ID:"
    for OUT_FILE in $OUT_FILES; do
        echo "  $OUT_FILE"
    done

    # Monitor job logs
    echo "Monitoring logs for Job ID: $JOB_ID"
    while true; do
        ERR_FILES=(${SLURM_OUTPUT_DIR}/${JOB_ID}_*_log.err)
        OUT_FILES=(${SLURM_OUTPUT_DIR}/${JOB_ID}_*_log.out)
        
        # Print updates from .out files
        for OUT_FILE in $OUT_FILES; do
            if [[ -f "$OUT_FILE" ]]; then
                echo "----- latest log [$OUT_FILE] -----"
                tail -n 10 "$OUT_FILE"
                echo ""
            fi
        done

        # Check for errors in .err files
        for ERR_FILE in $ERR_FILES; do
            if grep -q "srun: error:" "$ERR_FILE"; then
                echo "Error detected in: $ERR_FILE"
                echo "Restarting, will submit new job..."
                tail -n 50 "$ERR_FILE"
                break 2
            fi
        done

        # Check for successful completion in .out files
        ALL_SUCCESS=true
        for OUT_FILE in $OUT_FILES; do
            if [[ -f "$OUT_FILE" ]]; then
                if ! grep -q "Job completed successfully" "$OUT_FILE"; then
                    ALL_SUCCESS=false
                    break
                fi
            fi
        done

        if $ALL_SUCCESS; then
            echo "All tasks completed successfully. Exiting."
            exit 0
        fi

        # Wait a bit before checking again
        sleep 30
    done
done
