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
    echo "$(date '+%Y-%m-%d %H:%M:%S') Submitting job..."

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

    # Wait for .err and .out files to be created
    MAX_RETRIES=10
    RETRY_COUNT=0
    while true; do
        ERR_FILES=($(find "${SLURM_OUTPUT_DIR}" -type f -name "${JOB_ID}_*_log.err"))
        OUT_FILES=($(find "${SLURM_OUTPUT_DIR}" -type f -name "${JOB_ID}_*_log.out"))

        if [[ ${#ERR_FILES[@]} -gt 0 && ${#OUT_FILES[@]} -gt 0 ]]; then
            echo "Found log files, continuing."
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Files not found, waiting... (retry: $RETRY_COUNT / $MAX_RETRIES)"
            sleep 15
        fi

        if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') Max retries reached. Exiting."
            exit 1
        fi
    done

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
                echo "----- $(date '+%Y-%m-%d %H:%M:%S') latest log [$OUT_FILE] -----"
                tail -n 10 "$OUT_FILE"
                echo ""
            fi
        done

        # Check for errors in .err files
        for ERR_FILE in $ERR_FILES; do
            if grep -q "srun: error:" "$ERR_FILE"; then
                echo "$(date '+%Y-%m-%d %H:%M:%S') Error detected in: $ERR_FILE"
                tail -n 50 "$ERR_FILE"
                echo -e "\n\n$(date '+%Y-%m-%d %H:%M:%S') Restarting, will submit new job..."
                break 2
            fi
        done

        # Check for successful completion in .out files
        ALL_SUCCESS=false
        for OUT_FILE in $OUT_FILES; do
            if [[ -f "$OUT_FILE" ]]; then
                if grep -q "Job completed successfully" "$OUT_FILE"; then
                    ALL_SUCCESS=true
                else
                    ALL_SUCCESS=false
                    break
                fi
            fi
        done

        if $ALL_SUCCESS; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') All tasks completed successfully. Exiting."
            exit 0
        fi

        # Wait a bit before checking again
        sleep 30
    done
done
