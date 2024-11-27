import argparse
import os
import sqlite3
from pathlib import Path

import optuna
import submitit


def run_optuna_job(study_path, n_tasks, n_trials):
    """Creates and runs an Optuna study whose trials can be parallelized across processes."""
    # Ensure the study directory exists
    study_path = Path(study_path)
    study_path.mkdir(parents=True, exist_ok=True)

    # Define SQLite storage path
    db_path = study_path / "db.db"
    db_url = f"sqlite:////{db_path}"

    # Initialize SQLite database and serve with Datasette
    os.system(f"sqlite3 {db_path} 'VACUUM;'")  # noqa: S605
    os.system(f"datasette serve {db_path} &")  # noqa: S605

    # Create the Optuna study (if it doesn't already exist)
    try:
        optuna.create_study(study_name="par_optuna_trials", storage=db_url, direction="minimize")
    except (optuna.exceptions.DuplicatedStudyError, sqlite3.OperationalError):
        print("Study already exists. Loading existing study.")

    # Load the study and optimize
    study = optuna.load_study(study_name="par_optuna_trials", storage=db_url)
    study.optimize(objective, n_trials=(n_trials // n_tasks))  # divide trials across tasks
    print(f"Task completed. Best params: {study.best_params}")


def objective(trial):
    """Objective function for Optuna."""
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def main():
    """Main function for setting up Submitit and submitting the job."""
    parser = argparse.ArgumentParser(description="Run Optuna study with Submitit.")
    parser.add_argument(
        "--study-path",
        type=str,
        required=True,
        help="Full path to where the shared RDB should be created."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="SLURM out and err dir.")
    parser.add_argument("--partition", type=str, default="gpu_branco", help="SLURM partition.")
    parser.add_argument("--n-tasks", type=int, default=2, help="Number of parallel SLURM tasks.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials.")
    args = parser.parse_args()

    # Set up Submitit executor
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(output_dir))
    executor.update_parameters(
        slurm_job_name="par_optuna_trials",
        slurm_partition=args.partition,
        nodes=args.n_tasks,  # nodes correspond to tasks
    )

    # Submit the job
    job = executor.submit(run_optuna_job, args.study_path, args.n_tasks, args.n_trials)
    print(f"Submitted job ID: {job.job_id}")


if __name__ == "__main__":
    main()
