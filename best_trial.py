import optuna
import argparse
from pathlib import Path


def display_all_trials(study_name: str, db_path: str):
    """Display the best trial and statistics from the given Optuna study."""
    storage_url = f"sqlite:///{db_path}"

    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"Study: {study_name}")
    print(f"Number of Trials: {len(study.trials)}")
    print(f"Best Trial:")

    best_trial = study.best_trial

    print(f"  Trial ID: {best_trial.number}")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\nAll Trials Summary:")
    for trial in study.trials:
        print(f"  Trial ID: {trial.number}")
        print(f"    Value: {trial.value}")
        print(f"    State: {trial.state}")
        print(f"    Duration: {trial.duration}")
        print("    Params:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")

def display_top_n_trials(study_name: str, db_path: str, n: int):
    """Display the top N trials and statistics from the given Optuna study."""
    storage_url = f"sqlite:///{db_path}"

    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"Study: {study_name}")
    print(f"Number of Trials: {len(study.trials)}")
    print(f"Top {n} Trials:")

    completed_trials = [trial for trial in study.trials if trial.value is not None]
    top_trials = sorted(completed_trials, key=lambda trial: trial.value, reverse=True)[:n]

    for i, trial in enumerate(top_trials):
        print(f"  Trial {i + 1}")
        print(f"    Trial ID: {trial.number}")
        print(f"    Value: {trial.value}")
        print("    Params:")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Display the best trial and stats from an Optuna study.")
    parser.add_argument("--study-name", type=str, required=True, help="The name of the Optuna study.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the SQLite database file.")
    args = parser.parse_args()

    study_name = args.study_name
    db_path = args.db_path

    # Validate database path
    if not Path(db_path).is_file():
        print(f"Error: Database file '{db_path}' does not exist.")
        return

    # Display the best trial and stats
    # display_all_trials(study_name, db_path)
    display_top_n_trials(study_name, db_path, 20)


if __name__ == "__main__":
    main()