"""Parallelized Optuna hyperparameter optimization for SLEAP training."""

import argparse
import os
import sqlite3
from functools import partial
from pathlib import Path

import optuna
import sleap
import submitit
from sleap.nn.config import *

# Constants
model_type = "centered_instance_multiclass"  # or "centroid"
anchor_part = "centroid"


def create_cfg(optuna_params, labels_file):
    """Create a SLEAP training job config with Optuna parameters."""
    # set initial parameters
    session_id = Path(labels_file).stem
    parent_dir = str(Path(labels_file).parent)
    if model_type == "centroid":
        run_name = session_id + "_topdown_top.centroid"
    elif model_type == "centered_instance_multiclass":
        run_name = session_id + "_topdown_top.centered_instance_multiclass"
    runs_folder = parent_dir + "/models"
    labels = sleap.load_file(labels_file)

    cfg = TrainingJobConfig()
    cfg.data.labels.training_labels = parent_dir + "/" + session_id + ".train.pkg.slp"
    cfg.data.labels.validation_labels = parent_dir + "/" + session_id + ".val.pkg.slp"
    cfg.data.labels.validation_fraction = 0.1
    cfg.data.labels.skeletons = labels.skeletons

    cfg.data.preprocessing.input_scaling = (
        optuna_params["input_scaling"] if model_type == "centroid" else 1.0
    )

    cfg.data.instance_cropping.center_on_part = anchor_part
    cfg.data.instance_cropping.crop_size = optuna_params["crop_size"]

    cfg.optimization.augmentation_config.rotate = True
    cfg.optimization.epochs = 10
    cfg.optimization.batch_size = 8  # 4

    cfg.optimization.initial_learning_rate = optuna_params["initial_learning_rate"]
    cfg.optimization.learning_rate_schedule.reduce_on_plateau = True
    cfg.optimization.learning_rate_schedule.plateau_patience = 20  # default is 5

    cfg.optimization.early_stopping.stop_training_on_plateau = True
    cfg.optimization.early_stopping.plateau_patience = 10  # default is 10

    # configure nn and model
    cfg.model.backbone.unet = UNetConfig(
        max_stride=optuna_params["max_stride"],
        output_stride=2,
        filters=optuna_params["filters"],
        filters_rate=2.00 if model_type == "centroid" else 1.50,
        # up_interpolate=True, # save computations but may lower accuracy
    )
    if model_type == "centroid":
        cfg.model.heads.centroid = CentroidsHeadConfig(
            anchor_part=anchor_part, sigma=2.5, output_stride=2
        )
    else:
        confmaps = CenteredInstanceConfmapsHeadConfig(
            anchor_part=anchor_part,
            sigma=1.5,  # 2.5,
            output_stride=2,  # 4,
            loss_weight=1.0,
        )
        class_vectors = ClassVectorsHeadConfig(
            classes=[track.name for track in labels.tracks],
            output_stride=optuna_params["output_stride"],
            num_fc_layers=3,
            num_fc_units=optuna_params["num_fc_units"],
            global_pool=optuna_params["global_pool"],
            loss_weight=optuna_params["class_vectors_loss_weight"],
        )
        cfg.model.heads.multi_class_topdown = MultiClassTopDownConfig(
            confmaps=confmaps, class_vectors=class_vectors
        )
    # configure outputs
    cfg.outputs.run_name = run_name
    cfg.outputs.save_outputs = True
    cfg.outputs.runs_folder = runs_folder
    cfg.outputs.save_visualizations = True
    cfg.outputs.delete_viz_images = False
    cfg.outputs.checkpointing.initial_model = True
    cfg.outputs.checkpointing.best_model = True
    return cfg


def compute_id_accuracy(labels_gt_path, labels_pr_path, crop_size):
    """Compute average ID accuracy between ground truth and predicted labels."""
    labels_gt = sleap.load_file(labels_gt_path)
    labels_pr = sleap.load_file(labels_pr_path)
    framepairs = sleap.nn.evals.find_frame_pairs(labels_gt, labels_pr)
    matches = sleap.nn.evals.match_frame_pairs(framepairs, scale=crop_size)
    positive_pairs = matches[0]
    false_negatives = matches[1]

    track_names = [track.name for track in labels_gt.tracks]
    correct_id = {track_name: [] for track_name in track_names}

    for positive_pair in positive_pairs:
        gt = (
            positive_pair[0]
            if isinstance(positive_pair[1], sleap.PredictedInstance)
            else positive_pair[1]
        )
        correct_id[gt.track.name].append(
            positive_pair[0].track.name == positive_pair[1].track.name
        )

    track_sums = sum(sum(correct_id[track]) for track in track_names)
    track_lens = sum(len(correct_id[track]) for track in track_names)

    for track in track_names:
        accuracy = round(sum(correct_id[track]) / len(correct_id[track]), 3)
        print(f"- {track} Accuracy: {accuracy}")

    avg_accuracy = track_sums / track_lens
    print("- ID accuracy:", round(avg_accuracy, 3))
    print("- Total tracks:", len(labels_gt.all_instances))
    print("- Tracks identified:", len(labels_pr.all_instances))
    print("- Tracks correctly identified:", track_sums)
    return avg_accuracy


def objective(trial: optuna.Trial, labels_file) -> float:
    """Objective function for Optuna to optimise."""
    # define parameters to optimise
    crop_size_suggest = trial.suggest_int("crop_size", 80, 160, step=16)
    initial_learning_rate_suggest = trial.suggest_float(
        "initial_learning_rate", 1e-5, 1e-2, log=True
    )
    input_scaling_suggest = trial.suggest_float(
        "input_scaling", 0.5, 1.0, step=0.25
    )  # only for centroid model
    max_stride_suggest = trial.suggest_int("max_stride", 16, 32, step=16)
    filters_suggest = trial.suggest_int("filters", 16, 64, step=16)
    output_stride_suggest = trial.suggest_categorical("output_stride", [1, 2, 4])
    num_fc_units_suggest = trial.suggest_int("num_fc_units", 192, 448, step=64)
    global_pool_suggest = trial.suggest_categorical("global_pool", [True, False])
    class_vectors_loss_weight_suggest = trial.suggest_float(
        "class_vectors_loss_weight", 0.001, 1.0, log=True
    )
    # create config with selected params
    cfg = create_cfg(
        {
            "crop_size": crop_size_suggest,
            "initial_learning_rate": initial_learning_rate_suggest,
            "input_scaling": input_scaling_suggest,  # only for centroid model
            "max_stride": max_stride_suggest,
            "filters": filters_suggest,
            "output_stride": output_stride_suggest,
            "num_fc_units": num_fc_units_suggest,
            "global_pool": global_pool_suggest,
            "class_vectors_loss_weight": class_vectors_loss_weight_suggest,
        },
        labels_file,
    )
    trainer = sleap.nn.training.Trainer.from_config(cfg)
    trainer.setup()
    trainer.train()
    # get validation loss from last epoch to optimise
    history = trainer.keras_model.history
    last_epoch_val_loss = history.history["val_loss"][-1]
    # compute confusion matrix
    path_prefix = f"{trainer.config.outputs.runs_folder}/{trainer.config.outputs.run_name}{trainer.config.outputs.run_name_suffix}"
    accuracy = compute_id_accuracy(
        f"{path_prefix}/labels_val_gt.val.slp",
        f"{path_prefix}/labels_val_pr.val.slp",
        crop_size_suggest
    )
    return last_epoch_val_loss


def run_optuna_job(study_path, n_tasks, n_trials, labels_file):
    """Creates and runs an Optuna study whose trials can be parallelized across processes."""
    # Ensure the study directory exists
    study_path = Path(study_path)
    study_path.mkdir(parents=True, exist_ok=True)

    # Define SQLite storage path
    db_path = study_path / "db.db"
    db_url = f"sqlite:////{db_path}"

    # Initialize SQLite database and serve with Datasette
    # connection = sqlite3.connect(db_path)  # create db
    # connection.execute("PRAGMA journal_mode=WAL;")  # enable WAL for concurrent r/w
    os.system(f"sqlite3 {db_path} 'VACUUM;'")  # noqa: S605
    os.system(f"datasette serve {db_path} &")  # noqa: S605

    # Create the Optuna study (if it doesn't already exist)
    try:
        optuna.create_study(study_name="par_optuna_trials", storage=db_url, direction="minimize")
    except (optuna.exceptions.DuplicatedStudyError, sqlite3.OperationalError):
        print("Study already exists. Loading existing study.")

    # Load the study, enqueue the first trial, and optimize.
    study = optuna.load_study(study_name="par_optuna_trials", storage=db_url)
    study.enqueue_trial(
        {
            "crop_size": 112,
            "initial_learning_rate": 0.0001,
            "input_scaling": 1.0,
            "max_stride": 16,
            "filters": 32,
            "output_stride": 2,
            "num_fc_units": 256,
            "global_pool": True,
            "class_vectors_loss_weight": 0.001,
        }
    )
    # Divide trials across tasks
    partial_objective = partial(objective, labels_file=labels_file)
    study.optimize(partial_objective, n_trials=(n_trials // n_tasks))
    print(f"Task completed. Best params: {study.best_params}")


def main():
    parser = argparse.ArgumentParser(description="Run Optuna study with Submitit.")
    parser.add_argument(
        "--study-path",
        type=str,
        required=True,
        help="Full path to where the shared RDB should be created."
    )
    parser.add_argument("--labels-file", type=str, required=True, help="Path to SLEAP labels.")
    parser.add_argument("--output-dir", type=str, required=True, help="SLURM out and err dir.")
    parser.add_argument("--partition", type=str, default="gpu_branco", help="SLURM partition.")
    # parser.add_argument("--nodelist", type=str, default="gpu-sr675-34", help="SLURM node.")
    parser.add_argument("--n-tasks", type=int, default=2, help="Number of parallel SLURM tasks.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials.")
    args = parser.parse_args()

    # Set up Submitit executor
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(output_dir))
    executor.update_parameters(
        slurm_job_name="par_optuna_trials",
        nodes=args.n_tasks,  # nodes correspond to tasks
        slurm_partition=args.partition
    )

    # Submit the job
    job = executor.submit(
        run_optuna_job,
        args.study_path,
        args.n_tasks,
        args.n_trials,
        args.labels_file
    )
    print(f"Submitted job ID: {job.job_id}")


if __name__ == "__main__":
    main()
