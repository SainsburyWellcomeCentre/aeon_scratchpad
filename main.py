"""Parallelized Optuna hyperparameter optimization for SLEAP training."""

import argparse
import os
import sqlite3  # noqa: F401 (used in a os.system call) #type: ignore
import uuid
from functools import partial
from pathlib import Path

import optuna
import sleap
import submitit
from optuna.storages import RDBStorage
from sleap.nn.config import *  # noqa: F403
from sleap.nn.inference import TopDownMultiClassPredictor

# Constants
anchor_part = "centroid"

def create_cfg(optuna_params, labels_file, output_dir):
    """Create a SLEAP training job config with Optuna parameters."""
    # set initial parameters
    session_id = Path(labels_file).stem
    parent_dir = str(Path(labels_file).parent)
    unique_suffix = str(uuid.uuid4())[:8]
    run_name = session_id + "_topdown_top.centered_instance_multiclass_" + unique_suffix
    runs_folder = output_dir if output_dir is not None else parent_dir + "/models"
    labels = sleap.load_file(labels_file)

    cfg = TrainingJobConfig()  # noqa: F405 (import * above) # type: ignore
    cfg.data.labels.training_labels = parent_dir + "/" + session_id + ".train.pkg.slp"
    cfg.data.labels.validation_labels = parent_dir + "/" + session_id + ".val.pkg.slp"
    cfg.data.labels.validation_fraction = 0.1
    cfg.data.labels.skeletons = labels.skeletons

    cfg.data.preprocessing.input_scaling = 1.0
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
    cfg.model.backbone.unet = UNetConfig(  # noqa: F405 (import * above) # type: ignore
        max_stride=optuna_params["max_stride"],
        output_stride=optuna_params["output_stride"],
        filters=optuna_params["filters"],
        filters_rate=1.50,
        # up_interpolate=True, # save computations but may lower accuracy
    )
    confmaps = CenteredInstanceConfmapsHeadConfig(  # noqa: F405 (import * above) # type: ignore
        anchor_part=anchor_part,
        sigma=1.5,  # 2.5,
        output_stride=optuna_params["output_stride"],
        loss_weight=1.0,
    )
    class_vectors = ClassVectorsHeadConfig(  # noqa: F405 (import * above) # type: ignore
        classes=[track.name for track in labels.tracks],
        output_stride=optuna_params["output_stride"],
        num_fc_layers=3,
        num_fc_units=optuna_params["num_fc_units"],
        global_pool=optuna_params["global_pool"],
        loss_weight=optuna_params["class_vectors_loss_weight"],
    )
    cfg.model.heads.multi_class_topdown = (
        MultiClassTopDownConfig(  # noqa: F405 (import * above) # type: ignore
            confmaps=confmaps, class_vectors=class_vectors
        )
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


def compute_id_metrics(labels_gt, labels_pr, crop_size):
    """Compute average ID accuracy between ground truth and predicted labels."""
    framepairs = sleap.nn.evals.find_frame_pairs(labels_gt, labels_pr)
    matches = sleap.nn.evals.match_frame_pairs(framepairs, scale=crop_size)
    positive_pairs = matches[0]

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

    print("Total gt tracks:", len(labels_gt.tracks))
    print("Total pr tracks:", len(labels_pr.tracks))

    metrics = {}
    for i in range(2):
        track = track_names[i]
        other_track = track_names[1 - i]
        TP = correct_id[track].count(True)
        FN = correct_id[track].count(False)
        FP = correct_id[other_track].count(False)
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        accuracy = TP / (TP + FN + FP) if TP + FN + FP > 0 else 0
        metrics[track] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
        }
        print(f"Track: {track}")
        print(f"- Precision: {precision}")
        print(f"- Recall: {recall}")
        print(f"- Accuracy: {accuracy}")

    return metrics


def objective(trial: optuna.Trial, labels_file, model_output_dir, save_outputs) -> float:
    """Objective function for Optuna to optimise."""
    print(f"Starting trial {trial.number}.")
    # define parameters to optimise
    crop_size_suggest = trial.suggest_int("crop_size", 80, 128, step=16)
    initial_learning_rate_suggest = trial.suggest_float("initial_learning_rate", 1e-5, 1e-3, log=True)
    max_stride_suggest = trial.suggest_int("max_stride", 16, 32, step=8)
    filters_suggest = trial.suggest_int("filters", 16, 64, step=16)
    output_stride_suggest = trial.suggest_int("output_stride", 2, 4, step=2)
    num_fc_units_suggest = trial.suggest_int("num_fc_units", 128, 512, step=32)
    class_vectors_loss_weight_suggest = trial.suggest_float("class_vectors_loss_weight", 0.001, 1.0, log=True)
    # create config with selected params
    cfg = create_cfg(
        {
            "crop_size": crop_size_suggest,
            "initial_learning_rate": initial_learning_rate_suggest,
            "max_stride": max_stride_suggest,
            "filters": filters_suggest,
            "output_stride": output_stride_suggest,
            "num_fc_units": num_fc_units_suggest,
            "global_pool": True,
            "class_vectors_loss_weight": class_vectors_loss_weight_suggest,
        },
        labels_file,
        model_output_dir
    )
    trainer = sleap.nn.training.Trainer.from_config(cfg)
    trainer.setup()
    trainer.train()
    model_directory = f"{trainer.config.outputs.runs_folder}/{trainer.config.outputs.run_name}{trainer.config.outputs.run_name_suffix}"
    predictor = TopDownMultiClassPredictor.from_trained_models(
        confmap_model_path=model_directory,
    )
    labels_gt = sleap.load_file(f"{model_directory}/labels_gt.val.slp")
    labels_pr = predictor.predict(labels_gt)
    # get validation loss from last epoch to optimise
    history = trainer.keras_model.history
    last_epoch_val_loss = history.history["val_ClassVectorsHead_loss"][-1]
    # compute confusion matrix
    _id_metrics = compute_id_metrics(
        labels_gt,
        labels_pr,
        crop_size_suggest
    )
    if save_outputs:
        sleap.Labels.save_file(labels_pr, f"{model_directory}/labels_pr.val.slp")
    else:
        os.system(f"rm -r {model_directory}")  # noqa: S605
    return last_epoch_val_loss


def run_optuna_job(
        initialise,
        study_path,
        db_name,
        study_name,
        n_tasks,
        n_trials,
        labels_file,
        model_output_dir,
        save_outputs,
    ):
    """Creates and runs an Optuna study whose trials can be parallelized across processes."""
        # Ensure the study directory exists
    study_path = Path(study_path)
    study_path.mkdir(parents=True, exist_ok=True)

    # Define SQLite storage path
    db_path = study_path / db_name
    db_url = f"sqlite:////{db_path}"

    # Initialize SQLite database and serve with Datasette
    os.system(f"sqlite3 {db_path} 'VACUUM;'")  # noqa: S605
    os.system(f"datasette serve {db_path} &")  # noqa: S605

    # Create the Optuna study (if it doesn't already exist)
    slurm_procid = int(os.environ.get("SLURM_PROCID"))  # type: ignore
    print(f"SLURM_PROCID: {slurm_procid}")
    if slurm_procid != 0 and initialise:
        print("Waiting 30s for task 0 to create the database...")
        os.system("sleep 30")   # noqa: S605, S607
    storage = RDBStorage(db_url)
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="minimize", load_if_exists=True
    )

    # Print trials that already exist, if any
    if len(study.trials) > 0:
        print(f"Starting on trial {study.trials[-1].number}/{n_trials}.")
        print("Existing trials:")
        for trial in study.trials:
            print(f"Trial {trial.number}:")
            print(f"  Params: {trial.params}")
            print(f"  Value: {trial.value}")
            print(f"  State: {trial.state}")
            print(f"  Duration: {trial.duration}")
    if slurm_procid == 0 and initialise:
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
    partial_objective = partial(
        objective,
        labels_file=labels_file,
        model_output_dir=model_output_dir,
        save_outputs=save_outputs
    )
    study.optimize(partial_objective, n_trials=(n_trials // n_tasks))
    # Print all trial results
    print("Task completed.")
    print("All trials:")
    for trial in study.trials:
        print(f"Trial {trial.number}:")
        print(f"  Params: {trial.params}")
        print(f"  Value: {trial.value}")
        print(f"  State: {trial.state}")
        print(f"  Duration: {trial.duration}")
    print(f"Best params: {study.best_params}")


def main():
    """Parse command-line arguments and submit the sleap-optuna job."""
    parser = argparse.ArgumentParser(description="Run Optuna study with Submitit.")
    parser.add_argument(
        "--study-path",
        type=str,
        required=True,
        help="Full path to where the shared RDB should be created."
    )
    parser.add_argument(
        "--db-name",
        type=str,
        required=True,
        help="Name of SQLite '.db' file; e.g. 'db.db'"
    )
    parser.add_argument("--study-name", type=str, required=True, help="Optuna study name.")
    parser.add_argument("--labels-file", type=str, required=True, help="Path to SLEAP labels.")
    parser.add_argument("--slurm-output-dir", type=str, required=True, help="SLURM out & err dir.")
    parser.add_argument("--model-output-dir", type=str, required=True, help="Model output dir.")
    parser.add_argument("--partition", type=str, default="gpu_branco", help="SLURM partition.")
    parser.add_argument("--nodelist", type=str, default="gpu-sr675-34", help="SLURM node.")
    parser.add_argument("--n-tasks", type=int, default=2, help="Number of parallel SLURM tasks.")
    parser.add_argument("--n-trials", type=int, default=150, help="Number of Optuna trials.")
    parser.add_argument("--slurm-job-name", type=str, default="par_optuna", help="SLURM job name.")
    parser.add_argument("--save-outputs", type=bool, default=False, help="Save outputs.")
    args = parser.parse_args()

    # Set up Submitit executor
    output_dir = Path(args.slurm_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=str(output_dir))
    executor.update_parameters(
        slurm_job_name=args.slurm_job_name,
        tasks_per_node=args.n_tasks,
        slurm_partition=args.partition,
        slurm_gpus_per_task=1,
        cpus_per_task=16,
        mem_gb=64,
        slurm_time=60*48,
        slurm_additional_parameters={"nodelist": args.nodelist},
    )

    # Submit the job; catch exceptions and re-run if necessary.
    handled_error, initialize = True, True
    while handled_error:
        job = executor.submit(
            run_optuna_job,
            initialize,
            args.study_path,
            args.db_name,
            args.study_name,
            args.n_tasks,
            args.n_trials,
            args.labels_file,
            args.model_output_dir,
            args.save_outputs,
        )
        print(f"Submitted job ID: {job.job_id}")
        try:
            _result = job.result()  # hangs until job is finished
            handled_error = False
        except Exception as e:
            if (
                "Unable to load frame" in str(e)
                or "sqlite3.OperationalError: database is locked" in str(e)
            ):
                initialize = False
                os.system(f"scancel {job.job_id}")  # Job clean-up # type: ignore # noqa: S605
                print(f"Frame read or database error for {job.job_id}. See .err file for details.")
                continue  # continue loop; don't reraise exception here as it may break loop
            else:
                raise


if __name__ == "__main__":
    main()
