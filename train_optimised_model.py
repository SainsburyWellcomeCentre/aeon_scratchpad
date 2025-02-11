import argparse
import os
from pathlib import Path
import submitit
import optuna
import sleap
from sleap.nn.training import Trainer
from sleap.nn.config import TrainingJobConfig
from sleap.nn.config import *  # noqa: F403

def train_with_best_params(best_params, labels_file, output_dir, run_suffix):
    """Train a model using the best parameters from Optuna optimization."""
    session_id = Path(labels_file).stem
    run_name = f"{session_id}_topdown_top.centered_instance_multiclass_optimised_w_metrics{run_suffix}"
    if os.path.exists(f"{output_dir}/{run_name}/best_model.h5"):
        print(f"Model is already trained and saved in: {output_dir}/{run_name}")
        return
    runs_folder = Path(output_dir)

    labels = sleap.load_file(labels_file)

    cfg = TrainingJobConfig() # noqa: F405 (import * above) # type: ignore
    cfg.data.labels.training_labels = str(Path(labels_file).parent / f"{session_id}.train.pkg.slp")
    cfg.data.labels.validation_labels = str(Path(labels_file).parent / f"{session_id}.val.pkg.slp")
    cfg.data.labels.skeletons = labels.skeletons

    cfg.data.preprocessing.input_scaling = 1.0
    cfg.data.instance_cropping.center_on_part = "centroid"
    cfg.data.instance_cropping.crop_size = best_params["crop_size"]

    cfg.optimization.augmentation_config.rotate = True
    cfg.optimization.epochs = 600
    cfg.optimization.batch_size = 8

    cfg.optimization.initial_learning_rate = best_params["initial_learning_rate"]
    cfg.optimization.learning_rate_schedule.reduce_on_plateau = True
    cfg.optimization.learning_rate_schedule.plateau_patience = 20  # default is 5

    cfg.optimization.early_stopping.stop_training_on_plateau = True
    cfg.optimization.early_stopping.plateau_patience = 20

    # configure nn and model
    cfg.model.backbone.unet = UNetConfig(  # noqa: F405 (import * above) # type: ignore
        max_stride=best_params["max_stride"],
        output_stride=best_params["output_stride"],
        filters=best_params["filters"],
        filters_rate=1.50,
        # up_interpolate=True, # save computations but may lower accuracy
    )
    confmaps = CenteredInstanceConfmapsHeadConfig(  # noqa: F405 (import * above) # type: ignore
        anchor_part="centroid",
        sigma=1.5,  # 2.5,
        output_stride=best_params["output_stride"],
        loss_weight=1.0,
    )
    class_vectors = ClassVectorsHeadConfig(  # noqa: F405 (import * above) # type: ignore
        classes=[track.name for track in labels.tracks],
        output_stride=best_params["output_stride"],
        num_fc_layers=3,
        num_fc_units=best_params["num_fc_units"],
        global_pool=True,
        loss_weight=best_params["class_vectors_loss_weight"],
    )
    cfg.model.heads.multi_class_topdown = (
        MultiClassTopDownConfig(  # noqa: F405 (import * above) # type: ignore
            confmaps=confmaps, class_vectors=class_vectors
        )
    )
    # configure outputs
    cfg.outputs.run_name = run_name
    cfg.outputs.save_outputs = True
    cfg.outputs.runs_folder = str(runs_folder)
    cfg.outputs.save_visualizations = True
    cfg.outputs.delete_viz_images = False
    cfg.outputs.checkpointing.initial_model = True
    cfg.outputs.checkpointing.best_model = True

    trainer = Trainer.from_config(cfg)
    trainer.setup()
    trainer.train()

    print(f"Model trained and saved in: {runs_folder}")
    
    

def fetch_top_n_params_from_study(study_name, db_path, n=3):
    """Fetch the best parameters from an Optuna study."""
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    top_trials = sorted(completed_trials, key=lambda trial: trial.value, reverse=True)[:n]
    return [trial.params for trial in top_trials]

def main():
    """Main function to submit the training job."""
    parser = argparse.ArgumentParser(description="Submit SLEAP training job with best parameters from an Optuna study.")
    parser.add_argument("--study-name", type=str, required=True, 
                        help="Name of the Optuna study.")
    parser.add_argument("--db-path", type=str, required=True, 
                        help="Path to the Optuna study SQLite database.")
    parser.add_argument("--labels-file", type=str, required=True, 
                        help="Path to the SLEAP labels file.")
    parser.add_argument("--model-output-dir", type=str, required=True, 
                        help="Directory to save the trained model.")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of top trials to train a full sleep model for.")
    parser.add_argument("--slurm-output-dir", type=str, required=True, 
                        help="Directory to store SLURM logs.")
    parser.add_argument("--slurm-job-name", type=str, default="train_optimised_model",
                        help="Name of the SLURM job.")
    parser.add_argument("--partition", type=str, default="a100",
                        help="SLURM partition to submit the job.")

    args = parser.parse_args()

    executor = submitit.AutoExecutor(folder=args.slurm_output_dir)
    executor.update_parameters(
        slurm_job_name=args.slurm_job_name,
        tasks_per_node=1,
        slurm_partition=args.partition,
        gpus_per_node=1,
        cpus_per_task=16,
        mem_gb=256,
        slurm_time=60 * 24,  # 10 hours
    )

    def job_function():
        top_params = fetch_top_n_params_from_study(args.study_name, args.db_path, n=args.top_n)
        if len(top_params) == 1:
            run_suffix = ""
            train_with_best_params(top_params[0], args.labels_file, args.model_output_dir, run_suffix)
            print(f"Training completed.")
            os.system(f"scancel {os.environ['SLURM_JOB_ID']}")
        else:
            for i, params in enumerate(top_params):
                run_suffix = f"_{i+1}"

                train_with_best_params(params, args.labels_file, args.model_output_dir, run_suffix)
                if i == len(top_params) - 1:
                    print(f"Training completed for top {args.top_n} trials.")
                    os.system(f"scancel {os.environ['SLURM_JOB_ID']}")


    job = executor.submit(job_function)
    print(f"Submitted job ID: {job.job_id}")

if __name__ == "__main__":
    main()
