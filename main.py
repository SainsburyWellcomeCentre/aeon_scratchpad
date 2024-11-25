from pathlib import Path

import optuna
import sleap
from sleap.nn.config import *

labels_file = (
    "/ceph/aeon/aeon/code/scratchpad/sleap/multi_point_tracking/multi_animal_CameraNSEW/aeon3_social03_ceph.slp"
)
model_type = "centered_instance_multiclass"  # or "centroid"
anchor_part = "centroid"


def create_cfg(optuna_params):
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


def objective(trial: optuna.Trial) -> float:
    # define parameters to optimise
    crop_size_suggest = trial.suggest_int("crop_size", 80, 160, step=16)
    initial_learning_rate_suggest = trial.suggest_float("initial_learning_rate", 1e-5, 1e-2, log=True)
    input_scaling_suggest = trial.suggest_float("input_scaling", 0.5, 1.0, step=0.25) # only for centroid model
    max_stride_suggest = trial.suggest_int("max_stride", 16, 32, step=16)
    filters_suggest = trial.suggest_int("filters", 16, 64, step=16)
    output_stride_suggest = trial.suggest_categorical("output_stride", [1, 2, 4])
    num_fc_units_suggest = trial.suggest_int("num_fc_units", 192, 448, step=64)
    global_pool_suggest = trial.suggest_categorical("global_pool", [True, False])
    class_vectors_loss_weight_suggest = trial.suggest_float("class_vectors_loss_weight", 0.001, 1.0, log=True)
    # create config with selected params
    cfg = create_cfg(
        {
            "crop_size": crop_size_suggest,
            "initial_learning_rate": initial_learning_rate_suggest,
            "input_scaling": input_scaling_suggest, # only for centroid model
            "max_stride": max_stride_suggest,
            "filters": filters_suggest,
            "output_stride": output_stride_suggest,
            "num_fc_units": num_fc_units_suggest,
            "global_pool": global_pool_suggest,
            "class_vectors_loss_weight": class_vectors_loss_weight_suggest,
        }
    )

    trainer = sleap.nn.training.Trainer.from_config(cfg)
    trainer.setup()
    trainer.train()

    # return validation metric to optimise
    path_prefix = f"{trainer.config.outputs.runs_folder}/{trainer.config.outputs.run_name}{trainer.config.outputs.run_name_suffix}"
    labels_val_gt = sleap.load_file(f"{path_prefix}/labels_gt.val.slp")
    labels_val_pr = sleap.load_file(f"{path_prefix}/labels_pr.val.slp")
    val_metrics = sleap.nn.evals.evaluate(
        labels_val_gt, labels_val_pr, oks_scale=crop_size_suggest
    )
    # val_metrics = sleap.load_metrics(cfg.outputs.run_name, split="val")
    precision = val_metrics["vis.precision"]
    recall = val_metrics["vis.recall"]
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def main():
    study = optuna.create_study(direction="maximize")
    # The optimization finishes after evaluating 1000 times or 3 seconds.
    study.optimize(objective, n_trials=5)
    for trial in study.trials:
        print(trial)
    print(f"Best params is {study.best_params} with value {study.best_value}")


if __name__ == "__main__":
    main()
