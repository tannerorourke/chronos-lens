import argparse

import torch

from src.utils.io import init_exp_config
from src.utils.system import set_global_seed, load_exp_seed, set_cuda_precision
from src.utils.constants import SAE_TARGETS


parser = argparse.ArgumentParser(description="""Training pipeline for all models""")
parser.add_argument('--exp', required=True,
                    help="Run-id naming the input config 'experiments/<exp>.yaml'. If a run "
                         "with this id already has artifacts, a new versioned run dir is "
                         "populated. New models should add a unique 'experiments/<run-id>.yaml'.")

# `model` is the default action: `--exp <run-id>` with no subcommand trains the core model.
subparsers = parser.add_subparsers(dest="command")

model_parser = subparsers.add_parser(
    "model",
    help="Train core model architecture (default), as defined in 'experiments/<exp>.yaml'"
)
sae = subparsers.add_parser(
    "sae", 
    help="Train SAE on pre-trained model from --exp folder"
)
sae.add_argument(
    "--target", type=str, required=True, choices=SAE_TARGETS,
    help="Which vector to train on: z_enc (flattened encoder), z_pred, z_target, "
         "pred_error (z_pred - z_target). Must be present in the run's config['sae_config'].")
sae.add_argument(
    "--embeddings", type=str, default=None,
    help="Embeddings .npz filename within the run's embeddings/ dir (e.g. "
         "embedding_40.npz). If not provided, picks latest epoch.")


def main():
    print("Configuring..")
    args = parser.parse_args()
    command = args.command or "model"
    target = getattr(args, "target", None)
    
    run_dir, params = init_exp_config(args.exp, command, target)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {torch.cuda.get_device_name() if device.type == 'cuda' else device}")

    if command == "sae":
        # uses existing model's directory/seed
        set_global_seed(load_exp_seed(run_dir))
        
        print(f"  Experiment: '{run_dir.parent.name}' -> '{run_dir.name}'")
        print(f"  Artifact dir: {run_dir}")
        
        from src.training.train_sae import main as train_main
        train_main(params, run_dir, args.target, args.embeddings, device)
    else:
        set_global_seed(params["meta"]["seed"])
        
        if device.type == "cuda":
            set_cuda_precision(use_bf16=True)
        
        print(f"  Experiment: '{run_dir.name}'")
        print(f"  Artifact dir: {run_dir}")

        arch = params["model"].get("architecture", "")
        if arch == "ema":
            from src.training.train_ema import main as train_main
        elif arch == "stopgrad":
            from src.training.train_sg import main as train_main
        elif arch == "supervised":
            from src.training.train_supervised import main as train_main
        else:
            return
            # Add probe training

        train_main(params, run_dir, device)

    print("Done.")
    

if __name__ == "__main__":
    main()