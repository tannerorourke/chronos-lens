import argparse

import torch

from src.utils.io import get_model_config, init_run_dir
from src.utils.seed import set_global_seed, load_exp_seed
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
    "--target", type=str, required=True, default="z_enc", choices=SAE_TARGETS,
    help="Which vector to train on: z_enc (flattened encoder), z_pred, z_target, "
         "pred_error (z_pred - z_target). Must be present in the run's config['sae_config'].")
sae.add_argument(
    "--embeddings", type=str, default=None,
    help="Embeddings .npz filename within the run's embeddings/ dir (e.g. "
         "embedding_40.npz). If not provided, picks latest epoch.")


def main():
    print("Configuring..")
    args = parser.parse_args()
    command = args.command or "model"        # default action when no subcommand is given
    target = getattr(args, "target", None)   # only defined under the `sae` subcommand
    # config_path: in-repo input spec (git-tracked). run_dir: out-of-repo outputs.
    config_path, run_dir, params = get_model_config(args.exp, command, target)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device: {torch.cuda.get_device_name() if device.type == 'cuda' else device}")
    print(f"  Experiment: '{args.exp}'")
    print(f"  Run dir: {run_dir}")

    if command == "sae":
        # SAE operates inside an existing run dir; seed from its frozen config.
        set_global_seed(load_exp_seed(run_dir))
        from src.training.train_sae import main as train_main
        train_main(params, run_dir, args.target, args.embeddings, device)
    else:
        set_global_seed(params["meta"]["seed"])
        if device.type == "cuda":
            use_bf16 = params["meta"].get("use_bfloat16", False)
            torch.backends.cudnn.benchmark = True
            if use_bf16:
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cuda.matmul.allow_tf32 = False
            else:
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.set_float32_matmul_precision('high')

        # Scaffold the run dir: mkdir + freeze config.yaml + empty notes.md.
        init_run_dir(run_dir, params)

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