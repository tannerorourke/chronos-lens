import argparse

import torch

from src.utils.io import get_model_config
from src.utils.seed import set_global_seed
from src.utils.constants import SAE_TARGETS


parser = argparse.ArgumentParser(description="""Training pipeline for all models""")
parser.add_argument('--exp', required=True, 
                    help="Name of experiment folder (subdir of '/experiments'). If model "
                         "exists, a new model artifacts directory will be populated. New "
                         "models should specify a unique 'expirements/[model_name]' folder "
                         "with config.yaml in it")

subparsers = parser.add_subparsers(dest="command", required=True)

model_parser = subparsers.add_parser(
    "model", 
    help="Train core model architecture, as defined in 'experiments/<exp>/config.yaml"
)
sae = subparsers.add_parser(
    "sae", 
    help="Train SAE on pre-trained model from --exp folder"
)
sae.add_argument(
    "--target", type=str, required=True, choices=SAE_TARGETS,
    help="Which vector to train on: z_enc (flattened encoder), z_pred, z_target, "
         "pred_error (z_pred - z_target). Must be defined in experiments/<exp>/config_sae.yaml.")
sae.add_argument(
    "--embeddings", type=str, default=None,
    help="Embeddings .npz filename within experiments/<exp>/embeddings/ (e.g. "
         "embedding_40.npz). If not provided, picks latest epoch.")


def main():
    print("Configuring..")
    args = parser.parse_args()
    
    exp_dir, params = get_model_config(args)
    
    set_global_seed(params["meta"]["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Disable tf32 matmul when using bfloat16 to avoid stacking two levels of reduced precision
        use_bf16 = params.get("meta", {}).get("use_bfloat16", False)
        if use_bf16:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision('high')
    
    print(f"  Device: {torch.cuda.get_device_name() if device.type == 'cuda' else device}")
    print(f"  Experiment: '{args.exp}'")
    
    
    if args.command == "sae":
        from src.training.train_sae import main as train_main
        train_main(params, exp_dir, args.target, args.embeddings, device)
    else:
        arch = params.get("model", {}).get("architecture", "stopgrad")
        if arch == "ema":
            from src.training.train_ema import main as train_main
        elif arch == "supervised":
            from src.training.train_supervised import main as train_main
        else:
            from src.training.train_sg import main as train_main
        train_main(params, exp_dir, device)
    
    

if __name__ == "__main__":
    main()
    
    