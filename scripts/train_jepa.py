import argparse
import yaml
import torch
from src.utils.io import init_run_dir
from src.training.train import main as app_main
  

parser = argparse.ArgumentParser(
    description="Minimal JEPA training pipeline for patient sequences")
parser.add_argument(
    "--model", type=str, required=True, 
    help="""Name of model to run (subdir of 'experiments'). If model exists, 
            a new model artifacts directory will be populated. To create a new
            model, specify a new folder name 'expirements/[model_name]' with 
            config.yaml in it, and then specify --model=[model_name] to run""")


def main():
    args = parser.parse_args()
    
    run_dir = init_run_dir(args.model)
    params: dict = {}
    with open(run_dir / "config.yaml", 'r') as y_file:
        params = yaml.safe_load(y_file)
    
    if not params:
        raise SystemExit(f"[run_train] Exiting. No parameters provided from 'experiments/{args.model}/config.yaml'.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Enable TF32 only for float32 paths (i.e., when not using bfloat16).
        # Avoids stacking two levels of reduced precision
        use_bf16 = params.get('use_bfloat16', False)
        torch.backends.cuda.matmul.allow_tf32 = not use_bf16
        torch.backends.cudnn.allow_tf32 = not use_bf16
        if not use_bf16:
            torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        
    print(f"\nDevice: {device}")
    
    app_main(params, run_dir, device)
    

if __name__ == "__main__":
    main()
    
    