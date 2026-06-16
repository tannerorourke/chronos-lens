import argparse

import torch

from src.utils.io import init_exp_config
from src.utils.system import set_global_seed, load_exp_seed, set_cuda_precision


parser = argparse.ArgumentParser(description="""Training pipeline for core models""")
parser.add_argument('--exp', required=True,
                    help="Run-id naming the input config 'experiments/<exp>.yaml'. If a run "
                         "with this id already has artifacts, a new versioned run dir is "
                         "populated. New models should add a unique 'experiments/<run-id>.yaml'.")


def main():
    args = parser.parse_args()
    run_dir, params = init_exp_config(args.exp, "model")
    set_global_seed(params["meta"]["seed"])
    
    print(f"  Experiment: '{run_dir.name}'")
    print(f"  Artifact dir: {run_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        set_cuda_precision(use_bf16=True)
    print(f"  Device: {torch.cuda.get_device_name() if device.type == 'cuda' else device}")
    
    arch = params["model"].get("architecture", "")
    if arch == "ema":
        from src.training.train_ema import main as train
    elif arch == "stopgrad":
        from src.training.train_sg import main as train
    elif arch == "supervised":
        from src.training.train_supervised import main as train
    else:
        return
        # Add probe training

    train(params, run_dir, device)


if __name__ == "__main__":
    main()