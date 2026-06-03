import torch


def set_cuda_precision(use_bf16: bool = False) -> None:
    """ Disable tf32 matmul when using bfloat16 to avoid stacking
        two levels of reduced precision
    """
    torch.backends.cudnn.benchmark = True

    if use_bf16:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('high')
