from typing import Literal

import torch

from src.models.encoder import TransformerEncoder, EncounterEncoder
from src.models.jepa_ema import JEPA_EMA
from src.models.jepa_stopgrad import JEPAStopGrad
from src.models.supervised_transformer import SupervisedTransformer
from src.models.sae import SparseAutoencoder

MODEL_TYPE_STR = Literal["ema", "stopgrad", "supervised"]

MODEL_TYPE = JEPA_EMA | JEPAStopGrad | SupervisedTransformer

def build_model(
  model_params: dict,
  device: torch.device
) -> MODEL_TYPE:
    
    arch = model_params.get("architecture", "")
    if arch == "ema":
        return JEPA_EMA(**model_params).to(device) 
    elif arch == "stopgrad":
        return JEPAStopGrad(**model_params).to(device)
    elif arch == "supervised":
        return SupervisedTransformer(**model_params).to(device)
    raise ValueError(f"Unknown architecture: '{arch}'")
  
  
exports = {
    "TransformerEncoder": TransformerEncoder,
    "EncounterEncoder": EncounterEncoder,
    "JEPA_EMA": JEPA_EMA,
    "JEPAStopGrad": JEPAStopGrad,
    "SupervisedTransformer": SupervisedTransformer,
    "SparseAutoencoder": SparseAutoencoder,
    "build_model": build_model,
    "MODEL_TYPE": MODEL_TYPE,
    "MODEL_TYPE_STR": MODEL_TYPE_STR
}