import math
import numpy as np

import torch
import torch.nn as nn

from compressai.zoo import cheng2020_attn
from models.net import (
    cheng2020_baseline_woGMM,
    cheng2020_ModulatedAtt_woGMM
)

save_pretrain = False
temp_state_path = "./checkpoints/pretrain/cheng2020att_pretrain.ckpt"
baseline_path = "./checkpoints/pretrain/baseline_pretrain.ckpt"
ModulatedAtt_path = "./checkpoints/pretrain/modulatedAtt_pretrain.ckpt"

def main():
    cheng2020att_pretrain = cheng2020_attn(quality = 5, metric='mse', pretrained=save_pretrain)
    if save_pretrain:
        temp_state = {
            "cheng2020att_pretrain":cheng2020att_pretrain.state_dict()
        }
        torch.save(temp_state, temp_state_path)
    else:
        checkpoint = torch.load(temp_state_path)
        cheng2020att_pretrain.load_state_dict(checkpoint["cheng2020att_pretrain"])
    
    pretrain_state = cheng2020att_pretrain.state_dict()

    Baseline = cheng2020_baseline_woGMM()
    Baseline.load_state_dict_pretrain(pretrain_state, strict=False)
    torch.save(
        {"state_dict":Baseline.state_dict()},
        baseline_path
    )

    ModulatedAtt = cheng2020_ModulatedAtt_woGMM()
    ModulatedAtt.load_state_dict_pretrain(pretrain_state, strict=False)
    torch.save(
        {"state_dict":ModulatedAtt.state_dict()},
        ModulatedAtt_path
    )


if __name__ == "__main__":
    main()    

    

