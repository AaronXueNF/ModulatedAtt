from models.net_baseline import cheng2020_baseline_woGMM
from models.net_modified import (
    cheng2020_AdaptMAtt_woGMM, 
    cheng2020_ModulatedAtt_woGMM,
    cheng2020_AdaptScaleMAtt_woGMM
)

def find_net(name):
    if name == "cheng2020_baseline_woGMM":
        return cheng2020_baseline_woGMM
    elif name == "cheng2020_ModulatedAtt_woGMM":
        return cheng2020_ModulatedAtt_woGMM
    elif name == "cheng2020_AdaptMAtt_woGMM":
        return cheng2020_AdaptMAtt_woGMM
    elif name == "cheng2020_AdaptScaleMAtt_woGMM":
        return cheng2020_AdaptScaleMAtt_woGMM
    else:
        raise NotImplementedError


if __name__ == '__main__':
    net = cheng2020_baseline_woGMM()
    net.update()