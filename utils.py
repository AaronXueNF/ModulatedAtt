import json
import math
import torch
import torch.optim as optim
from PIL import Image
from pytorch_msssim import ms_ssim

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, lr_init, aux_lr_init):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=lr_init,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=aux_lr_init,
    )
    return optimizer, aux_optimizer


def adjust_lr(optimizer, lr, verbose=True, epoch=None):
    old_lr = optimizer.state_dict()['param_groups'][0]['lr']
    for p in optimizer.param_groups:
        p['lr'] = lr
    new_lr = optimizer.state_dict()['param_groups'][0]['lr']
    if verbose:
        print("[Adjust lr] In epoch{}, lr: {} -> {}".format(
            epoch, old_lr, new_lr
        ))


def concat_images(image1, image2):
    """
    Concatenates two images together
    """
    result_image = Image.new('RGB', (image1.width + image2.width, image1.height))
    result_image.paste(image1, (0, 0))
    result_image.paste(image2, (image1.width, 0))
    return result_image


def compute_psnr(a, b):
    if a.ndim > 3:
        mse = torch.mean((a - b) ** 2, dim=(1,2,3), keepdim=False)
    else:
        mse = torch.mean((a - b) ** 2)
    return mse, -10 * torch.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[2] * size[3]
    y_sum = torch.sum(torch.log(out_net['likelihoods']['y']), dim=(1, 2, 3), keepdim=False)
    z_sum = torch.sum(torch.log(out_net['likelihoods']['z']), dim=(1, 2, 3), keepdim=False)
    bpp = (y_sum + z_sum) / (-math.log(2) * num_pixels)
    return bpp.detach()


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def parse_json_param(path):
    with open(path,"r") as f:
        param = json.load(f)
    return param


if __name__=='__main__':
    param = parse_json_param("hyperParam.json")
    print(param)
    print(param["device"])