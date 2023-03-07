import numpy as np
import torch
from torch import nn


class PPNetAdversarialWrapper(nn.Module):
    """
    Wrapper over the PPNet model that allows for adversarially attack activations of selected prototypes,
    over a selected image, and with a selected mask.
    The attack aims to minimize the activation of the selected prototypes, while modifying only the masked pixels.
    """
    def __init__(
            self,
            model: nn.Module,
            img: torch.Tensor,
            proto_nums: np.ndarray,
            mask: torch.Tensor,
            ):
        """
        :param model: PPNet model
        :param img: an image to attack
        :param proto_nums: vector of prototype numbers to attack
        :param mask: binary mask, 1 for pixels that can be modified, 0 for pixels that cannot be modified
        """
        super(PPNetAdversarialWrapper, self).__init__()
        self.model = model
        self.proto_nums = proto_nums
        self.mask = mask

        # ensure that we do not propagate gradients through the image and the mask
        self.img = img.clone()
        self.img.requires_grad = False
        # self.mask = torch.tensor(mask, device=self.img.device)
        self.mask.requires_grad = False

        self.initial_activation, self.final_activation = None, None

    def forward(self, x):
        # 'x' can be modified by cleverhans
        # 'x2' is the actual output image. We use masking to ensure that cleverhans can affect only the masked pixels.
        # x2 = x * self.mask + self.img * (1 - self.mask)
        x2 = x

        conv_output, distances = self.model.push_forward(x2)
        distances = distances[:, self.proto_nums]
        activations = self.model.distance_2_similarity(distances).flatten(start_dim=2)
        activations, _ = torch.max(activations, dim=-1)
        if self.initial_activation is None:
            self.initial_activation = activations[0].clone().cpu().detach().numpy()
        self.final_activation = activations[0].clone().cpu().detach().numpy()
        activations = torch.mean(activations).unsqueeze(0).unsqueeze(0)
        return activations
