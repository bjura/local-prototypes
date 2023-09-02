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
            model_name: str,
            img: torch.Tensor,
            proto_nums: np.ndarray,
            mask: torch.Tensor,
            focal_sim: bool = False
    ):
        """
        :param model: PPNet model
        :param img: an image to attack
        :param proto_nums: vector of prototype numbers to attack
        :param mask: binary mask, 1 for pixels that can be modified, 0 for pixels that cannot be modified
        """
        super(PPNetAdversarialWrapper, self).__init__()
        self.model = model
        self.model_name = model_name
        self.proto_nums = proto_nums
        self.mask = mask
        self.focal_sim = focal_sim

        # ensure that we do not propagate gradients through the image and the mask
        self.img = img.clone()
        self.img.requires_grad = False
        # self.mask = torch.tensor(mask, device=self.img.device)
        self.mask.requires_grad = False

        self.initial_activation, self.final_activation = None, None

    def forward(self, x):
        # 'x' can be modified by cleverhans
        # 'x2' is the actual output image. We use masking to ensure that cleverhans can affect only the masked pixels.
        x2 = x * self.mask + self.img * (1 - self.mask)

        if self.model_name == 'prototree':
            conv_output, distances, _ = self.model.forward_partial(x2)
        else:
            conv_output, distances = self.model.push_forward(x2)
        distances = distances[:, self.proto_nums]
        if self.model_name == 'tesnet':
            activations = -distances
        elif self.model_name == 'prototree':
            activations = torch.exp(-distances)
        else:
            activations = self.model.distance_2_similarity(distances)
        activations = activations.flatten(start_dim=2)
        max_activations, _ = torch.max(activations, dim=-1)
        self.final_activation = max_activations[0].clone().cpu().detach().numpy()
        if self.initial_activation is None:
            self.initial_activation = max_activations[0].clone().cpu().detach().numpy()

        if self.focal_sim:
            distances = distances.flatten(start_dim=2)
            mean_dist = torch.mean(distances, dim=-1).unsqueeze(-1)
            min_dist, _ = torch.min(distances, dim=-1)

            sim_diff = self.model.distance_2_similarity(min_dist) - self.model.distance_2_similarity(mean_dist)
            return torch.mean(sim_diff).unsqueeze(0).unsqueeze(0)
        else:
            self.final_activation = max_activations[0].clone().cpu().detach().numpy()
            return torch.mean(max_activations).unsqueeze(0).unsqueeze(0)
