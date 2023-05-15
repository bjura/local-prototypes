import os

import torch
from torch import nn
from torch.nn.functional import gumbel_softmax
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from preprocess import mean, std

normalize = transforms.Normalize(mean=mean, std=std)


def run_model_on_batch(
        model: torch.nn.Module,
        model_name: str,
        batch: torch.Tensor,
        proto_pool: bool = False
):
    if model_name in ('ppnet', 'protopool', 'tesnet'):
        _, distances = model.push_forward(batch)
    elif model_name == 'prototree':
        _, distances, _ = model.forward_partial(batch)

    if proto_pool:
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])).squeeze()  # [b, p]
        avg_dist = F.avg_pool2d(distances, kernel_size=(distances.size()[2],
                                                        distances.size()[3])).squeeze()  # [b, p]

        gumbel_scale = 10e3
        with torch.no_grad():
            proto_presence = gumbel_softmax(model.proto_presence * gumbel_scale, dim=1, tau=0.5)

        # noinspection PyProtectedMember
        min_mixed_distances = model._mix_l2_convolution(min_distances, proto_presence)  # [b, c, n]
        # noinspection PyProtectedMember
        avg_mixed_distances = model._mix_l2_convolution(avg_dist, proto_presence)  # [b, c, n]
        x = model.distance_2_similarity(min_mixed_distances)  # [b, c, n]
        x_avg = model.distance_2_similarity(avg_mixed_distances)  # [b, c, n]
        x = x - x_avg
        if model.use_last_layer:
            prototype_activations = x.flatten(start_dim=1)
        else:
            raise NotImplementedError('Not implemented for proto_pool')
        patch_activations = model.distance_2_similarity(distances).cpu().detach().numpy()
        predictions = model.last_layer(prototype_activations)
    elif model_name == 'ppnet':
        min_distances = -nn.functional.max_pool2d(-distances,
                                                  kernel_size=(distances.size()[2],
                                                               distances.size()[3]))
        min_distances = min_distances.view(-1, model.num_prototypes)
        prototype_activations = model.distance_2_similarity(min_distances)
        patch_activations = model.distance_2_similarity(distances).cpu().detach().numpy()
        predictions = model.last_layer(prototype_activations)
    elif model_name == 'tesnet':
        patch_activations = (-distances).cpu().detach().numpy()
        prototype_activations = model.global_max_pooling(-distances)
        predictions = model.last_layer(prototype_activations)
    elif model_name == 'prototree':
        patch_activations = torch.exp(-distances).cpu().detach().numpy()
        predictions, _ = model.forward(batch)

    predicted_cls = torch.argmax(predictions, dim=-1)

    return predicted_cls.cpu().detach().numpy(), patch_activations


def run_model_on_dataset(
        model: nn.Module,
        model_name: str,
        dataset: Dataset,
        num_workers: int,
        batch_size: int,
        proto_pool: bool = False,
):
    """
    Runs the model on all images in the given directory and saves the results.
    :param model: the model to run
    :param dataset: pytorch dataset
    :param num_workers: number of parallel workers for the DataLoader
    :param batch_size: batch size for the DataLoader
    :param proto_pool: whether the model is ProtoPool
    :return a generator of model outputs for each of the images, together with batch data
    """
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    current_idx = 0

    for img_tensor, target in test_loader:
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        batch_samples = dataset.samples[current_idx:current_idx + batch_size]
        batch_filenames = [os.path.basename(s[0]) for s in batch_samples]
        with torch.no_grad():
            predicted_cls, patch_activations = run_model_on_batch(
                model=model, batch=img_tensor, proto_pool=proto_pool, model_name=model_name
            )
            current_idx += img_tensor.shape[0]

        img_numpy = img_tensor.clone().cpu().detach().numpy()
        for d in range(3):
            img_numpy[:, d] = (img_numpy[:, d] * std[d] + mean[d])

        yield {
            'filenames': batch_filenames,
            'target': target.cpu().detach().numpy(),
            'img_tensor': img_tensor,
            'img_original_numpy': img_numpy,
            'patch_activations': patch_activations,
            'predicted_cls': predicted_cls,
        }
