import numpy as np
from tqdm import tqdm
import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from prototree.prototree import ProtoTree

from util.log import Log

def train_epoch(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                args,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:

    tree = tree.to(device)
    # Make sure the model is in eval mode
    tree.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = float(len(train_loader))
    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+' %s'%epoch,
                    ncols=0)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        # Make sure the model is in train mode
        tree.train()
        # Reset the gradients
        optimizer.zero_grad()

        xs, ys = xs.to(device), ys.to(device)

        if args.augmentations:
            img_size = 224
            min_box_size = img_size // 8
            max_box_size = img_size // 2
            masking_prob = 0.5
            max_num_boxes = 5

            with torch.no_grad():
                # TODO move this to the Dataset
                for sample_i in range(xs.shape[0]):
                    if np.random.random() < masking_prob:
                        continue

                    possible_modifications = [
                        torch.zeros_like(xs[sample_i]),
                        torch.rand(xs.shape[1:]),
                        xs[sample_i] + torch.rand(xs[sample_i].shape, device=xs.device)
                    ]

                    num_boxes = np.random.randint(1, max_num_boxes + 1)

                    for _ in range(num_boxes):
                        width = np.random.randint(min_box_size, max_box_size)
                        height = np.random.randint(min_box_size, max_box_size)
                        left = np.random.randint(0, img_size - width)
                        top = np.random.randint(0, img_size - height)

                        xs[sample_i, top:top + height, left:left + width] = \
                            possible_modifications[np.random.randint(3)][top:top + height,
                            left:left + width]

        # Perform a forward pass through the network
        ys_pred, info, distances = tree.forward(xs)

        # Learn prototypes and network with gradient descent.
        # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)

        if args.high_act_loss:
            all_similarities = torch.exp(-distances)
            with torch.no_grad():
                proto_sim = []
                proto_nums = []
                for sample_sim in all_similarities:
                    proto_max_act, _ = torch.max(sample_sim.reshape(sample_sim.shape[0], -1), axis=-1)
                    proto_num = torch.argmax(proto_max_act)
                    proto_nums.append(proto_num)
                    proto_sim.append(sample_sim[proto_num])
                proto_sim = torch.stack(proto_sim, dim=0).unsqueeze(1)

            if args.quantized_mask:
                all_sim_scaled = torch.nn.functional.interpolate(proto_sim,
                                                                 size=(xs.shape[-1], xs.shape[-1]),
                                                                 mode='bilinear')
                q = np.random.uniform(0.5, 0.98)
                quantile_mask = torch.quantile(all_sim_scaled.flatten(start_dim=-2), q=q, dim=-1)
                quantile_mask = quantile_mask.unsqueeze(-1).unsqueeze(-1)

                high_act_mask_img = (all_sim_scaled > quantile_mask).float()
                high_act_mask_act = torch.nn.functional.interpolate(high_act_mask_img,
                                                                    size=(all_similarities.shape[-1],
                                                                          all_similarities.shape[-1]),
                                                                    mode='bilinear')
            else:
                proto_sim_min = proto_sim.flatten(start_dim=1).min(-1)[0] \
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                proto_sim_norm = proto_sim - proto_sim_min
                proto_sim_max = proto_sim_norm.flatten(start_dim=1).max(-1)[0] \
                    .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                proto_sim_norm /= proto_sim_max
                high_act_mask_act = proto_sim_norm
                high_act_mask_img = torch.nn.functional.interpolate(high_act_mask_act,
                                                                    size=(xs.shape[-1], xs.shape[-1]),
                                                                    mode='bilinear')
            new_data = xs * high_act_mask_img
            _, distances2, _ = tree.forward_partial(new_data.detach())
            all_similarities2 = torch.exp(-distances2)

            proto_sim2 = []
            for sample_i, sample_label in enumerate(ys):
                proto_sim2.append(all_similarities2[sample_i, proto_nums[sample_i]])
            proto_sim2 = torch.stack(proto_sim2, dim=0).unsqueeze(1)

            if args.sim_diff_function == 'l2':
                sim_diff = (proto_sim - proto_sim2) ** 2
            elif args.sim_diff_function == 'l1':
                sim_diff = torch.abs(proto_sim - proto_sim2)
            else:
                raise ValueError(f'Unknown sim_diff_function: ', args.sim_diff_function)

            if args.quantized_mask:
                sim_diff_loss = torch.sum(sim_diff * high_act_mask_act) / torch.sum(high_act_mask_act)
            else:
                sim_diff_loss = torch.mean(sim_diff)

            loss += args.sim_diff_weight * sim_diff_loss
        else:
            sim_diff_loss = None

        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        if not disable_derivative_free_leaf_optim:
            #Update leaves with derivate-free algorithm
            #Make sure the tree is in eval mode
            tree.eval()
            with torch.no_grad():
                target = eye[ys] #shape (batchsize, num_classes) 
                for leaf in tree.leaves:  
                    if tree._log_probabilities:
                        # log version
                        update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                    else:
                        update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)  
                    leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                    F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                    leaf._dist_params += update

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
        )
        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info 


def train_epoch_kontschieder(tree: ProtoTree,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:

    tree = tree.to(device)

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    # Create a log if required
    log_loss = f'{log_prefix}_losses'
    if log is not None and epoch==1:
        log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')

    # Reset the gradients
    optimizer.zero_grad()

    if disable_derivative_free_leaf_optim:
        print("WARNING: kontschieder arguments will be ignored when training leaves with gradient descent")
    else:
        if tree._kontschieder_normalization:
            # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
            for _ in range(10):
                # Train leaves with derivative-free algorithm using normalization factor
                train_leaves_epoch(tree, train_loader, epoch, device)
        else:
            # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
            train_leaves_epoch(tree, train_loader, epoch, device)
    # Train prototypes and network. 
    # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)
    # Make sure the model is in train mode
    tree.train()
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Perform a forward pass through the network
        ys_pred, _, _ = tree.forward(xs)
        # Compute the loss
        if tree._log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)
        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
        )
        # Compute metrics over this batch
        total_loss+=loss.item()
        total_acc+=acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info['loss'] = total_loss/float(i+1)
    train_info['train_accuracy'] = total_acc/float(i+1)
    return train_info 

# Updates leaves with derivative-free algorithm
def train_leaves_epoch(tree: ProtoTree,
                        train_loader: DataLoader,
                        epoch: int,
                        device,
                        progress_prefix: str = 'Train Leafs Epoch'
                        ) -> dict:

    #Make sure the tree is in eval mode for updating leafs
    tree.eval()

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree._num_classes).to(device)

        # Show progress on progress bar
        train_iter = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)


        # Iterate through the data set
        update_sum = dict()

        # Create empty tensor for each leaf that will be filled with new values
        for leaf in tree.leaves:
            update_sum[leaf] = torch.zeros_like(leaf._dist_params)

        for i, (xs, ys) in train_iter:
            xs, ys = xs.to(device), ys.to(device)
            #Train leafs without gradient descent
            out, info, _ = tree.forward(xs)
            target = eye[ys] #shape (batchsize, num_classes) 
            for leaf in tree.leaves:  
                if tree._log_probabilities:
                    # log version
                    update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - out, dim=0))
                else:
                    update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/out, dim=0)
                update_sum[leaf] += update

        for leaf in tree.leaves:
            leaf._dist_params -= leaf._dist_params #set current dist params to zero
            leaf._dist_params += update_sum[leaf] #give dist params new value
