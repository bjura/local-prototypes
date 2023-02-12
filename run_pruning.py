import os
import shutil

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse

from helpers import makedir
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
import neptune.new as neptune

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('--masking_type', type=str, default='none')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

optimize_last_layer = True

# pruning parameters
k = 6
prune_threshold = 3

original_model_dir = args.modeldir[0]  # './saved_models/densenet161/003/'
original_model_name = args.model[0]  # '10_16push0.8007.pth'
original_experiment_name = os.path.basename(os.path.normpath(original_model_dir))

need_push = ('nopush' in original_model_name)
if need_push:
    assert (False)  # pruning must happen after push

model_dir = os.path.join(original_model_dir, 'pruned_prototypes')

makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

ppnet = torch.load(os.path.join(original_model_dir, original_model_name))
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# load the data
from settings import train_dir, test_dir, train_push_dir, NEPTUNE_API_TOKEN, num_workers, coefs

train_batch_size = 80
test_batch_size = 100
img_size = 224
train_push_batch_size = 80

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=False)

# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# push set: needed for pruning because it is unnormalized
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

log('push set size: {0}'.format(len(train_push_loader.dataset)))

tnt.test(model=ppnet_multi, dataloader=test_loader,
         class_specific=class_specific, log=log)

# prune prototypes
log('prune')
prune.prune_prototypes(dataloader=train_push_loader,
                       prototype_network_parallel=ppnet_multi,
                       k=k,
                       prune_threshold=prune_threshold,
                       preprocess_input_function=preprocess_input_function,  # normalize
                       original_model_dir=original_model_dir,
                       epoch_number=0,
                       # model_name=None,
                       log=log,
                       copy_prototype_imgs=True)
accu, _, metrics = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                            model_name='prune',
                            accu=accu,
                            target_accu=0.10, log=log)

# last layer optimization
if optimize_last_layer:
    if isinstance(NEPTUNE_API_TOKEN, str) and len(NEPTUNE_API_TOKEN) > 0:
        log('initializing neptune')
        neptune_run = neptune.init_run(
            project='mikolajsacha/protobased-research',
            name=f'{original_experiment_name}_pruning',
            api_token=NEPTUNE_API_TOKEN,
            tags=['local_prototypes', 'pruning']
        )
    else:
        neptune_run = None
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    log('optimize last layer')
    tnt.last_only(model=ppnet_multi, log=log)
    accu = 0.0
    for i in range(100):
        # log('iteration: \t{0}'.format(i))
        train_accu, _, metrics = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                           class_specific=class_specific, coefs=coefs, log=log,
                                           masking_type=args.masking_type, neptune_run=neptune_run)
        if neptune_run is not None:
            neptune_run["train/epoch/accuracy"].append(train_accu)
            neptune_run["train/epoch/stage"].append(3.0)

        accu, _, metrics = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log, masking_type=args.masking_type,
                                    neptune_run=neptune_run)
        if neptune_run is not None:
            neptune_run["test/epoch/accuracy"].append(accu)

        # if accu > best_accu:
            # save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                        # model_name='prune_best',
                                        # accu=accu,
                                        # target_accu=0.10, log=log)
            # best_accu = accu

    save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                model_name='prune_last',
                                accu=accu,
                                target_accu=0.10, log=log)
    if neptune_run is not None:
        neptune_run.stop()

logclose()
