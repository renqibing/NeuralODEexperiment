from functools import partial

import torch
from torch import optim

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from visdom_observer.visdom_observer import VisdomObserver
import pytorch_utils.sacred_trainer as st
from pytorch_utils.updaters import averager

from examples.model_ingredient import model_ingredient, make_model
from examples.data_ingredient import data_ingredient, make_dataloaders

from examples.utils  import train_on_batch, create_val_scheduler_callback

from examples.adversarial import fgsm,advloader,pgd

ex = Experiment('advOdeExperiment',ingredients=[model_ingredient,data_ingredient])
SAVE_DIR = 'run/advOdeTrain'
ex.observers.append(FileStorageObserver.create(SAVE_DIR))

ATTACK = {
    'pgd': pgd,
    'fgsm':fgsm
}

class CombineLoader:
    def __init__(self,*loaders):
        super().__init__()
        self.loaders = list(loaders)[0]
    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders]
        while True:
            items = [next(iter) for iter in iters]
            images = torch.cat([i[0] for i in items])
            labels = torch.cat([i[1] for i in items])
            yield images,labels

    def __len__(self):
        return min(len(loader) for loader in self.loaders)

@ex.config
def optimizer_config():

    lr = 0.1
    weight_decay = 0.0005
    opt = 'sgd'

@ex.capture
def make_optimizer(model,opt,lr,weight_decay):
    options = {
        'adam':optim.Adam,
        'sgd':optim.SGD,
        'rmsprop':optim.RMSprop
    }
    optimizer = options[opt](model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer


@ex.config
def scheduler_config():
    milestones = [135,185,240]
    gamma = 0.1

@ex.capture
def make_scheduler(optimizer,milestones,gamma):
    return create_val_scheduler_callback(optimizer,milestones,gamma)

@ex.config
def adv_config():
    attack = 'pgd'
    step_size = 0.01
    num_step = 40
    random_start = True
    eplison = 0.3

@ex.automain
def main(_run,attack,step_size,num_step,random_start,eplison):
    dset,train,val,test = make_dataloaders()
    model = make_model()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)

    if attack == 'pgd':
        adv = partial(ATTACK[attack],step_size = step_size,
                             num_step = num_step,
                             eplison = eplison,
                             random_start = random_start)
    else:
        adv = partial(ATTACK[attack],eplison = eplison)

    adv_train = advloader(train,model,adv)
    total_trainloader = CombineLoader([train,adv_train])

    st.loop(
        **{**_run.config,
           **dict(_run=_run,
                  model=model,
                  optimizer=optimizer,
                  save_dir=SAVE_DIR,
                  trainOnBatch=train_on_batch,
                  train_loader=total_trainloader,
                  val_loader=val,
                  callback=scheduler,
                  callback_metric_names=['val_loss', 'val_acc', 'learning_rate'],
                  batch_metric_names=['loss', 'acc', 'nfef', 'nfeb'],
                  updaters=[averager]*4)})


