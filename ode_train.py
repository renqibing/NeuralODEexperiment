import torch
import torch.optim as optim
import os

from sacred import Experiment,SETTINGS
from sacred.observers import FileStorageObserver
from visdom_observer.visdom_observer import VisdomObserver
# from pytorch_utils import sacred_utils as su
from pytorch_utils import sacred_trainer as st
from pytorch_utils.updaters import averager

from examples.model_ingredient import model_ingredient,make_model
from examples.data_ingredient import data_ingredient,make_dataloaders

from examples.utils import train_on_batch,create_scheduler_callback,create_val_scheduler_callback

rand_flag = False
ex = Experiment('odenet_cifar10_train',ingredients = [data_ingredient,model_ingredient])

if rand_flag:
    SAVE_DIR = './run/train/odenet_randtime_cifar10'
else:
    SAVE_DIR = './run/train/odenet_cifar10'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

ex.observers.append(FileStorageObserver.create(SAVE_DIR))
# ex.observers.append(VisdomObserver())


# optimizer
@ex.config
def optimizer_config():
    lr = 0.1
    opt = 'sgd'
    weight_decay = 0.0005

@ex.capture
def make_optimizer(model,opt,lr,weight_decay):
    options = {
        'adam':optim.Adam,
        'rmsp':optim.RMSprop,
        'sgd':optim.SGD
    }

    optimizer = options[opt](model.parameters(),lr = lr,momentum = 0.9,weight_decay = weight_decay)

    return optimizer

# scheduler
@ex.config
def scheduler_config():
    milestones = [135,185,240]
    gamma = 0.1

@ex.capture
def make_scheduler(optimizer,milestones,gamma):
    return create_scheduler_callback(optimizer,milestones,gamma)

# train
@ex.config
def train_congif():
    epochs = 400
    save_every = 10
    start_epoch = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    isgpu = True
    isrand = False
@ex.automain
def main(_run,device,isrand):
    dset,train,val,test = make_dataloaders()
    model = make_model()
    optimizer = make_optimizer(model)
    callback = make_scheduler(optimizer)

    st.loop(**{
        **_run.config,
        **dict(_run = _run,
               model = model,
               optimizer = optimizer,
               save_dir = SAVE_DIR,
               trainOnBatch = train_on_batch,
               callback = callback,
               train_loader = train,
               val_loader = val,
               callback_metric_names = {'val_loss','val_acc','lr'},
               batch_metric_names = {'loss','acc','fnfe','bnfe'},
               updaters = [averager]*4
               )
    })




