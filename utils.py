import torch
from torch import nn
from torch import optim
from tqdm import tqdm
import numpy as np
import logging

import pytorch_utils.sacred_trainer as st


def train_on_batch(model,batch,optimizer,device = torch.device('cuda:0'),isgpu = True):
    if isinstance(model,nn.DataParallel):
        ode_model = model.module
    else:
        ode_model = model

    images,labels = batch

    if isgpu:
        images = images.cuda()
        labels = labels.cuda()

    criterion = nn.CrossEntropyLoss()

    # forward and inference
    nfe_forward = ode_model.odefunc.nfe.item()
    predictions = ode_model(images)
    loss = criterion(predictions,labels)

    # backward and optimizer
    ode_model.odefunc.nfe.fill_(0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    nfe_backward = ode_model.odefunc.nfe.item()

    loss = loss.cpu().detach().numpy()
    accuracy = st.accuracy(predictions.cpu(),labels.cpu())

    return loss,accuracy,nfe_forward,nfe_backward


def train_on_batch_resnet(model,batch,optimizer,isgpu = True):

    if isinstance(model,nn.DataParallel):
        resnet = model.module
    else:
        resnet = model

    images,labels = batch

    if isgpu:
        images,labels = images.cuda(),labels.cuda()

    # forward
    predictions = resnet(images)
    if isgpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    # print('predictions shape {}'.format(predictions.cpu().detach().numpy().shape))
    # print('labels shape {}'.format(labels.shape))
    loss = criterion(predictions,labels)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss = loss.cpu().detach().numpy()
    acc = st.accuracy(predictions.cuda(),labels.cuda())

    return loss,acc

# one hot np.array(labels[:,None] == np.arange(K)[,:], dtype = int)


def validate(model,val_loader,_log = logging.getLogger('validate')):
    total = 0
    total_accuracy = 0.0
    total_loss = 0.0
    model = model.eval()
    _log.info('begin to validate with %d steps'%len(val_loader))
    for images,labels in tqdm(val_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()
            criterion = nn.CrossEntropyLoss()
            batch_size = images.shape[0]
            predictions = model(images)

            loss = criterion(predictions, labels)
            total_loss += loss.cpu().detach().numpy() * batch_size

            accuracy = st.accuracy(predictions.cpu(),labels.cpu())
            total_accuracy += accuracy*batch_size

            total += batch_size

    total_accuracy /= total
    total_loss /= total
    model = model.train()
    return total_loss, total_accuracy


def learning_rate_with_decay(batch_size,initial_learning_rate,batch_denom,milestones,decay_rates):
    initial_learning_rate = initial_learning_rate*batch_size/batch_denom
    updated_learning_rates = [initial_learning_rate*weight for weight in decay_rates]

    def learning_rate_function(step):
        mask = [ step < s for s in milestones] + [True]
        i = np.argmax(mask)
        return updated_learning_rates[i]

    return learning_rate_function


def scheduler_generator(optimizer,milestones,gamma):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones,gamma)

    while True:
        scheduler.step()
        yield (optimizer.param_groups[0]['lr'],)


def create_scheduler_callback(optimizer,milestones,gamma):

    g = scheduler_generator(optimizer,milestones,gamma)
    def scheduler_callback(model,val_loader,batch_metrics_dict):
        return next(g)
    return scheduler_callback


def create_val_scheduler_callback(optimizer,milestones,gamma):
    g = scheduler_generator(optimizer,milestones,gamma)

    def schduler_callback(model,val_loader,batch_metrics_dict):
        lr = next(g)[0]
        val_loss,val_acc = validate(model,val_loader)
        return val_loss,val_acc,lr

    return schduler_callback
