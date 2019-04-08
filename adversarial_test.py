from functools import partial

import torch
from torch import nn
from sacred import Experiment
from sacred.observers import FileStorageObserver
from pytorch_utils.sacred_utils import get_model_path, read_config, import_source
import logging

from examples.utils import validate
import examples.adversarial as adv

ATTACK = {
    'pgd':adv.pgd,
    'fgsm':adv.fgsm
}

ex = Experiment('advOdeTest')
SAVE_DIR ='runs/advOdeTest'
ex.observers.append(FileStorageObserver(SAVE_DIR))

class truncIterator:
    def __init__(self,iterator,end):
        super().__init__()
        self.end = end
        self.iterator = iterator
    def __iter__(self):
        if self.end != -1:
            epochs = self.end
        else:
            epochs = len(self.iterator)
        ite = iter(self.iterator)
        for i in range(epochs):
            yield next(ite)
    def __len__(self):
        return self.end if self.end != -1 else len(self.iterator)


@ex.config
def make_main():
    isgpu = 'True'
    run_dir = 'runs/advOdeTrain'
    epoch = 'latest'
    batches = -1
    # odefunc config
    atol = 1e-4
    rtol = 1e-4
    # optimizer config
    att = 'pgd'
    epsilon = 0.3
    step_size = 0.01
    num_step = 40
    random_start = True

@ex.automain
def main(run_dir,
         epoch,
         isgpu,
         batches,
         atol,
         rtol,
         att,
         epsilon,
         step_size,
         num_step,
         random_start,
         _log): # construct model and load parameters
    config = read_config(run_dir)
    _log.info('Read config from {run_dir}'.format(run_dir))

    mod_ing = import_source(run_dir,'model_ingredient')
    model = mod_ing.make_model(**{**config['model'],'isgpu':isgpu},_log = _log)
    path = get_model_path(run_dir,epoch)

    if isinstance(model,nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))
    model = model.eval()
    _log.info('Load model from {path}'.format(path))

    data_ing = import_source(run_dir,'data_ingredient')
    dset,train,val,test = data_ing.make_dataloaders(**{**config['dataset'],'isgpu':isgpu},_log = _log)

    if att == 'pgd':
        attack = partial(ATTACK[att],
                         epsilon = epsilon,
                         step_size = step_size,
                         num_step = num_step,
                         random_start = random_start
                         )
    else:
        attack = partial(ATTACK[att],
                         epsilon = epsilon)

    if hasattr(model,'ODEnet'):
        model.ODEnet.odeblock.atol = atol
        model.ODEnet.odeblock.rtol = rtol

    advTestLoader = adv.advloader(test,model,attack)
    advTestLoader = truncIterator(advTestLoader,batches)

    _log.info('Testing Model...')
    test_loss,test_acc = validate(model,advTestLoader,_log=logging.getLogger('validate'))

    _log.info('Test loss : {loss}, accuracy : {}'.format(test_loss,test_acc))

    return test_loss,test_acc


