import torch
from torch import nn
from sacred import Experiment
from pytorch_utils.sacred_utils import read_config, get_model_path, import_source
from examples.utils import validate

ex = Experiment('odenet_cifar10_test')


@ex.config
def input_config():
    rtol = 1e-5
    atol = 1e-5
    run_dir = ''
    epoch = 'latest'
@ex.capture
def main(run_dir,
         epoch,
         rtol,
         atol,
         _log):
    config = read_config(run_dir)
    _log.info('Load config from {}'.format(run_dir))

    model_ing = import_source(run_dir,'model_ingredient')
    model = model_ing.make_model(**{**config['model'],'isgpu':True},_log= _log)
    path = get_model_path(run_dir,epoch)
    if isinstance(model,nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))
    _log.info('Load paras from {}'.format(path))

    model = model.eval()

    if hasattr(model,'ODEnet'):
        model.ODEnet.odeblock.rtol = rtol
        model.ODEnet.odeblock.atol = atol

    data_loader_ing = import_source(run_dir,'data_ingredient')
    dset,train,val,test = data_loader_ing.make_dataloaders(**{**config['dataset'],'isgpu':True},_log = _log)

    _log.info('Testing models...')
    loss,acc = validate(model,test)
    _log.info("Test loss = {test_loss:.6f}, Test accuracy = {test_acc:.4f}".format(test_loss = loss,test_acc = acc))

