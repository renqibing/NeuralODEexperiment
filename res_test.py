import torch
from torch import nn
from sacred import Experiment
from pytorch_utils.sacred_utils import read_config, get_model_path, import_source
from examples.utils import validate

ex = Experiment('odenet_cifar10_test')

@ex.config
def input_config():
    run_dir = './run/train/resnet_cifar10/50'
    epoch = 'latest'
@ex.automain
def main(run_dir,
         epoch,
         _log):
    config = read_config(run_dir)
    _log.info('Load config from {}'.format(run_dir))

    model_ing = import_source(run_dir,'model_ingredient')
    print(config['model'])
    # model = model_ing.make_resnet(**{**config['model'],'isgpu':True},_log= _log)
    model = model_ing.make_resnet(config['model']['layers'], True)
    path = get_model_path(run_dir,epoch)
    path = path.replace('\\','/')
    if isinstance(model,nn.DataParallel):
        model.module.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path))
    _log.info('Load paras from {}'.format(path))

    model = model.eval()

    data_loader_ing = import_source(run_dir,'data_ingredient')
    dset,train,val,test = data_loader_ing.make_dataloaders(**{**config['dataset'],'isgpu':True},_log = _log)

    _log.info('Testing resnet models...')
    loss,acc = validate(model,test)
    _log.info("Resnet Test loss = {test_loss:.6f}, Test accuracy = {test_acc:.4f}".format(test_loss = loss,test_acc = acc))

