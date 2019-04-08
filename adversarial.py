import torch
from torch import nn

# epsilon input gradient perturbation


def fgsm(model,input,label,eplison=0.3):

    if eplison == 0:
        return input.clone().detach()

    inp = input.detach().requires_grad_(True)
    output = model(inp)
    loss = nn.CrossEntropyLoss()(output,label)
    loss.backward()

    perturbation = inp.grad.sign()*eplison

    return (inp+perturbation).detach()
    # a leaf variable cannot be modified directly

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(784,10)
    def forward(self,x):
        return self.layer(x.view(x.shape[0],-1))

# mymodel = model().cuda()
# x = torch.randn(5,28,28).cuda()
# label = torch.LongTensor([5,4,3,2,1]).cuda()
# inp = x.clone().requires_grad_(True)
# inp = inp+1
# output = mymodel(inp)
# loss = nn.CrossEntropyLoss()(output,label)
# loss.backward()
# adv_inp = fgsm(mymodel,inp,label)
#
# with torch.no_grad():
#     print(mymodel(x)[range(len(x)),label])
#     print(mymodel(adv_inp)[range(len(adv_inp)),label])

def pgd(model,input,labels,
        step_size = 0.01,
        num_step = 40,
        eplison = 0.3,
        random_start = True,
        pixel_range = (-0.5,0.5)):

    new_inp = input.clone().detach()
    if eplison == 0:
        return new_inp
    if random_start:
        new_inp += torch.rand(*new_inp.shape,device = new_inp.device)*2*eplison - eplison
    for i in range(num_step):
        inp_var = new_inp.clone() # clone the original data and send into the network
        inp_var.requires_grad = True
        output = model(inp_var.cuda())
        loss = nn.CrossEntropyLoss().cuda()(output,labels.cuda())
        loss.backward()

        new_inp += inp_var.grad.sign()*step_size

    new_inp = torch.max(torch.min(new_inp,inp_var+eplison),inp_var-eplison)
    new_inp = torch.clamp(new_inp,*pixel_range)

    return new_inp.clone().detach()

# criterion device
class advloader():
    def __init__(self,dataloader,model,attack):
        super().__init__()
        self.dataloader = dataloader
        self.model = model
        self.attack = attack
    def __iter__(self): # yield defines a generator function ; every iteration it would run this function again from the last call
        for images,labels in self.dataloader:
            yield self.attack(self.model,images,labels),labels
    # def __getitem__(self, item):
    #     image,labels = self.dataloader[item]
    #     image = self.fgsm(self.model,image,labels,self.eplison,self.criterion)
    #     return image,labels
    def __len__(self):
        return len(self.dataloader)