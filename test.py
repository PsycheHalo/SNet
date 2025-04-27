import torch
import math
import copy
from Atiny import Atiny
from AccessNet import AccessNet
from SNet import SNet

maxTimes=100000
BatchSize=100

size=64
layers=100
features=128

lr1=5e-4
lr2=5e-5
weight_decay=0

device = "cuda" if torch.cuda.is_available() else "cpu"

module1=AccessNet(size,size,layers,features,device=device)
module2=SNet(size,size,layers,features,device=device)
optimizer1 = Atiny(module1.parameters(),lr = lr1,weight_decay=weight_decay)
optimizer2 = Atiny(module2.parameters(),lr = lr2,weight_decay=weight_decay)

index=torch.arange(0,2*size,1,device=device,dtype=torch.float32).unsqueeze(0)

import visdom
wind = visdom.Visdom(env="Optimizer Test", use_incoming_socket=False)
    
wind.line([[float('nan'),float('nan')]],[0],win = 'loss',opts = dict(title = 'log(loss)/log(batchs)',legend = ['AccessNet','SNet']))
wind.line([[float('nan'),float('nan')]],[0],win = 'AccessNet',opts = dict(title = 'AccessNet: Comparison of curve prediction results',legend = ['output','target']))
wind.line([[float('nan'),float('nan')]],[0],win = 'SNet',opts = dict(title = 'SNet: Comparison of curve prediction results',legend = ['output','target']))

print(module2)
    
for time in range(maxTimes):
    a=torch.randn(BatchSize,device=device).unsqueeze(1)
    b=torch.randn(BatchSize,device=device).unsqueeze(1)
    c=torch.randn(BatchSize,device=device).unsqueeze(1)*size
    input,target=torch.chunk(((((index*a+c).sin()+1)/2)**b.exp())-(((1-((index*a+c).sin()))/2)**(-b).exp()),chunks=2,dim=1)
    
    module1.zero_grad()
    output=module1(input)
    loss1=torch.nn.functional.mse_loss(output,target)
    loss1.backward()
    optimizer1.step()
    
    L1=[output[0].tolist(),target[0].tolist()]
    L1=list(map(list, zip(*L1)))
    

    module2.zero_grad()
    output=module2(input)
    loss2=torch.nn.functional.mse_loss(output,target)
    loss2.backward()
    optimizer2.step()
    
    L2=[output[0].tolist(),target[0].tolist()]
    L2=list(map(list, zip(*L2)))
    

    wind.line(L1,win = 'AccessNet',opts = dict(title = 'AccessNet: Comparison of curve prediction results',legend = ['output','target']))
    wind.line(L2,win = 'SNet',opts = dict(title = 'SNet: Comparison of curve prediction results',legend = ['output','target']))
    wind.line([[float(loss1.log()),float(loss2.log())]],[math.log(time+1)],win = 'loss',update = 'append')

    
