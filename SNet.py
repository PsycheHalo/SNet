import torch
import math
from SLU import SLU

class SNet(torch.nn.Module):
    def __init__(self,in_features,out_features,hidden_layers,hidden_features,**kwargs):
        super(SNet, self).__init__()
        if hidden_features is None:
            hidden_features=max(in_features,out_features)
            
        self.Blocks=torch.nn.Sequential() 
        self.Blocks.add_module('inLayer',SLU(in_features,hidden_features,bias=False,keepNorm=True,**kwargs))
        for t in range(hidden_layers-2):
            self.Blocks.add_module('hiddenLayer{0}'.format(t),SLU(hidden_features,hidden_features,keepNorm=True,**kwargs))
        self.Blocks.add_module('outLayer',SLU(hidden_features,out_features,bias=False,**kwargs))
        with torch.no_grad():
            for t in range(hidden_layers//2):
                self.Blocks[t].LinearN.weight.data=self.Blocks[t].LinearP.weight.clone()
                self.Blocks[hidden_layers-t-1].LinearP.weight.data=torch.linalg.pinv(self.Blocks[t].LinearP.weight)
                self.Blocks[hidden_layers-t-1].LinearN.weight.data=self.Blocks[hidden_layers-t-1].LinearP.weight.clone()
    def forward(self,input):
        return self.Blocks(input)
