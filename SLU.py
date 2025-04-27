import torch
import math

class SLU(torch.nn.Module):
    def __init__(self,in_features,out_features,bias=True,gain=1,keepNorm=False,**kwargs):
        super(SLU, self).__init__()
        self.LinearP=torch.nn.Linear(in_features,out_features,bias=False,**kwargs)
        self.LinearN=torch.nn.Linear(in_features,out_features,bias=False,**kwargs)
        if bias:
            self.bias=torch.nn.Parameter(torch.zeros((1,out_features),**kwargs),requires_grad=True)
        else:
            self.bias=None
        self.keepNorm=keepNorm
        with torch.no_grad():
            torch.nn.init.orthogonal_(self.LinearP.weight,gain=gain)
            torch.nn.init.orthogonal_(self.LinearN.weight,gain=gain)
        
    def forward(self,input):
        inputP=input.clamp(min=0)
        inputN=input.clamp(max=0)
        if self.keepNorm:
            output=self.LinearP(inputP)/self.LinearP.weight.norm(dim=1,keepdim=True).T+self.LinearN(inputN)/self.LinearN.weight.norm(dim=1,keepdim=True).T
        else:
            output=self.LinearP(inputP)+self.LinearN(inputN)
            
        if self.bias is not None:
            output=output+self.bias
        
        return output
      
