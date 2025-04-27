from SLU import SLU
from AccessBlock import AccessBlock as Block

class AccessNet(torch.nn.Module):
    def __init__(self,in_features,out_features,hidden_layers,hidden_features,**kwargs):
        super(AccessNet, self).__init__()
        if hidden_features is None:
            hidden_features=max(in_features,out_features)
        columnGain=1/math.sqrt(1+hidden_layers)
        self.inLayer=Block(in_features,hidden_features,columnGain=columnGain,**kwargs)
        self.Blocks=torch.nn.Sequential() 
        for t in range(hidden_layers):
            self.Blocks.add_module('hiddenLayer{0}'.format(t),Block(hidden_features,hidden_features,columnGain=columnGain,**kwargs))
        self.outLayer=SLU(hidden_features,out_features,bias=False,keepNorm=True,**kwargs)
    def forward(self,input):
        step,columnStep=self.inLayer(input)
        column=columnStep.clone()
        for block in self.Blocks:
            step,columnStep=block(step)
            column=column+columnStep
        
        output=self.outLayer(column)
        return output
