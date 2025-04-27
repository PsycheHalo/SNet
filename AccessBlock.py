from SLU import SLU

class AccessBlock(torch.nn.Module):
    def __init__(self,in_features,out_features,columnGain=1.0,**kwargs):
        super(AccessBlock, self).__init__()
        self.MainSLU=SLU(in_features,out_features,bias=True,keepNorm=True,**kwargs)
        self.ColumnSLU=SLU(out_features,out_features,bias=False,gain=columnGain,**kwargs)
        
    def forward(self,input):
        step=self.MainSLU(input)
        column=self.ColumnSLU(step)
        return step,column
