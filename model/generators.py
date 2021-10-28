import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):

    def __init__(self, d_model, voc_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)
        print('Using vanilla Generator')

    def forward(self, x):
        '''
        Inputs:
            x: (B, Sc, Dc)
        Outputs:
            (B, seq_len, voc_size)
        '''
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)

class MDVCGenerator(nn.Module):
    
    def __init__(self, d_model, voc_size, dout_p=0.1):
        super(MDVCGenerator, self).__init__()
        self.linear = nn.Linear(d_model, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)
        print('using MDVC Generator')
        
    def forward(self, x):
        x = self.linear(x)        
        x = self.linear2(self.dropout(F.relu(x)))        
        return F.log_softmax(x, dim=-1)