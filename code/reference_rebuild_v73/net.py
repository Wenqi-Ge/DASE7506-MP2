"""Candidate set-attention models used during training."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualAttention(nn.Module):
    def __init__(self,width,dropout,activation):
        super().__init__();self.activation=activation
        self.norm_attention=nn.LayerNorm(width);self.norm_feedforward=nn.LayerNorm(width)
        self.qkv=nn.Linear(width,3*width);self.attended=nn.Linear(width,width)
        self.expand=nn.Linear(width,2*width);self.contract=nn.Linear(2*width,width);self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        n,t,w=x.shape;d=w//4
        q,k,v=self.qkv(self.norm_attention(x)).chunk(3,-1)
        q,k,v=[a.reshape(n,t,4,d).transpose(1,2) for a in (q,k,v)]
        probabilities=self.dropout(torch.softmax((q@k.transpose(-1,-2))/(d**.5),-1))
        out=(probabilities@v).transpose(1,2).reshape(n,t,w);x=x+self.dropout(self.attended(out))
        out=self.expand(self.norm_feedforward(x));out=torch.relu(out) if self.activation=='relu' else F.gelu(out,approximate='tanh')
        return x+self.dropout(self.contract(self.dropout(out)))


class SetVerifier(nn.Module):
    def __init__(self,dimension,config):
        super().__init__();width=config['width'];self.stem=nn.Linear(dimension,width)
        self.layers=nn.ModuleList(ResidualAttention(width,config['dropout'],config['activation']) for _ in range(config['layers']))
        self.final_norm=nn.LayerNorm(width) if config['final_norm'] else nn.Identity();self.head=nn.Linear(width,1)
        self.direct=nn.Linear(dimension,1,bias=False) if config['skip'] else None
        if self.direct is not None:nn.init.zeros_(self.direct.weight)
    def forward(self,x):
        h=self.stem(x)
        for layer in self.layers:h=layer(h)
        output=self.head(self.final_norm(h)).squeeze(-1)
        return output if self.direct is None else output+self.direct(x).squeeze(-1)
