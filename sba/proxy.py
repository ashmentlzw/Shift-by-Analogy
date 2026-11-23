"""One train/inference scoring function; labels never enter representations."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ResidualProjection(nn.Module):
    def __init__(self,d,width):
        super().__init__();self.net=nn.Sequential(nn.Linear(d,width),nn.SiLU(),nn.Linear(width,d))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return F.normalize(x+self.net(x),dim=-1)

class JointProxy(nn.Module):
    def __init__(self,asset,width=128,attention=32,gain=.1):
        super().__init__();bank=asset['bank'];d=bank['text'].shape[-1];self.d=d;self.gain=gain;self.attention=attention
        self.attribute=nn.Parameter(asset['primitive_text'][:len(asset['attrs'])].clone())
        self.object=nn.Parameter(asset['primitive_text'][len(asset['attrs']):].clone())
        self.visual_heads=nn.ModuleDict({k:ResidualProjection(d,width) for k in ['composition','attribute','object']})
        self.query=nn.Linear(d,attention,bias=False);self.key=nn.Linear(d,attention,bias=False)
        self.shift=nn.Sequential(nn.Linear(3*d,width),nn.SiLU(),nn.Linear(width,d))
        nn.init.zeros_(self.shift[-1].weight);nn.init.zeros_(self.shift[-1].bias)
        self.gate=nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)
        for key in ['text','residual','cross','local_ids','local_mask','cross_available']:
            self.register_buffer('support_'+key,bank[key].clone(),persistent=False)
        self.register_buffer('pair_ids',asset['pair_ids'].clone(),persistent=False)

    def visual(self,v):
        gate=self.gate(v).sigmoid();c=self.visual_heads['composition'](v)
        a=self.visual_heads['attribute'](v);o=self.visual_heads['object'](v)
        return torch.cat([v,self.gain*gate*c,self.gain*(1-gate)*c,self.gain*a,self.gain*o],-1)

    def text(self,t,ids):
        rc=self.support_cross[ids];local=self.support_local_ids[ids];mask=self.support_local_mask[ids]
        keys=self.support_text[local].masked_fill(~mask[:,:,None],0)
        residual=self.support_residual[local].masked_fill(~mask[:,:,None],0)
        logits=(t[:,None]*keys).sum(-1)/.07+torch.einsum('bd,bkd->bk',self.query(t),self.key(keys))/math.sqrt(self.attention)
        w=logits.masked_fill(~mask,-1e4).softmax(-1)*mask
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-12)
        rf=(w[:,:,None]*residual).sum(1);tf=(w[:,:,None]*keys).sum(1)
        cross=F.normalize(t+rc+self.shift(torch.cat([t,rc,torch.zeros_like(t)],-1)),dim=-1)
        family=F.normalize(t+rf+self.shift(torch.cat([t,rf,t-tf],-1)),dim=-1)
        has_c=self.support_cross_available[ids,None];has_f=mask.any(-1)[:,None]
        # Either branch falls back to its available counterpart. Neither -> raw current text.
        qx=torch.where(has_c,cross,torch.where(has_f,family,t))
        qf=torch.where(has_f,family,torch.where(has_c,cross,t))
        pairs=self.pair_ids[ids]
        return torch.cat([t,qx,qf,F.normalize(self.attribute,dim=-1)[pairs[:,0]],F.normalize(self.object,dim=-1)[pairs[:,1]]],-1)

    def scores(self,image,text,scale):
        d=self.d
        return dict(fused=scale*(image@text.T),main=scale*(image[:,:d]@text[:,:d].T),
            visual=scale*((image[:,d:3*d]/self.gain)@text[:,d:3*d].T),
            attribute=scale*((image[:,3*d:4*d]/self.gain)@F.normalize(self.attribute,dim=-1).T),
            object=scale*((image[:,4*d:]/self.gain)@F.normalize(self.object,dim=-1).T))
