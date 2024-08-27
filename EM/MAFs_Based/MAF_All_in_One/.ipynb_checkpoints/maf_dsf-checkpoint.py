import sys
import os
cwd = '/mnt/users/joao.pires/norm_flow/EM/MAF_All_in_One'
sys.path.append(cwd)

import torch.nn as nn
import torch
import nn as nn_
import flows
import utils
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class MAF(nn.Module):
    
    def __init__(self, p, flowtype = 'dsf'):
        
        super().__init__()
        # self.args = args        
        # self.__dict__.update(args.__dict__)
        self.p = p
        
        dim = p
        dimc = 1
        dimh = 50
        # flowtype = 'dsf'
        num_flow_layers = 5
        num_ds_dim = 16
        num_ds_layers = 1
        fixed_order = True
        num_hid_layers = 1
                 
        act = nn.ELU()
        if flowtype == 'affine':
            flow = flows.IAF
        elif flowtype == 'dsf':
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs)
        elif flowtype == 'ddsf':
            flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                                  num_ds_layers=num_ds_layers,
                                                  **kwargs)
        
        
        sequels = [nn_.SequentialFlow(
            flow(dim=dim,
                 hid_dim=dimh,
                 context_dim=dimc,
                 num_layers=num_hid_layers+1,
                 activation=act,
                 fixed_order=fixed_order),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(dim, dimc),]
                
                
        self.flow = nn.Sequential(
                *sequels)
        
        self.cuda = True
        
        if self.cuda:
            self.flow = self.flow.cuda()
        
        # self.add_module('flow', self.flow)
        
        
    def density(self, spl):
        n = spl.size(0)
        context = Variable(torch.FloatTensor(n, 1).zero_()) 
        lgd = Variable(torch.FloatTensor(n).zero_())
        zeros = Variable(torch.FloatTensor(n, self.p).zero_())
        if self.cuda:
            context = context.cuda()
            lgd = lgd.cuda()
            zeros = zeros.cuda()
            
        z, logdet, _ = self.flow((spl, lgd, context))
        losses = - utils.log_normal(z, zeros, zeros+1.0).sum(1) - logdet
        return z, losses

    def loss(self, x):
        loss_ = self.density(x)[1]
        return loss_
        
    def state_dict(self):
        return self.flow.state_dict()

    def load_state_dict(self, states):
        self.flow.load_state_dict(states)
         
    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.flow.parameters(),
                                self.clip)
