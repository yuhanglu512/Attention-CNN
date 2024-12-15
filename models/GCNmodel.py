import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, ELU
from torch_geometric.nn import GCNConv, radius_graph, MessagePassing
from torch_geometric.nn import global_mean_pool
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import r2_score
from torch_scatter import scatter
import torch
from torch import Tensor
from typing import Optional, List, Tuple, Union
import math
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch._torch_docs import reproducibility_notes
from torch.utils.data import DataLoader,TensorDataset
import random
import math
import os
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.decomposition import PCA
from mendeleev import element
from brokenaxes import brokenaxes


outer_e = []+4*['s'] \
    + 6*['p'] \
    + 2*['s'] \
    + 6*['p'] \
    + 2*['s'] \
    + 10*['d'] \
    + 6*['p'] \
    + 2*['s'] \
    + 10*['d'] \
    + 6*['p'] \
    + 2*['s'] \
    + 15*['f'] \
    + 9*['d'] \
    + 6*['p'] \
    + 2*['s'] \
    + 15*['f'] \
    + 9*['d'] 

class HeuslerAlloy(torch.nn.Module):
    def __init__(self, embedding_num=128,Fc_num=[256,128,1], num_filters=128,
                 hidden_channels=128,num_interaction=5,cutoff=6,num_gaussians=50,
                 read_out='add'):
        super(HeuslerAlloy, self).__init__()
        self.embedding = Embedding(100, embedding_dim=embedding_num)
        self.lin=nn.Linear(embedding_num, hidden_channels)
        self.readout=read_out
        self.distance_expansion = GaussianSmearing(0.0,stop=cutoff, num_gaussians=num_gaussians)
        self.hidden_channels=hidden_channels
        self.interactions = nn.ModuleList()
        self.num_interaction=num_interaction
        for i in range(num_interaction):
            if i==0:
                task=InteractionBlock(embedding_num, hidden_channels, num_gaussians, num_filters, cutoff)
            else:
                task=InteractionBlock(hidden_channels, hidden_channels, num_gaussians, num_filters, cutoff)
            self.interactions.append(task)
        self.FCN=FCN([hidden_channels]+Fc_num,end_with_activation=True)

    def forward(self, x, edge_index, pos, batch): 
        atomic_embedding=self.embedding(torch.tensor(x[:,0],dtype=torch.int32))
        #h=torch.concatenate([atomic_embedding,x[:,1:3],(x[:,3]-x[:,4]).reshape(-1,1)],dim=1)
        h=self.embedding(torch.tensor(x[:,0],dtype=torch.int32))
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        # Initialize the batch tensor.
        batch_attr = batch

        # get the unique elements according to the batch.
        z=x[:,0]
        batch_size = batch_attr.unique().shape[0]
        # batch_c_list is the middle variable, and the batch_c is the concatenated variable.
        batch_c_list = []
        batch_c_index = []
        b_components_list = []
        for b in range(batch_size):
            b_index = torch.nonzero(batch_attr == b)
            b_z = z[b_index]
            b_components = b_z.unique()
            idx_comp = []
            for element in b_components:
                element_index = torch.nonzero(b_z == element.item(), as_tuple=True)
                idx_comp.append(b_index[element_index])
            batch_c_index.append(idx_comp)
            batch_c_list.append(torch.ones_like(input=b_components) * b)
            b_components_list.append(b_components)
        batch_c = torch.cat(tensors=batch_c_list, dim=0)
        batch_components = torch.cat(tensors=b_components_list, dim=0)
        # components = z.unique()
        # batch_c = torch.zeros_like(components)
        # idx_comp = []
        # for i in components:
        #     idx = np.where(z.cpu() == i.item())[0]
        #     idx_comp.append(idx)
        components_info = {'batch_c': batch_c, 'batch_c_index': batch_c_index, 'batch_components': batch_components}
        types = self.embedding(torch.tensor(batch_components,dtype=torch.int32))

        for interaction in self.interactions:
            types = types + interaction(h, components_info, edge_index, edge_weight, edge_attr)
        h=types
        h=self.FCN(h)
        out = [scatter(h, batch_c.type(torch.int64), dim=0, reduce=self.readout)]  # to keep the same data type in the output.
        return out


class InteractionBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_gaussians, num_filters, cutoff):
        super(InteractionBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.LeakyReLU(),
            nn.Linear(num_filters, num_filters),
        )
        self.conv = CFConv(in_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = nn.LeakyReLU()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, idx_comp, edge_index, edge_weight, edge_attr):
        x = self.conv(x, idx_comp, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, mlp, cutoff):
        super(CFConv, self).__init__(aggr='add')
        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.lin2 = nn.Linear(num_filters, out_channels)
        self.mlp = mlp
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, idx_comp, edge_index, edge_weight, edge_attr):
        """
        E is the number of edges. N is the number of nodes.

        :param x: x has shape of [N, in_channels]; where N is the number of nodes.
        :param idx_comp: list. index of the specific component.
        :param edge_index: edge_index has shape of [2, E]
        :param edge_weight:
        :param edge_attr:
        :return:
        """
        # C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        epsilon = 1e-10
        C = self.cutoff / (epsilon + edge_weight.pow(2)) - 1
        W = self.mlp(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index.type(torch.int64), x=x, W=W)
        # collate the information of the specific materials in the idx_comp.
        batch_c_index = idx_comp['batch_c_index']
        x_merge = []
        for i in batch_c_index:
            n_comps = len(i)
            # comp_x is the information for the specific materials
            comp_x = []
            for j in range(n_comps):
                comp_x.append(x[i[j]])
            comp_x = [torch.mean(input=t, dim=0, ) for t in comp_x]
            comp_x = torch.stack(tensors=comp_x, dim=0)
            x_merge.append(comp_x)
        x_merge = torch.cat(tensors=x_merge, dim=0)
        x_merge = self.lin2(x_merge)
        return x_merge

    def message(self, x_j, W):
        # x_j has shape of [E, in_channels]
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class single_CNN(nn.Module):
    def __init__(self,is_embedding=True, channel_in=64, is_structure=True,
                 is_space_embedding=True, space_embedding_len=150,
                 channels=[120,120,120],padding='same',pool_size=[2,2,2],
                 kernel_size=[5,3,3],stride=1,block_num=[2,3,0],
                 is_pool=[False,False,False],fc=[256,1],
                 end_with_activation=False,):
        super().__init__()
        if is_embedding:
            if is_structure:
                self.embedding_len=channel_in-8
            else:
                self.embedding_len=channel_in-10
        else:
            self.embedding_len=channel_in-1
        self.embedding = nn.Embedding(100, self.embedding_len)
        self.space_embedding = nn.Embedding(230, space_embedding_len)
        self.is_embedding=is_embedding
        self.is_space_embedding=is_space_embedding

        assert len(channels)==len(kernel_size)==len(block_num)==len(is_pool), 'The length of each list should be the same'
        channels=[channel_in]+channels if is_embedding else [5]+channels

        size=10
        self.conv_block=nn.ModuleList()
        for i in range(len(block_num)):
            for j in range(block_num[i]):
                if j==0:
                    self.conv_block.append(Convblock(channels[i],channels[i+1],kernel_size[i],stride,padding='same'))                
                else:
                    self.conv_block.append(Convblock(channels[i+1],channels[i+1],kernel_size[i],stride,padding='same'))
            if is_pool[i]:
                self.conv_block.append(nn.MaxPool2d(kernel_size=pool_size[i], stride=pool_size[i]))
                size=size//pool_size[i]

        size=int(size)
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.conv_block.append(nn.LayerNorm([size,size]))
        self.conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            length0=channels[-1]*size**2+space_embedding_len
        else:
            length0=sum(channels[-1])*size**2
        self.fc_block=FCN([length0]+fc,end_with_activation=end_with_activation)

    def forward(self, x, space_group=torch.tensor([216,225],device='cuda'),\
                energy=None,vol=torch.tensor([70,90],device='cuda')):
        if self.is_embedding:
            appendix=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,1,10,10)
            site_arg=x!=0
            element_arg=torch.argwhere(x.reshape(-1,100)!=0)[:,1]
            embedding=torch.zeros(x.shape[0],x.shape[1],10,10,self.embedding_len,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,4,2,3,1).squeeze()
            x=torch.concatenate([x,embedding,appendix],dim=1)
        else:
            #x=x[:,-4:1,:,:].reshape(-1,5,10,10)
            x=torch.concat([x[:,-4:,:,:],x[:,0,:,:].reshape(-1,1,10,10)],dim=1)
        
        for i in range(len(self.conv_block)):
            x=self.conv_block[i](x)
        
        if self.is_space_embedding and space_group is not None:
            space_embedding=self.space_embedding(space_group)
            x=torch.concatenate([x,space_embedding],dim=1)

        x=self.fc_block(x)
        return x


class attention_CNN(nn.Module):
    def __init__(self,is_embedding=True, channel_in=64, is_appendix=True,is_vol=True,
                 is_space_embedding=True, space_embedding_len=150,
                 channels=[[88,40],[40,88],[40,88]],padding='same',pool_size=[2,2,2],
                 kernel_size=[[5,3],[5,3],[5,3]],stride=1,block_num=[2,3,3],
                 is_pool=[False,False,False],fc=[512,128,1],
                 use_resnet=True,norm_first=False,norm_activation=True,
                 use_channels=False,use_spatial=True,use_encoder=False,
                 num_heads=8,dim_feedforward=256,attention_num=[1,0,0],
                 end_with_activation=False,):
        super().__init__()
        if is_embedding:
            if is_appendix:
                self.embedding_len=channel_in-8
            else:
                self.embedding_len=channel_in-5
            self.embedding = nn.Embedding(100, self.embedding_len)

        self.is_appendix=is_appendix
        self.space_embedding = nn.Embedding(230, space_embedding_len)
        self.is_embedding=is_embedding
        self.is_space_embedding=is_space_embedding
        self.is_vol=is_vol

        assert len(channels)==len(kernel_size)==len(block_num)==len(is_pool)==len(attention_num), 'The length of each list should be the same'
        channels=[[channel_in]]+channels if is_embedding else [[8]]+channels

        size=10
        self.conv_block=nn.ModuleList()
        for i in range(len(block_num)):
            #if i==0 and use_spatial:
            #    self.conv_block.append(CBAM(sum(channels[i]),use_spatial=True))
            for j in range(attention_num[i]):
                if use_encoder:
                    self.conv_block.append(TransformerEncoderLayer(d_model=sum(channels[i]), nhead=num_heads,\
                                            dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
                else:
                    self.conv_block.append(MultiheadAttention(sum(channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(block_num[i]):
                if i==0 and j==0 and use_spatial:
                    self.conv_block.append(CBAM(sum(channels[i]),use_spatial=True))
                if j==0:
                    self.conv_block.append(multikernel_Convblock(sum(channels[i]),channels[i+1],\
                                                               kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.conv_block.append(multikernel_Convblock(sum(channels[i+1]),channels[i+1],\
                                                               kernel_size[i],stride,padding,use_resnet))
                
            if is_pool[i]:
                self.conv_block.append(nn.MaxPool2d(kernel_size=pool_size[i], stride=pool_size[i]))
                size=size//pool_size[i]

        size=int(size)
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.conv_block.append(nn.LayerNorm([size,size]))
        self.conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            if is_vol:
                length0=sum(channels[-1])*size**2+space_embedding_len+1
            else:
                length0=sum(channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(channels[-1])*size**2
        self.fc_block=FCN([length0]+fc,end_with_activation=end_with_activation)

    def forward(self, x, space_group=torch.tensor([216,225],device='cuda'),\
                vol=torch.tensor([[20],[30]],device='cuda'),return_attention=False,task='Mag'):
        mask=torch.matmul(x[:,0,:,:].reshape(-1,100,1),x[:,0,:,:].reshape(-1,1,100))
        if self.is_embedding:
            x_other=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,100)
            mask=torch.matmul(x.reshape(-1,100,1),x.reshape(-1,1,100))
            x=x.reshape(-1,1,10,10)
            site_arg=x!=0
            element_arg=torch.argwhere(x.reshape(-1,100)!=0)[:,1]
            embedding=torch.zeros(x.shape[0],x.shape[1],10,10,self.embedding_len,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,4,2,3,1).squeeze()
            #site_arg=x!=0
            #element_arg=torch.argwhere(site_arg)[:,1]
            #embedding=torch.zeros(x.shape[0],100,self.embedding_len,device='cuda')
            #embedding[site_arg]=self.embedding(element_arg)
            #embedding=embedding.permute(0,2,1).reshape(-1,self.embedding_len,10,10)
            #x=x.reshape(-1,1,10,10)
            if self.is_appendix:
                if x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                else:
                    raise ValueError('input x donnot contain correct appendix')
            else:
                if x_other.shape[1]==4:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                elif x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other[:,-4:,:,:]],dim=1)
        
        attention_list=[]
        for i in range(len(self.conv_block)):
            if isinstance(self.conv_block[i],TransformerEncoderLayer) or isinstance(self.conv_block[i],MultiheadAttention):
                if return_attention:
                    x,attention_weight=self.conv_block[i](x,mask,return_attention)
                    attention_list.append(attention_weight)
                else:
                    x=self.conv_block[i](x,mask)
            else:
                x=self.conv_block[i](x)
        
        if self.is_space_embedding and space_group is not None:
            space_embedding=self.space_embedding(space_group)
            if self.is_vol:
                x=torch.concatenate([x,space_embedding,vol],dim=1)
            else:
                x=torch.concatenate([x,space_embedding],dim=1)

        x=self.fc_block(x)
        if return_attention:
            return x,attention_list
        else:
            return x


class attention_structure(nn.Module):
    def __init__(self,is_embedding=True, channel_in=256,nhead=8,
                 dim_feedforward=[256,256,256],block_num=[5,5,5],
                 is_space_embedding=True, space_embedding_len=150,
                 fc=[1024,256,1],norm_activation=True,norm_first=True):
        super().__init__()
        self.is_embedding=is_embedding
        self.embedding_len=channel_in-8
        self.embedding = nn.Embedding(100, self.embedding_len)
        self.block_num=block_num
        self.is_space_embedding=is_space_embedding
        self.space_embedding = nn.Embedding(230, space_embedding_len)

        self.attention_block=nn.ModuleList()
        for i in range(len(block_num)):
            for k in range(block_num[i]):
                self.attention_block.append(TransformerEncoderLayer(d_model=channel_in, nhead=nhead, dim_feedforward=dim_feedforward[i],norm_activation=norm_activation,norm_first=norm_first))        
        #self.attention_block.append(nn.Flatten())

        # define the fc block
        self.fc_block=FCN([channel_in*100+space_embedding_len]+fc,end_with_activation=False)

    def forward(self, x, space_group=torch.tensor([216,225],device='cuda'),\
                energy=None,vol=torch.tensor([70,90],device='cuda')):
        if self.is_embedding:
            appendix=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,1,10,10)
            site_arg=x!=0
            element_arg=torch.argwhere(x.reshape(-1,100)!=0)[:,1]
            embedding=torch.zeros(x.shape[0],x.shape[1],10,10,self.embedding_len,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,4,2,3,1).squeeze()
            x=torch.concatenate([x,embedding,appendix],dim=1)
        
        #x=x.view(x.shape[0],x.shape[1],-1).permute(0,2,1)

        for i in range(len(self.attention_block)):
            x=self.attention_block[i](x)
        
        x=nn.Flatten()(x)
        
        if self.is_space_embedding and space_group is not None:
            space_embedding=self.space_embedding(space_group)
            x=torch.concatenate([x,space_embedding],dim=1)

        x=self.fc_block(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=64, nhead=8, dim_feedforward=256, dropout=0, 
                 norm_first=False, norm_activation=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,norm_activation=norm_activation)
        self.feed_forward = nn.Sequential(
            Transpose(1,2),
            nn.Linear(d_model, dim_feedforward),
            #nn.LayerNorm(dim_feedforward),
            #nn.InstanceNorm1d(100,affine=True),
            nn.BatchNorm1d(100),
            Transpose(1,2),
            #nn.LayerNorm(100),
            #nn.InstanceNorm1d(dim_feedforward,affine=True),
            #nn.BatchNorm1d(dim_feedforward),
            nn.LeakyReLU(),
            #nn.Dropout(dropout),
            Transpose(1,2),
            nn.Linear(dim_feedforward, d_model),
            #nn.LayerNorm(d_model),
            #nn.InstanceNorm1d(100,affine=True),
            nn.BatchNorm1d(100),
            Transpose(1,2),
            #nn.LayerNorm(100),
            #nn.InstanceNorm1d(d_model,affine=True),
            #nn.BatchNorm1d(d_model),
        )
        #self.norm1 = nn.LayerNorm(100)
        #self.norm2 = nn.LayerNorm(100)
        #self.norm1 = nn.InstanceNorm1d(d_model,affine=True)
        #self.norm2 = nn.InstanceNorm1d(d_model,affine=True)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        #self.norm1 = nn.Sequential(
        #    Transpose(1,2),
        #    nn.LayerNorm(d_model),
        #    Transpose(1,2),
        #)
        #self.norm2 = nn.Sequential(
        #    Transpose(1,2),
        #    nn.LayerNorm(d_model),
        #    Transpose(1,2),
        #)
        #self.norm1 = nn.Sequential(
        #    Transpose(1,2),
        #    nn.InstanceNorm1d(100,affine=True),
        #    Transpose(1,2),
        #)
        #self.norm2 = nn.Sequential(
        #    Transpose(1,2),
        #    nn.InstanceNorm1d(100,affine=True),
        #    Transpose(1,2),
        #)
        #self.norm1 = nn.Sequential(
        #    Transpose(1,2),
        #    nn.BatchNorm1d(100),
        #    Transpose(1,2),
        #)
        #self.norm2 = nn.Sequential(
        #    Transpose(1,2),
        #    nn.BatchNorm1d(100),
        #    Transpose(1,2),
        #)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()
        self.norm_first=norm_first

    def forward(self, src, src_mask=None, return_attention=False):
        src=src.view(src.shape[0],src.shape[1],-1)
        if return_attention:
            src2, attn_weights = self.self_attn(src, src_mask, return_attention=return_attention)
        else:
            src2 = self.self_attn(src, src_mask)
        if src2.dim()==4:
            src2=src2.view(src2.shape[0],src2.shape[1],-1)
        if self.norm_first:
            src2 = self.norm1(self.dropout1(src2))
            src = src+src2
            src2 = self.feed_forward(src)
            src2 = self.norm2(self.dropout2(src2))
            src = src+src2
        else:
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.feed_forward(src)
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        src=src.view(src.shape[0],src.shape[1],10,10)

        if return_attention:
            return src, attn_weights
        else:
            return src

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, dropout=0.0,norm_activation=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for query, key, and value
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # Output projection
        if norm_activation:
            self.out_linear = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                #nn.LayerNorm([embed_dim]),
                Transpose(1,2),
                nn.LayerNorm([100]),
                nn.LeakyReLU(),
            )
        else:
            self.out_linear = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                Transpose(1,2),
            )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None,return_attention=False):
        # Linear projections
        if x.dim()==4:
            x=x.view(x.shape[0],x.shape[1],-1)
        
        x=x.transpose(1,2)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split into multiple heads
        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            if mask.dim()!=scores.dim():
                mask=mask.unsqueeze(1)
                mask=mask.repeat(1,self.num_heads,1,1)
            scores = scores.masked_fill(mask == 0, -10000)
        attn_weights = self.softmax_2d(scores)
        attn_output = torch.matmul(self.dropout(attn_weights), v)

        # Combine heads and project to output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            attn_output.size(0), -1, self.embed_dim
        )
        output = self.out_linear(attn_output)
        output=output.view(output.shape[0],output.shape[1],10,10)

        if return_attention:
            return output, attn_weights
        else:
            return output
        
    def softmax_2d(self,x):
        orginal_shape=x.shape
        x=x.view(*orginal_shape[:-2],-1)
        x=torch.softmax(x,dim=-1)
        return x.view(*orginal_shape)

#class TransformerEncoderLayer(nn.Module):
#    def __init__(self, d_model=64, nhead=8, dim_feedforward=256, dropout=0, 
#                 norm_first=False, norm_activation=False):
#        super().__init__()
#        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,norm_activation=norm_activation)
#        self.feed_forward = nn.Sequential(
#            Transpose(1,2),
#            nn.Linear(d_model, dim_feedforward),
#            #nn.LayerNorm(dim_feedforward),
#            #nn.InstanceNorm1d(100,affine=True),
#            nn.BatchNorm1d(100),
#            Transpose(1,2),
#            #nn.LayerNorm(100),
#            #nn.InstanceNorm1d(dim_feedforward,affine=True),
#            #nn.BatchNorm1d(dim_feedforward),
#            nn.LeakyReLU(),
#            #nn.Dropout(dropout),
#            Transpose(1,2),
#            nn.Linear(dim_feedforward, d_model),
#            #nn.LayerNorm(d_model),
#            #nn.InstanceNorm1d(100,affine=True),
#            nn.BatchNorm1d(100),
#            Transpose(1,2),
#            #nn.LayerNorm(100),
#            #nn.InstanceNorm1d(d_model,affine=True),
#            #nn.BatchNorm1d(d_model),
#        )
#        #self.norm1 = nn.LayerNorm(100)
#        #self.norm2 = nn.LayerNorm(100)
#        #self.norm1 = nn.InstanceNorm1d(d_model,affine=True)
#        #self.norm2 = nn.InstanceNorm1d(d_model,affine=True)
#        self.norm1 = nn.BatchNorm1d(d_model)
#        self.norm2 = nn.BatchNorm1d(d_model)
#        #self.norm1 = nn.Sequential(
#        #    Transpose(1,2),
#        #    nn.LayerNorm(d_model),
#        #    Transpose(1,2),
#        #)
#        #self.norm2 = nn.Sequential(
#        #    Transpose(1,2),
#        #    nn.LayerNorm(d_model),
#        #    Transpose(1,2),
#        #)
#        #self.norm1 = nn.Sequential(
#        #    Transpose(1,2),
#        #    nn.InstanceNorm1d(100,affine=True),
#        #    Transpose(1,2),
#        #)
#        #self.norm2 = nn.Sequential(
#        #    Transpose(1,2),
#        #    nn.InstanceNorm1d(100,affine=True),
#        #    Transpose(1,2),
#        #)
#        #self.norm1 = nn.Sequential(
#        #    Transpose(1,2),
#        #    nn.BatchNorm1d(100),
#        #    Transpose(1,2),
#        #)
#        #self.norm2 = nn.Sequential(
#        #    Transpose(1,2),
#        #    nn.BatchNorm1d(100),
#        #    Transpose(1,2),
#        #)
#        self.dropout1 = nn.Dropout(dropout)
#        self.dropout2 = nn.Dropout(dropout)
#        self.activation = nn.LeakyReLU()
#        self.norm_first=norm_first
#
#    def forward(self, src, src_mask=None, return_attention=False):
#        src=src.view(src.shape[0],src.shape[1],-1)
#        if return_attention:
#            src2, attn_weights = self.self_attn(src, src_mask, return_attention=return_attention)
#        else:
#            src2 = self.self_attn(src, src_mask)
#        if src2.dim()==4:
#            src2=src2.view(src2.shape[0],src2.shape[1],-1)
#        if self.norm_first:
#            src2 = self.norm1(self.dropout1(src2))
#            src = src+src2
#            src2 = self.feed_forward(src)
#            src2 = self.norm2(self.dropout2(src2))
#            src = src+src2
#        else:
#            src = src + self.dropout1(src2)
#            src = self.norm1(src)
#            src2 = self.feed_forward(src)
#            src = src + self.dropout2(src2)
#            src = self.norm2(src)
#        
#        src=src.view(src.shape[0],src.shape[1],10,10)
#
#        if return_attention:
#            return src, attn_weights
#        else:
#            return src
#
#class MultiheadAttention(nn.Module):
#    def __init__(self, embed_dim=64, num_heads=8, dropout=0.0,norm_activation=False):
#        super().__init__()
#        self.embed_dim = embed_dim
#        self.num_heads = num_heads
#        self.head_dim = embed_dim // num_heads
#
#        # Linear projections for query, key, and value
#        self.q_linear = nn.Linear(embed_dim, embed_dim)
#        self.k_linear = nn.Linear(embed_dim, embed_dim)
#        self.v_linear = nn.Linear(embed_dim, embed_dim)
#
#        # Output projection
#        if norm_activation:
#            self.out_linear = nn.Sequential(
#                nn.Linear(embed_dim, embed_dim),
#                #nn.LayerNorm([embed_dim]),
#                Transpose(1,2),
#                nn.LayerNorm([100]),
#                nn.LeakyReLU(),
#            )
#        else:
#            self.out_linear = nn.Sequential(
#                nn.Linear(embed_dim, embed_dim),
#                Transpose(1,2),
#            )
#
#        self.dropout = nn.Dropout(p=dropout)
#
#    def forward(self, x, mask=None,return_attention=False):
#        # Linear projections
#        if x.dim()==4:
#            x=x.view(x.shape[0],x.shape[1],-1)
#        
#        x=x.transpose(1,2)
#        q = self.q_linear(x)
#        k = self.k_linear(x)
#        v = self.v_linear(x)
#
#        # Split into multiple heads
#        q = q.view(q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
#        k = k.view(k.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
#        v = v.view(v.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
#
#        # Scaled dot-product attention
#        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
#        if mask is not None:
#            if mask.dim()!=scores.dim():
#                mask=mask.unsqueeze(1)
#                mask=mask.repeat(1,self.num_heads,1,1)
#            scores = scores.masked_fill(mask == 0, -10000)
#        attn_weights = torch.softmax(scores, dim=-1)
#        attn_output = torch.matmul(self.dropout(attn_weights), v)
#
#        # Combine heads and project to output
#        attn_output = attn_output.transpose(1, 2).contiguous().view(
#            attn_output.size(0), -1, self.embed_dim
#        )
#        output = self.out_linear(attn_output)
#        output=output.view(output.shape[0],output.shape[1],10,10)
#
#        if return_attention:
#            return output, attn_weights
#        else:
#            return output


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class attention_CNN_bitask(nn.Module):
    def __init__(self,is_embedding=True, channel_in=64, is_appendix=True,is_vol=True,
                 is_space_embedding=True, space_embedding_len=150,
                 common_channels=[[88,40],[40,88],[40,88]],padding='same',
                 common_kernel_size=[[5,3],[5,3],[5,3]],stride=1,common_block_num=[0,0,0],
                 mag_channels=[[88,40],[40,88],[40,88]],mag_kernel_size=[[5,3],[5,3],[5,3]],
                 mag_block_num=[2,3,3],a_channels=[[88,40],[40,88],[40,88]],
                 a_kernel_size=[[5,3],[5,3],[5,3]],a_block_num=[2,3,3],
                 mag_fc=[512,128,1],a_fc=[512,128,1],
                 use_resnet=True,norm_first=False,norm_activation=True,
                 use_channels=False,use_spatial=True,use_encoder=True,
                 num_heads=8,dim_feedforward=256,attention_num=[1,0,0],
                 end_with_activation=True,):
        super().__init__()
        if is_embedding:
            if is_appendix:
                self.embedding_len=channel_in-8
            else:
                self.embedding_len=channel_in-5
            self.embedding = nn.Embedding(100, self.embedding_len)

        self.is_appendix=is_appendix
        self.space_embedding = nn.Embedding(230, space_embedding_len)
        self.is_embedding=is_embedding
        self.is_space_embedding=is_space_embedding
        self.is_vol=is_vol

        assert len(common_channels)==len(common_kernel_size)==len(common_block_num)==len(attention_num), 'The length of each list should be the same'
        common_channels=[[channel_in]]+common_channels if is_embedding else [[5]]+common_channels

        size=10
        #position_record=[]
        self.common_conv_block=nn.ModuleList()
        for i in range(len(common_block_num)):
            #for j in range(attention_num[i]):
            #    #position_record.append(len(self.common_conv_block))
            #    if use_encoder:
            #        self.common_conv_block.append(TransformerEncoderLayer(d_model=sum(common_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.common_conv_block.append(MultiheadAttention(sum(common_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(common_block_num[i]):
                if i==0 and j==0 and use_spatial:
                    self.common_conv_block.append(CBAM(sum(common_channels[i]),use_spatial=True))
                if j==0:
                    self.common_conv_block.append(multikernel_Convblock(sum(common_channels[i]),common_channels[i+1],\
                                                               common_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.common_conv_block.append(multikernel_Convblock(sum(common_channels[i+1]),common_channels[i+1],\
                                                               common_kernel_size[i],stride,padding,use_resnet))
        #self.position_record=position_record
        
        mag_channels=[common_channels[-1] if sum(common_block_num) else [channel_in]]+mag_channels
        self.mag_conv_block=nn.ModuleList()
        for i in range(len(mag_block_num)):
            for j in range(attention_num[i]):
            #    position_record.append(len(self.mag_conv_block))
                if use_encoder:
                    self.mag_conv_block.append(TransformerEncoderLayer(d_model=sum(mag_channels[i]), nhead=num_heads,\
                                            dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
                else:
                    self.mag_conv_block.append(MultiheadAttention(sum(mag_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(mag_block_num[i]):
                if i==0 and j==0 and use_spatial:
                    self.mag_conv_block.append(CBAM(sum(mag_channels[i]),use_spatial=True))
                if j==0:
                    self.mag_conv_block.append(multikernel_Convblock(sum(mag_channels[i]),mag_channels[i+1],\
                                                               mag_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.mag_conv_block.append(multikernel_Convblock(sum(mag_channels[i+1]),mag_channels[i+1],\
                                                               mag_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.position_record=position_record
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.mag_conv_block.append(nn.LayerNorm([size,size]))
        self.mag_conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            if is_vol:
                length0=sum(mag_channels[-1])*size**2+space_embedding_len+1
            else:
                length0=sum(mag_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(mag_channels[-1])*size**2
        self.mag_fc_block=FCN([length0]+mag_fc,end_with_activation=end_with_activation)

        a_channels=[common_channels[-1] if sum(common_block_num) else [channel_in]]+a_channels
        self.a_conv_block=nn.ModuleList()
        for i in range(len(a_block_num)):
            #for j in range(attention_num[i]):
            #    position_record.append(len(self.a_conv_block))
            #    if use_encoder:
            #        self.a_conv_block.append(TransformerEncoderLayer(d_model=sum(a_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.a_conv_block.append(MultiheadAttention(sum(a_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(a_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.a_conv_block.append(CBAM(sum(a_channels[i]),use_spatial=True))
                if j==0:
                    self.a_conv_block.append(multikernel_Convblock(sum(a_channels[i]),a_channels[i+1],\
                                                               a_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.a_conv_block.append(multikernel_Convblock(sum(a_channels[i+1]),a_channels[i+1],\
                                                               a_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.position_record=position_record
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.a_conv_block.append(nn.LayerNorm([size,size]))
        self.a_conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            length0=sum(a_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(a_channels[-1])*size**2
        self.a_fc_block=FCN([length0]+a_fc,end_with_activation=end_with_activation)

    def forward(self, x, space_group=torch.tensor([216,225],device='cuda'),\
                return_attention=False,task='Mag'):
        if self.is_embedding:
            x_other=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,100)
            mask=torch.matmul(x.reshape(-1,100,1),x.reshape(-1,1,100))
            site_arg=x!=0
            element_arg=torch.argwhere(site_arg)[:,1]
            embedding=torch.zeros(x.shape[0],100,self.embedding_len,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,2,1).reshape(-1,self.embedding_len,10,10)
            x=x.reshape(-1,1,10,10)
            if self.is_appendix:
                if x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                else:
                    raise ValueError('input x donnot contain correct appendix')
            else:
                if x_other.shape[1]==4:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                elif x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other[:,-4:,:,:]],dim=1)
        
        attention_list=[]
        position_count=0
        for i in range(len(self.common_conv_block)):
            #if return_attention and i<=self.position_record[-1] \
            #    and i==self.position_record[position_count]:
            #    x,attention_weight=self.common_conv_block[i](x,src_mask=mask,return_attention=return_attention)
            #    attention_list.append(attention_weight)
            #    position_count+=1
            #elif i<=self.position_record[-1] and i==self.position_record[position_count]:
            #    x=self.common_conv_block[i](x,src_mask=mask)
            #    position_count+=1
            #else:
            #    x=self.common_conv_block[i](x)
            if isinstance(self.common_conv_block[i],TransformerEncoderLayer):
                if return_attention:
                    x,attention_weight=self.common_conv_block[i](x,src_mask=mask,return_attention=return_attention)
                    attention_list.append(attention_weight)
                else:
                    x=self.common_conv_block[i](x,src_mask=mask)
            else:
                x=self.common_conv_block[i](x)

        common_x=x
        for i in range(len(self.a_conv_block)):
            x=self.a_conv_block[i](x)
        if self.is_space_embedding and space_group is not None:
            space_embedding=self.space_embedding(space_group)
            x=torch.concatenate([x,space_embedding],dim=1)
        x=self.a_fc_block(x)
        if task=='Mag':
            v=((x.detach())**3/(2**(1/2))).detach()
            #v=a**3/(2**(1/2))
            x=common_x
            for i in range(len(self.mag_conv_block)):
                if isinstance(self.mag_conv_block[i],TransformerEncoderLayer):
                    if return_attention:
                        x,attention_weight=self.mag_conv_block[i](x,src_mask=mask,return_attention=return_attention)
                        attention_list.append(attention_weight)
                    else:
                        x=self.mag_conv_block[i](x,src_mask=mask)
                else:
                    x=self.mag_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                if self.is_vol:
                    x=torch.concatenate([x,space_embedding,v],dim=1)
                else:
                    x=torch.concatenate([x,space_embedding],dim=1)
            x=self.mag_fc_block(x)

        if return_attention:
            return x,attention_list
        else:
            return x


class attention_CNN_tritask(nn.Module):
    def __init__(self,is_embedding=True, channel_in=80, is_appendix=True,
                 is_space_embedding=True, space_embedding_len=150,
                 common_channels=[[88,40],[40,88],[40,88]],padding='same',
                 common_kernel_size=[[5,3],[5,3],[5,3]],stride=1,common_block_num=[1,1,0],
                 E0_channels=[[88,40],[40,88],[40,88]],E0_kernel_size=[[5,3],[5,3],[5,3]],
                 E0_block_num=[2,2,0],Ef_channels=[[88,40],[40,88],[40,88]],
                 Ef_kernel_size=[[5,3],[5,3],[5,3]],Ef_block_num=[2,2,0],
                 stability_channels=[[88,40],[40,88],[40,88]],stability_kernel_size=[[5,3],[5,3],[5,3]],
                 stability_block_num=[2,2,0],
                 E0_fc=[512,128,1],Ef_fc=[512,128,1],stability_fc=[512,128,1],
                 use_resnet=True,norm_first=False,norm_activation=True,
                 use_channels=False,use_spatial=True,use_encoder=True,
                 num_heads=8,dim_feedforward=256,attention_num=[1,0,0],):
        super().__init__()
        if is_embedding:
            if is_appendix:
                self.embedding_len=channel_in-8
            else:
                self.embedding_len=channel_in-5
            self.embedding = nn.Embedding(100, self.embedding_len)

        self.is_appendix=is_appendix
        self.space_embedding = nn.Embedding(230, space_embedding_len)
        self.is_embedding=is_embedding
        self.is_space_embedding=is_space_embedding

        assert len(common_channels)==len(common_kernel_size)==len(common_block_num)==len(attention_num), 'The length of each list should be the same'
        common_channels=[[channel_in]]+common_channels if is_embedding else [[5]]+common_channels

        size=10
        position_record=[]
        self.common_conv_block=nn.ModuleList()
        for i in range(len(common_block_num)):
            for j in range(attention_num[i]):
                position_record.append(len(self.common_conv_block))
                if use_encoder:
                    self.common_conv_block.append(TransformerEncoderLayer(d_model=sum(common_channels[i]), nhead=num_heads,\
                                            dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
                else:
                    self.common_conv_block.append(MultiheadAttention(sum(common_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(common_block_num[i]):
                if i==0 and j==0 and use_spatial:
                    self.common_conv_block.append(CBAM(sum(common_channels[i]),use_spatial=True))
                if j==0:
                    self.common_conv_block.append(multikernel_Convblock(sum(common_channels[i]),common_channels[i+1],\
                                                               common_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.common_conv_block.append(multikernel_Convblock(sum(common_channels[i+1]),common_channels[i+1],\
                                                               common_kernel_size[i],stride,padding,use_resnet))
        self.position_record=position_record
        
        E0_channels=[common_channels[-1]]+E0_channels
        self.E0_conv_block=nn.ModuleList()
        for i in range(len(E0_block_num)):
            #for j in range(attention_num[i]):
            #    position_record.append(len(self.E0_conv_block))
            #    if use_encoder:
            #        self.E0_conv_block.append(TransformerEncoderLayer(d_model=sum(E0_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.E0_conv_block.append(MultiheadAttention(sum(E0_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(E0_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.E0_conv_block.append(CBAM(sum(E0_channels[i]),use_spatial=True))
                if j==0:
                    self.E0_conv_block.append(multikernel_Convblock(sum(E0_channels[i]),E0_channels[i+1],\
                                                               E0_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.E0_conv_block.append(multikernel_Convblock(sum(E0_channels[i+1]),E0_channels[i+1],\
                                                               E0_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.position_record=position_record
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.E0_conv_block.append(nn.LayerNorm([size,size]))
        self.E0_conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            length0=sum(E0_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(E0_channels[-1])*size**2
        self.E0_fc_block=FCN([length0]+E0_fc,end_with_activation=False)

        Ef_channels=[common_channels[-1]]+Ef_channels
        self.Ef_conv_block=nn.ModuleList()
        for i in range(len(Ef_block_num)):
            #for j in range(attention_num[i]):
            #    position_record.append(len(self.a_conv_block))
            #    if use_encoder:
            #        self.a_conv_block.append(TransformerEncoderLayer(d_model=sum(Ef_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.a_conv_block.append(MultiheadAttention(sum(Ef_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(Ef_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.a_conv_block.append(CBAM(sum(Ef_channels[i]),use_spatial=True))
                if j==0:
                    self.Ef_conv_block.append(multikernel_Convblock(sum(Ef_channels[i]),Ef_channels[i+1],\
                                                               Ef_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.Ef_conv_block.append(multikernel_Convblock(sum(Ef_channels[i+1]),Ef_channels[i+1],\
                                                               Ef_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.position_record=position_record
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.Ef_conv_block.append(nn.LayerNorm([size,size]))
        self.Ef_conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            length0=sum(Ef_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(Ef_channels[-1])*size**2
        self.Ef_fc_block=FCN([length0]+Ef_fc,end_with_activation=False)

        stability_channels=[common_channels[-1]]+stability_channels
        self.stability_conv_block=nn.ModuleList()
        for i in range(len(stability_block_num)):
            #for j in range(attention_num[i]):
            #    position_record.append(len(self.stability_conv_block))
            #    if use_encoder:
            #        self.stability_conv_block.append(TransformerEncoderLayer(d_model=sum(stability_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.stability_conv_block.append(MultiheadAttention(sum(stability_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(stability_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.stability_conv_block.append(CBAM(sum(stability_channels[i]),use_spatial=True))
                if j==0:
                    self.stability_conv_block.append(multikernel_Convblock(sum(stability_channels[i]),stability_channels[i+1],\
                                                               stability_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.stability_conv_block.append(multikernel_Convblock(sum(stability_channels[i+1]),stability_channels[i+1],\
                                                               stability_kernel_size[i],stride,padding,use_resnet))
            
        size=int(size)
        #self.position_record=position_record
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.stability_conv_block.append(nn.LayerNorm([size,size]))
        self.stability_conv_block.append(nn.Flatten())

        if is_space_embedding:
            length0=sum(stability_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(stability_channels[-1])*size**2
        self.stability_fc_block=FCN([length0]+stability_fc,end_with_activation=True)

    def forward(self, x, space_group=torch.tensor([216,225],device='cuda'),\
                return_attention=False,task='E0'):
        if self.is_embedding:
            x_other=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,100)
            mask=torch.matmul(x.reshape(-1,100,1),x.reshape(-1,1,100))
            site_arg=x!=0
            element_arg=torch.argwhere(site_arg)[:,1]
            embedding=torch.zeros(x.shape[0],100,self.embedding_len,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,2,1).reshape(-1,self.embedding_len,10,10)
            x=x.reshape(-1,1,10,10)
            if self.is_appendix:
                if x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                else:
                    raise ValueError('input x donnot contain correct appendix')
            else:
                if x_other.shape[1]==4:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                elif x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other[:,-4:,:,:]],dim=1)
        
        attention_list=[]
        position_count=0
        for i in range(len(self.common_conv_block)):
            if return_attention and i<=self.position_record[-1] \
                and i==self.position_record[position_count]:
                x,attention_weight=self.common_conv_block[i](x,src_mask=mask,return_attention=return_attention)
                attention_list.append(attention_weight)
                position_count+=1
            elif i<=self.position_record[-1] and i==self.position_record[position_count]:
                x=self.common_conv_block[i](x,src_mask=mask)
                position_count+=1
            else:
                x=self.common_conv_block[i](x)
        
        if task=='E0':
            for i in range(len(self.E0_conv_block)):
                x=self.E0_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.E0_fc_block(x)
        elif task=='Ef':
            for i in range(len(self.Ef_conv_block)):
                x=self.Ef_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.Ef_fc_block(x)
        elif task=='stability':
            for i in range(len(self.stability_conv_block)):
                x=self.stability_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.stability_fc_block(x)

        if return_attention:
            return x,attention_list
        else:
            return x

class attention_CNN_quadtask(nn.Module):
    def __init__(self,is_embedding=True, channel_in=64, is_appendix=True,
                 is_space_embedding=True, space_embedding_len=150,
                 common_channels=[[88,40],[40,88],[40,88]],padding='same',
                 common_kernel_size=[[5,3],[5,3],[5,3]],stride=1,common_block_num=[1,1,0],
                 E0_channels=[[88,40],[40,88],[40,88]],E0_kernel_size=[[5,3],[5,3],[5,3]],
                 E0_block_num=[2,2,0],a_channels=[[88,40],[40,88],[40,88]],
                 a_kernel_size=[[5,3],[5,3],[5,3]],a_block_num=[2,2,0],
                 a_fc=[256,128,1],
                 Ef_channels=[[88,40],[40,88],[40,88]],
                 Ef_kernel_size=[[5,3],[5,3],[5,3]],Ef_block_num=[2,2,0],
                 stability_channels=[[88,40],[40,88],[40,88]],stability_kernel_size=[[5,3],[5,3],[5,3]],
                 stability_block_num=[1,2,0],
                 E0_fc=[512,128,1],Ef_fc=[512,128,1],stability_fc=[512,128,1],
                 use_resnet=True,norm_first=False,norm_activation=True,
                 use_channels=False,use_spatial=True,use_encoder=False,
                 num_heads=8,dim_feedforward=256,attention_num=[1,0,0],):
        super().__init__()
        if is_embedding:
            if is_appendix:
                self.embedding_len=channel_in-8
            else:
                self.embedding_len=channel_in-5
            self.embedding = nn.Embedding(100, self.embedding_len)

        self.is_appendix=is_appendix
        self.space_embedding = nn.Embedding(230, space_embedding_len)
        self.is_embedding=is_embedding
        self.is_space_embedding=is_space_embedding

        assert len(common_channels)==len(common_kernel_size)==len(common_block_num)==len(attention_num), 'The length of each list should be the same'
        common_channels=[[channel_in]]+common_channels if is_embedding else [[8]]+common_channels

        size=10
        self.common_conv_block=nn.ModuleList()
        for i in range(len(common_block_num)):
            for j in range(attention_num[i]):
                if use_encoder:
                    self.common_conv_block.append(TransformerEncoderLayer(d_model=sum(common_channels[i]), nhead=num_heads,\
                                            dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
                else:
                    self.common_conv_block.append(MultiheadAttention(sum(common_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(common_block_num[i]):
                if i==0 and j==0 and use_spatial:
                    self.common_conv_block.append(CBAM(sum(common_channels[i]),use_spatial=True))
                if j==0:
                    self.common_conv_block.append(multikernel_Convblock(sum(common_channels[i]),common_channels[i+1],\
                                                               common_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.common_conv_block.append(multikernel_Convblock(sum(common_channels[i+1]),common_channels[i+1],\
                                                               common_kernel_size[i],stride,padding,use_resnet))
        
        E0_channels=[common_channels[-1]]+E0_channels
        self.E0_conv_block=nn.ModuleList()
        for i in range(len(E0_block_num)):
            #for j in range(attention_num[i]):
            #    position_record.append(len(self.E0_conv_block))
            #    if use_encoder:
            #        self.E0_conv_block.append(TransformerEncoderLayer(d_model=sum(E0_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.E0_conv_block.append(MultiheadAttention(sum(E0_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(E0_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.E0_conv_block.append(CBAM(sum(E0_channels[i]),use_spatial=True))
                if j==0:
                    self.E0_conv_block.append(multikernel_Convblock(sum(E0_channels[i]),E0_channels[i+1],\
                                                               E0_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.E0_conv_block.append(multikernel_Convblock(sum(E0_channels[i+1]),E0_channels[i+1],\
                                                               E0_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.E0_conv_block.append(nn.LayerNorm([size,size]))
        self.E0_conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            length0=sum(E0_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(E0_channels[-1])*size**2
        self.E0_fc_block=FCN([length0]+E0_fc,end_with_activation=False)

        a_channels=[common_channels[-1]]+a_channels
        self.a_conv_block=nn.ModuleList()
        for i in range(len(a_block_num)):
            #for j in range(attention_num[i]):
            #    position_record.append(len(self.a_conv_block))
            #    if use_encoder:
            #        self.a_conv_block.append(TransformerEncoderLayer(d_model=sum(a_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.a_conv_block.append(MultiheadAttention(sum(a_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(a_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.a_conv_block.append(CBAM(sum(a_channels[i]),use_spatial=True))
                if j==0:
                    self.a_conv_block.append(multikernel_Convblock(sum(a_channels[i]),a_channels[i+1],\
                                                               a_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.a_conv_block.append(multikernel_Convblock(sum(a_channels[i+1]),a_channels[i+1],\
                                                               a_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.a_conv_block.append(nn.LayerNorm([size,size]))
        self.a_conv_block.append(nn.Flatten())

        if is_space_embedding:
            length0=sum(a_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(a_channels[-1])*size**2
        self.a_fc_block=FCN([length0]+a_fc,end_with_activation=True)

        Ef_channels=[common_channels[-1]]+Ef_channels
        self.Ef_conv_block=nn.ModuleList()
        for i in range(len(Ef_block_num)):
            #for j in range(attention_num[i]):
            #    if use_encoder:
            #        self.Ef_conv_block.append(TransformerEncoderLayer(d_model=sum(Ef_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.Ef_conv_block.append(MultiheadAttention(sum(Ef_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(Ef_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.Ef_conv_block.append(CBAM(sum(Ef_channels[i]),use_spatial=True))
                if j==0:
                    self.Ef_conv_block.append(multikernel_Convblock(sum(Ef_channels[i]),Ef_channels[i+1],\
                                                               Ef_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.Ef_conv_block.append(multikernel_Convblock(sum(Ef_channels[i+1]),Ef_channels[i+1],\
                                                               Ef_kernel_size[i],stride,padding,use_resnet))

        size=int(size)
        #self.position_record=position_record
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.Ef_conv_block.append(nn.LayerNorm([size,size]))
        self.Ef_conv_block.append(nn.Flatten())
        
        if is_space_embedding:
            length0=sum(Ef_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(Ef_channels[-1])*size**2
        self.Ef_fc_block=FCN([length0]+Ef_fc,end_with_activation=False)

        stability_channels=[common_channels[-1]]+stability_channels
        self.stability_conv_block=nn.ModuleList()
        for i in range(len(stability_block_num)):
            #for j in range(attention_num[i]):
            #    if use_encoder:
            #        self.stability_conv_block.append(TransformerEncoderLayer(d_model=sum(stability_channels[i]), nhead=num_heads,\
            #                                dim_feedforward=dim_feedforward, norm_first=norm_first, norm_activation=norm_activation))
            #    else:
            #        self.stability_conv_block.append(MultiheadAttention(sum(stability_channels[i]), num_heads, norm_activation=norm_activation))
            for j in range(stability_block_num[i]):
                #if i==0 and j==0 and use_spatial:
                #    self.stability_conv_block.append(CBAM(sum(stability_channels[i]),use_spatial=True))
                if j==0:
                    self.stability_conv_block.append(multikernel_Convblock(sum(stability_channels[i]),stability_channels[i+1],\
                                                               stability_kernel_size[i],stride,padding,use_resnet,invariant=False))
                else:
                    self.stability_conv_block.append(multikernel_Convblock(sum(stability_channels[i+1]),stability_channels[i+1],\
                                                               stability_kernel_size[i],stride,padding,use_resnet))
            
        size=int(size)
        #self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.stability_conv_block.append(nn.LayerNorm([size,size]))
        self.stability_conv_block.append(nn.Flatten())

        if is_space_embedding:
            length0=sum(stability_channels[-1])*size**2+space_embedding_len
        else:
            length0=sum(stability_channels[-1])*size**2
        self.stability_fc_block=FCN([length0]+stability_fc,end_with_activation=True)

    def forward(self, x, space_group=torch.tensor([216,225],device='cuda'),\
                return_attention=False,task='E0'):
        mask=torch.matmul(x[:,0,:,:].reshape(-1,100,1),x[:,0,:,:].reshape(-1,1,100))
        if self.is_embedding:
            x_other=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,100)
            mask=torch.matmul(x.reshape(-1,100,1),x.reshape(-1,1,100))
            site_arg=x!=0
            element_arg=torch.argwhere(site_arg)[:,1]
            embedding=torch.zeros(x.shape[0],100,self.embedding_len,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,2,1).reshape(-1,self.embedding_len,10,10)
            x=x.reshape(-1,1,10,10)
            if self.is_appendix:
                if x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                else:
                    raise ValueError('input x donnot contain correct appendix')
            else:
                if x_other.shape[1]==4:
                    x=torch.concatenate([x,embedding,x_other],dim=1)
                elif x_other.shape[1]==7:
                    x=torch.concatenate([x,embedding,x_other[:,-4:,:,:]],dim=1)
        
        attention_list=[]
        for i in range(len(self.common_conv_block)):
            if isinstance(self.common_conv_block[i],TransformerEncoderLayer) or isinstance(self.common_conv_block[i],MultiheadAttention):
                if return_attention:
                    x,attention_weight=self.common_conv_block[i](x,mask,return_attention=return_attention)
                    attention_list.append(attention_weight)
                else:
                    x=self.common_conv_block[i](x,mask)
            else:
                x=self.common_conv_block[i](x)
        
        if task=='E0':
            for i in range(len(self.E0_conv_block)):
                x=self.E0_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.E0_fc_block(x)
        elif task=='lattice_constant':
            for i in range(len(self.a_conv_block)):
                x=self.a_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.a_fc_block(x)
        elif task=='Ef':
            for i in range(len(self.Ef_conv_block)):
                x=self.Ef_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.Ef_fc_block(x)
        elif task=='stability':
            for i in range(len(self.stability_conv_block)):
                x=self.stability_conv_block[i](x)
            if self.is_space_embedding and space_group is not None:
                space_embedding=self.space_embedding(space_group)
                x=torch.concatenate([x,space_embedding],dim=1)
            x=self.stability_fc_block(x)

        if return_attention:
            return x,attention_list
        else:
            return x


class MTL_CNN(nn.Module):
    def __init__(self,is_embedding=True, embedding_len=64,
                 channels=100,kernel_size=[[5],[3,3],[3,3]],block_num=[[2],[4,5],[0,0]],
                 is_pool=[[False],[False,False],[False,False]],
                 fc=[[512,1024],[128,128],[1,1]],stride=1,):
        '''
        input label like kernel_size,block_num,is_pool,fc are a list of sublist, 
        in each sublist, the number corresponds to Tc,Eg,Ef respectively
        '''
        super(MTL_CNN,self).__init__()
        self.embedding = nn.Embedding(100, embedding_len-5)
        self.is_embedding=is_embedding
        self.embedding_len=embedding_len
        self.channels=channels
        self.block_num=block_num
        self.is_pool=is_pool
        self.space_embedding = nn.Embedding(230, 100)
        # used for embedding
        self.conv_pre=Convblock(in_channels=embedding_len, out_channels=channels, kernel_size=5, stride=1, padding=2)
        self.batchnorm = nn.BatchNorm2d(channels)
        
        # define the conv block
        self.conv_block_Mag=nn.ModuleList()
        self.conv_block_E=nn.ModuleList()
        for i in range(len(block_num)):
            assert len(block_num[i])==len(kernel_size[i])==len(is_pool[i]) and len(block_num[i]) in [1,2], 'the length of each sublist should be 1,2 or 3'
            block_tmp=nn.ModuleList()
            for _ in range(2):
                block_tmp.append(nn.ModuleList())
            for k in range(len(block_num[i])):
                padding=int((kernel_size[i][k]-1)/2)
                if (not is_embedding) and i==0:
                    block_tmp[k].append(Convblock(1,channels,kernel_size[i][k],stride,padding))
                else:
                    for _ in range(block_num[i][k]):
                        block_tmp[k].append(Convblock(channels,channels,kernel_size[i][k],stride,padding))
                if is_pool[i][k]:
                    block_tmp[k].append(nn.MaxPool2d(kernel_size=2, stride=2))
            # block_tmp shared by Tc, Eg and Ef
            if len(block_num[i])==1 and block_num[i][0]!=0:
                self.conv_block_Mag.append(block_tmp[0])
                self.conv_block_E.append(block_tmp[0])
            # block_tmp[0] shared by Tc and Eg, block_tmp[1] for Ef
            elif len(block_num[i])==2:
                if block_num[i][0]!=0:
                    self.conv_block_Mag.append(block_tmp[0])
                if block_num[i][1]!=0:
                    self.conv_block_E.append(block_tmp[1])

        
        self.conv_block_Mag.append(nn.Flatten())
        self.conv_block_E.append(nn.Flatten())

        # define the fc block
        Mag_list=[channels*5*5*4]
        E_list=[channels*5*5*4]
        for i in range(len(fc)):
            assert len(fc[i]) in [1,2], 'the length of each sublist should be 1,2 or 3'
            if len(fc[i])==1:
                Mag_list.append(fc[i][0])
            elif len(fc[i])==2:
                Mag_list.append(fc[i][0])
                E_list.append(fc[i][1])
        self.fc_block_Mag=FCN(Mag_list,end_with_activation=True)
        self.fc_block_E=FCN(E_list,end_with_activation=False)

    def forward(self, x, task='Mag',space_group=torch.tensor([216,225],device='cuda')):
        if self.is_embedding:
            appendix=x[:,1:,:,:]
            x=x[:,0,:,:].reshape(-1,1,10,10)
            site_arg=x!=0
            element_arg=torch.argwhere(x.reshape(-1,100)!=0)[:,1]
            embedding=torch.zeros(x.shape[0],x.shape[1],10,10,self.embedding_len-5,device='cuda')
            embedding[site_arg]=self.embedding(element_arg)
            embedding=embedding.permute(0,4,2,3,1).squeeze()
            space_embedding=self.space_embedding(space_group).reshape(-1,1,10,10)
            x=torch.concatenate([x,embedding,appendix,space_embedding],dim=1)
            x=self.conv_pre(x)
            #x=F.relu(self.batchnorm(x))
        
        assert task in ['Mag','E'], 'task should be Mag or E'
        if task=='Mag':
            for i in range(len(self.conv_block_Mag)):
                if isinstance(self.conv_block_Mag[i],nn.ModuleList):
                    for j in range(len(self.conv_block_Mag[i])):
                        x=self.conv_block_Mag[i][j](x)
                else:
                    x=self.conv_block_Mag[i](x)
            x=self.fc_block_Mag(x)
            return x
        elif task=='E':
            for i in range(len(self.conv_block_E)):
                if isinstance(self.conv_block_E[i],nn.ModuleList):
                    for j in range(len(self.conv_block_E[i])):
                        x=self.conv_block_E[i][j](x)
                else:
                    x=self.conv_block_E[i](x)
            x=self.fc_block_E(x)
            return x


class adjacent_CNN(nn.Module):
    def __init__(self,channels=[64,100,150,200],padding=[6,5,3,1],
                 kernel_size=[11,11,7,3],stride=1,block_num=[1,3,3,3],pool_size=[1,1,1,1],
                 is_pool=[False,False,False,False],fc=[1000,128,1],
                 contain_near_numbers=5,elementlist=np.zeros(100)):
        super(adjacent_CNN,self).__init__()
        self.channels=channels
        self.block_num=block_num
        self.is_pool=is_pool
        self.conv_block=nn.ModuleList()
        channels=[contain_near_numbers*2]+channels
        size=len(elementlist)
        assert len(channels)-1==len(padding)==len(kernel_size)==len(block_num)\
            ==len(is_pool)==len(pool_size), 'The length of each list should be the same'
        for i in range(len(block_num)):
            for j in range(block_num[i]):
                if j==0:
                    self.conv_block.append(Convblock(channels[i],channels[i+1],kernel_size[i],stride,padding[i]))
                else:
                    self.conv_block.append(Convblock(channels[i+1],channels[i+1],kernel_size[i],stride,padding[i]))
                size=(size-kernel_size[i]+2*padding[i])/stride+1
            if is_pool[i]:
                self.conv_block.append(nn.MaxPool2d(kernel_size=pool_size[i], stride=pool_size[i]))
                size=size/pool_size[i]
        self.conv_block.append(nn.BatchNorm2d(channels[-1]))
        self.conv_block.append(nn.Flatten())
        self.fc_block=FCN([int(size)**2*channels[-1]]+fc,end_with_activation=True)

    def forward(self, x, space_group=None,energy=None):
        for i in range(len(self.conv_block)):
            x=self.conv_block[i](x)
        
        x=self.fc_block(x)
        return x


class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, \
                 stride, padding,end_with_activation=True,\
                    normal_order=True, end_with_norm=True):
        super(Convblock, self).__init__()
        self.conv_layer=nn.ModuleList()
        if normal_order:
            #self.conv_layer.append(nn.LayerNorm([10,10]))
            self.conv_layer.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels, \
                                                kernel_size=kernel_size, stride=stride, \
                                                    padding=padding))
            #self.conv_layer.append(nn.BatchNorm2d(out_channels))
            if end_with_norm:
                self.conv_layer.append(nn.LayerNorm([10,10]))
            #self.conv_layer.append(nn.InstanceNorm2d(out_channels,affine=True))
            if end_with_activation:
                self.conv_layer.append(nn.LeakyReLU())
            #self.conv_layer.append(nn.LayerNorm([10,10]))
        else:
            self.conv_layer.append(nn.LayerNorm([10,10]))
            self.conv_layer.append(nn.LeakyReLU())
            self.conv_layer.append(nn.Conv2d(in_channels=in_channels,out_channels=out_channels, \
                                                kernel_size=kernel_size, stride=stride, \
                                                    padding=padding))

    def forward(self, x):
        for i in range(len(self.conv_layer)):
            x=self.conv_layer[i](x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            #nn.BatchNorm1d(in_channels // reduction),
            nn.LeakyReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            #nn.BatchNorm1d(in_channels),
        )

    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        max_out = self.max_pool(x).view(x.size(0), -1)
        channel_att = (self.fc(avg_out) + self.fc(max_out)).view(x.size(0), x.size(1), 1, 1)
        return x * channel_att.sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding="same")
        self.instance_norm = nn.InstanceNorm2d(1)

    def forward(self, x):
        max_out, _ = x.max(dim=1, keepdim=True)
        avg_out = x.mean(dim=1, keepdim=True)
        spatial_att = torch.cat([max_out, avg_out], dim=1)
        spatial_att = self.conv(spatial_att)
        #spatial_att = self.instance_norm(spatial_att)
        return x * spatial_att.sigmoid()


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=2,use_channel=False,use_spatial=False):
        super(CBAM, self).__init__()
        self.use_channel=use_channel
        self.use_spatial=use_spatial
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        if self.use_channel:
            x = self.channel_att(x)
        if self.use_spatial:
            x = self.spatial_att(x)
        return x


class multikernel(nn.Module):
    def __init__(self, channel_in:int, attention_channel_out:list,kernel_size:list,\
                 stride=1,padding='same',norm_all=True,end_with_activation=True):
        super().__init__()
        self.conv_layer=nn.ModuleList()
        self.norm_all=norm_all
        self.end_with_activation=end_with_activation
        assert len(attention_channel_out)==len(kernel_size), 'The length of attention_channel_out should be the same as kernel_size'
        for i in range(len(attention_channel_out)):
            if attention_channel_out[i]==0:
                continue
            if norm_all:
                self.conv_layer.append(Convblock(in_channels=channel_in,out_channels=attention_channel_out[i], \
                                                kernel_size=kernel_size[i], stride=stride, \
                                                padding=padding, end_with_activation=False,end_with_norm=False))
            else:
                self.conv_layer.append(Convblock(in_channels=channel_in,out_channels=attention_channel_out[i], \
                                                kernel_size=kernel_size[i], stride=stride, \
                                                padding=padding))
                
    def forward(self, x):
        x=torch.concatenate([layer(x) for layer in self.conv_layer],dim=1)
        if self.norm_all:
            x=nn.LayerNorm([10,10],device='cuda')(x)
        if self.end_with_activation:
            x=nn.LeakyReLU()(x)
        return x


class multikernel_Convblock(nn.Module):
    def __init__(self, channel_in:int, attention_channel_out:list,kernel_size:list,\
                 stride=1,padding='same',use_resnet=False, invariant=True):
        super().__init__()
        self.conv_layer=nn.ModuleList()
        self.use_resnet=use_resnet
        for i in range(len(attention_channel_out)):
            if attention_channel_out[i]==0:
                continue
            self.conv_layer.append(nn.Sequential(
                Convblock(in_channels=channel_in,out_channels=attention_channel_out[i], \
                                                kernel_size=kernel_size[i], stride=stride, \
                                                    padding=padding),
                Convblock(in_channels=attention_channel_out[i],out_channels=attention_channel_out[i], \
                                                 kernel_size=kernel_size[i], stride=stride, \
                                                     padding=padding,end_with_activation=False)
            ))
        if channel_in==sum(attention_channel_out) and invariant:
            self.shortcut=nn.Identity()
        else:
            self.shortcut=Convblock(in_channels=channel_in,out_channels=sum(attention_channel_out), \
                                                kernel_size=1, stride=stride, \
                                                    padding=0,end_with_activation=False)
        
    def forward(self, x):
        identity = x
        if self.use_resnet:
            x=torch.concatenate([layer(x) for layer in self.conv_layer],dim=1)
            # this has been changed
            x+=self.shortcut(identity)
            x= nn.LeakyReLU()(x)
            #x+=self.shortcut(identity)
        else:
            x=torch.concatenate([layer(x) for layer in self.conv_layer],dim=1)
        return x


class multikernel_Convblock_norm_all(nn.Module):
    def __init__(self, channel_in:int, attention_channel_out:list,kernel_size:list,\
                 stride=1,padding='same',use_resnet=False, invariant=True):
        super().__init__()
        self.use_resnet=use_resnet
        self.conv_layer=nn.Sequential(
            multikernel(channel_in,attention_channel_out,kernel_size,stride,padding,norm_all=True),
            multikernel(sum(attention_channel_out),attention_channel_out,kernel_size,\
                        stride,padding,norm_all=True,end_with_activation=False),
        )
        if channel_in==sum(attention_channel_out) and invariant:
            self.shortcut=nn.Identity()
        else:
            self.shortcut=Convblock(in_channels=channel_in,out_channels=sum(attention_channel_out), \
                                                kernel_size=1, stride=stride, \
                                                    padding=0,end_with_activation=False,end_with_norm=False)
        
    def forward(self, x):
        identity = x
        if self.use_resnet:
            x=self.conv_layer(x)
            x+=self.shortcut(identity)
            x= nn.LeakyReLU()(x)
        else:
            x=torch.concatenate([layer(x) for layer in self.conv_layer],dim=1)
        return x


class resnet_simplify(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = Convblock(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        x += identity
        return x

class resnet(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = Convblock(in_channels=channels_in, out_channels=channels_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = Convblock(in_channels=channels_out, out_channels=channels_out, kernel_size=kernel_size, stride=stride, padding=padding,end_with_activation=False)
        if channels_out==channels_in:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Convblock(in_channels=channels_in, out_channels=channels_out, kernel_size=1, stride=stride, padding=0,end_with_activation=False)
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(identity)
        x= nn.LeakyReLU()(x)
        return x


class preactivate_resnet(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = Convblock(in_channels=channels, out_channels=channels, kernel_size=kernel_size,\
                                stride=stride, padding=padding,normal_order=False)
        self.conv2 = Convblock(in_channels=channels, out_channels=channels, kernel_size=kernel_size,\
                                stride=stride, padding=padding,normal_order=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += identity
        return x

# a fully connected network
class FCN(nn.Module):
    def __init__(self, nodes_num:list, dropout=0.2,end_with_drop=False, end_with_activation=False):
        super(FCN, self).__init__()
        assert len(nodes_num)>=2, 'The length of nodes_num should be larger than 2'
        self.fcmodule=nn.ModuleList()
        for i in range(1,len(nodes_num)):
            self.fcmodule.append(nn.Linear(in_features=nodes_num[i-1], out_features=nodes_num[i]))
            #self.fcmodule.append(nn.BatchNorm1d(nodes_num[i]))
            if i!=len(nodes_num)-1:
                self.fcmodule.append(nn.BatchNorm1d(nodes_num[i]))
                #self.fcmodule.append(nn.LayerNorm(nodes_num[i]))
            #if i!=len(nodes_num)-1 or end_with_drop:
            #    self.fcmodule.append(nn.Dropout(dropout))
            if i!=len(nodes_num)-1 or end_with_activation:
                # this has been changed
                self.fcmodule.append(nn.LeakyReLU())
                #self.fcmodule.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.fcmodule)):
            x=self.fcmodule[i](x)
        return x


#def position_encoding_init(n_position, emb_dim, base=10000, device='cuda'):
#    ''' Init the sinusoid position encoding table '''
#    # keep dim 0 for padding token position encoding zero vector
#    position_enc = np.array([
#        [pos / np.power(base, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
#        if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
#    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
#    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
#    return torch.from_numpy(position_enc).type(torch.FloatTensor).to(device)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, emb_dim, device='cuda'):
        super().__init__()
        self.base=Parameter(torch.tensor(100.0,device=device),requires_grad=True)
        self.max_len=max_len
        self.emb_dim=emb_dim

    def forward(self, input: torch.Tensor,device='cuda'):
        position_enc=torch.arange(self.max_len,device=device).reshape(-1,1)/torch.pow(self.base,torch.arange(0,self.emb_dim,step=2,device=device).reshape(1,-1)/self.emb_dim)
        position_sin=torch.sin(position_enc)
        position_cos=torch.cos(position_enc)
        position=torch.cat([position_sin.reshape(-1,1),position_cos.reshape(-1,1)],dim=1).reshape(self.max_len,self.emb_dim)
        return position[input]


def calculate_result(data,model,task='Mag'):
    if task=='Mag':
        return model(data[0],data[2],(data[3])**3/(2**0.5),task=task)
    else:
        return model(data[0],data[2],task=task)


class Train_Test_Plot():
    def __init__(self,model,dataset_list:List,task_list:List,loss_Fn,optimizer,\
                 model_type='cnn',device='cuda',plot_num=100,split_ratio:Union[List,float]=0.8,\
                    batch_size_train=128,batch_size_test=256):
        self.model=model
        self.dataset_list=dataset_list
        self.task_list=[sub_task_list if isinstance(sub_task_list,list)\
                         else [sub_task_list] for sub_task_list in task_list]
        assert len(dataset_list)==len(task_list), 'The length of dataset_list should be the same as task_list'
        
        self.dataloader=[]
        for i in range(len(dataset_list)):
            dataloader_tmp=[]
            if isinstance(split_ratio,float) or len(split_ratio)==2:
                train_mask,test_mask=random_split(len(dataset_list[i]),split_ratio)
                train_dataset,test_dataset=dataset_list[i][train_mask],dataset_list[i][test_mask]
                if i==0:
                    batch_size=batch_size_train
                    self.batch_num=math.ceil(len(train_dataset[0])/batch_size)
                else:
                    batch_size=math.ceil(len(train_dataset[0])/self.batch_num)
                dataloader_tmp.append(DataLoader(TensorDataset(*train_dataset),batch_size=batch_size,shuffle=True))
                dataloader_tmp.append(DataLoader(TensorDataset(*test_dataset),batch_size=batch_size_test,shuffle=False))
            elif len(split_ratio)==3:
                train_mask,val_mask,test_mask=random_split(len(dataset_list[i]),split_ratio)
                train_dataset,val_dataset,test_dataset=dataset_list[i][train_mask],dataset_list[i][val_mask],dataset_list[i][test_mask]
                if i==0:
                    batch_size=batch_size_train
                    self.batch_num=math.ceil(len(train_dataset[0])/batch_size)
                else:
                    batch_size=math.ceil(len(train_dataset[0])/self.batch_num)
                dataloader_tmp.append(DataLoader(TensorDataset(*train_dataset),batch_size=batch_size,shuffle=True))
                dataloader_tmp.append(DataLoader(TensorDataset(*val_dataset),batch_size=batch_size_test,shuffle=False))
                dataloader_tmp.append(DataLoader(TensorDataset(*test_dataset),batch_size=batch_size_test,shuffle=False))
            self.dataloader.append(dataloader_tmp)

        self.task_flatten=[]
        for task in task_list:
            if isinstance(task,list):
                self.task_flatten+=task
            else:
                self.task_flatten.append(task)
        self.loss_Fn=loss_Fn
        self.optimizer=optimizer
        self.model_type=model_type
        self.device=device
        self.plot_num=plot_num

    def trainer(self,epoch_max=1000,patience=50):
        if epoch_max==0:
            self.model.load_state_dict(torch.load('result/%s_model.pth'%('_'.join(self.task_flatten))))
            print('Load model from result/%s_model.pth'%('_'.join(self.task_flatten)))
            loss_test=self.test() if len(self.dataloader[0])==2 else self.true_test()
            print('Test loss: {}'.format([round(i,5) for i in loss_test]))

        if not os.path.exists('result'):
            os.makedirs('result')
        loss_all_epoch=[]
        epoch_num=0
        patience_num=0
        contrast_loss=[np.inf]
        min_loss=[np.inf]*len(self.task_flatten)
        if epoch_max>0:
            print('the following loss of each task is arrange by train loss, test loss, contrast loss, min loss')
        while epoch_num<epoch_max:
            loss_train=self.train()
            loss_test=self.test()
            loss_all_epoch.append([loss_train,loss_test])
            if np.sum(loss_test)>=np.sum(contrast_loss):
                patience_num+=1
                if patience_num>=patience:
                    print('Early stop')
                    break
            else:
                patience_num=0
                contrast_loss=loss_test
                torch.save(self.model.state_dict(), 'result/%s_model.pth'%('_'.join(self.task_flatten)))
                print('Save model to result/%s_model.pth'%('_'.join(self.task_flatten)))
            epoch_num+=1
            min_loss=[min(min_loss[i],loss_test[i]) for i in range(len(self.task_flatten))]
            print("Epoch:{},Task:".format(epoch_num),end=' ')
            for i in range(len(self.task_flatten)):
                print('{0}:[{1},{2},{3},{4}]'.format(self.task_flatten[i],\
                    *[round(sub_loss[i],5) for sub_loss in [loss_train,loss_test,contrast_loss,min_loss]]),\
                        end=',')
            print()

    def train(self):
        self.model.train()
        loss_all = [0]*len(self.task_flatten)
        count_number=[0]*len(self.task_flatten)
        if len(self.dataset_list)==1:
            #loss_all[0]=train(self.dataloader[0][0],self.model,self.loss_Fn,self.optimizer)
            for data in self.dataloader[0][0]:
                loss=0
                for i,task in enumerate(self.task_flatten):
                    output=calculate_result(data,self.model,task)
                    if task=="lattice_constant" or task=="stability":
                        label=data[3]
                    elif task=="E0":
                        label=data[4]
                    else:
                        label=data[1]
                    loss_tmp=self.loss_Fn(output,label)
                    loss_all[i]+=loss_tmp.item()*len(data[0])
                    loss+=loss_tmp
                    count_number[i]+=len(data[0])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        elif len(self.dataset_list)==2:
            fixed_length_trainloader=[FixedLengthLoader(dataloader[0],self.batch_num) for dataloader in self.dataloader]
            for data1,data2 in list(zip(fixed_length_trainloader[0],fixed_length_trainloader[1])):
                loss=0
                loss_index=0
                for i,task in enumerate(self.task_list[0]):
                    output=calculate_result(data1,self.model,task)
                    if task=="lattice_constant" or task=="stability":
                        label=data1[3]
                    elif task=="E0":
                        label=data1[4]
                    else:
                        label=data1[1]
                    loss_tmp=self.loss_Fn(output,label)
                    loss_all[loss_index]+=loss_tmp.item()*len(data1[0])
                    loss+=loss_tmp
                    count_number[loss_index]+=len(data1[0])
                    loss_index+=1
                for i,task in enumerate(self.task_list[1]):
                    output=calculate_result(data2,self.model,task)
                    if task=="lattice_constant" or task=="stability":
                        label=data2[3]
                    elif task=="E0":
                        label=data2[4]
                    else:
                        label=data2[1]
                    loss_tmp=self.loss_Fn(output,label)
                    loss_all[loss_index]+=loss_tmp.item()*len(data2[0])
                    loss+=loss_tmp
                    count_number[loss_index]+=len(data2[0])
                    loss_index+=1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        #loss_all=torch.tensor(loss_all)
        for i in range(len(self.task_flatten)):
            loss_all[i]/=count_number[i]
        return loss_all

    def test(self):
        self.model.eval()
        with torch.no_grad():
            loss_all = [0]*len(self.task_flatten)
            if len(self.dataset_list)==1:
                for data in self.dataloader[0][1]:
                    for i,task in enumerate(self.task_flatten):
                        output=calculate_result(data,self.model,task)
                        if task=="lattice_constant" or task=="stability":
                            label=data[3]
                        elif task=="E0":
                            label=data[4]
                        else:
                            label=data[1]
                        loss=self.loss_Fn(output,label)
                        loss_all[i]+=loss.item()*len(data[0])
            elif len(self.dataset_list)==2:
                loss_index=0
                for i,task in enumerate(self.task_list[0]):
                    for data1 in self.dataloader[0][1]:
                        output=calculate_result(data1,self.model,task)
                        if task=="lattice_constant" or task=="stability":
                            label=data1[3]
                        elif task=="E0":
                            label=data1[4]
                        else:
                            label=data1[1]
                        loss=self.loss_Fn(output,label)
                        loss_all[loss_index]+=loss.item()*len(data1[0])
                    loss_index+=1
                for i,task in enumerate(self.task_list[1]):
                    for data2 in self.dataloader[1][1]:
                        output=calculate_result(data2,self.model,task)
                        if task=="lattice_constant" or task=="stability":
                            label=data2[3]
                        elif task=="E0":
                            label=data2[4]
                        else:
                            label=data2[1]
                        loss=self.loss_Fn(output,label)
                        loss_all[loss_index]+=loss.item()*len(data2[0])
                    loss_index+=1

            #loss_all=torch.tensor(loss_all)
            loss_index=0
            for i in range(len(self.task_list)):
                for j in range(len(self.task_list[i])):
                    loss_all[loss_index]/=len(self.dataloader[i][1].dataset)
                    loss_index+=1
        return loss_all
    
    def true_test(self):
        self.model.eval()
        with torch.no_grad():
            loss_all = [0]*len(self.task_flatten)
            if len(self.dataset_list)==1:
                for data in self.dataloader[0][2]:
                    for i,task in enumerate(self.task_flatten):
                        output=calculate_result(data,self.model,task)
                        if task=="lattice_constant" or task=="stability":
                            label=data[3]
                        elif task=="E0":
                            label=data[4]
                        else:
                            label=data[1]
                        loss=self.loss_Fn(output,label)
                        loss_all[i]+=loss.item()*len(data[0])
            elif len(self.dataset_list)==2:
                loss_index=0
                for i,task in enumerate(self.task_list[0]):
                    for data1 in self.dataloader[0][2]:
                        output=calculate_result(data1,self.model,task)
                        if task=="lattice_constant" or task=="stability":
                            label=data1[3]
                        elif task=="E0":
                            label=data1[4]
                        else:
                            label=data1[1]
                        loss=self.loss_Fn(output,label)
                        loss_all[loss_index]+=loss.item()*len(data1[0])
                    loss_index+=1
                for i,task in enumerate(self.task_list[1]):
                    for data2 in self.dataloader[1][2]:
                        output=calculate_result(data2,self.model,task)
                        if task=="lattice_constant" or task=="stability":
                            label=data2[3]
                        elif task=="E0":
                            label=data2[4]
                        else:
                            label=data2[1]
                        loss=self.loss_Fn(output,label)
                        loss_all[loss_index]+=loss.item()*len(data2[0])
                    loss_index+=1

            #loss_all=torch.tensor(loss_all)
            loss_index=0
            for i in range(len(self.task_list)):
                for j in range(len(self.task_list[i])):
                    loss_all[loss_index]/=len(self.dataloader[i][2].dataset)
                    loss_index+=1
        return loss_all


    def plot_result(self,delta=0.4,scatter_size=3):
        self.model.load_state_dict(torch.load('result/%s_model.pth'%('_'.join(self.task_flatten))))
        self.model.eval()
        plt.rcParams['axes.titlesize']= plt.rcParams['axes.labelsize']
        plot_indice=0
        for i in range(len(self.dataset_list)):
            for j,task in enumerate(self.task_list[i]):
                fig, axs = plt.subplots(1, 1, figsize=(3.5, 3.5))
                #plt.rcParams['font.family'] = 'serif'
                #plt.rcParams['font.serif'] = ['Times New Roman']
                fig.set_tight_layout(True)
                plot_label=['Train','Test'] if len(self.dataloader[i])==2 else ['Train','Validation','Test']
                color=['b','r'] if len(self.dataloader[i])==2 else ['#43A3EF','#EF767B','#A5B55D']
                for k,dataloader in enumerate(self.dataloader[i]):
                    label=[]
                    predict=[]
                    with torch.no_grad():
                        for data in dataloader:
                            output=calculate_result(data,self.model,task)
                            predict.append(output.cpu().detach().numpy())
                            if task=="lattice_constant" or task=="stability":
                                label.append(data[3].cpu().detach().numpy())
                            elif task=="E0":
                                label.append(data[4].cpu().detach().numpy())
                            else:
                                label.append(data[1].cpu().detach().numpy())
                    predict=np.concatenate(predict,axis=0)
                    label=np.concatenate(label,axis=0)
                    random_indices=np.array(random.sample(range(len(label)),self.plot_num))
                    label_plot,predict_plot=label[random_indices],predict[random_indices]
                    axs.scatter(np.array(label_plot),np.array(predict_plot),\
                                c=color[k],s=scatter_size)
                    axs.plot([],[],c=color[k],label=r'$\mathrm{%s}$'%(plot_label[k]),linewidth=7)
                    #if k==1:
                    #    axs.text(0.04,0.50,"RMSE:%0.4f\nMAE:%0.4f\nMSE:%0.4f"%(np.mean((label-predict)**2)**0.5,np.mean(np.abs(label-predict)),np.mean((label-predict)**2)),\
                                 #c=color[k],transform=axs.transAxes,horizontalalignment='left',fontsize=plt.rcParams['axes.labelsize'])
                    #elif k==2:
                    #    axs.text(0.5,0.15,"RMSE:%0.4f\nMAE:%0.4f\nMSE:%0.4f"%(np.mean((label-predict)**2)**0.5,np.mean(np.abs(label-predict)),np.mean((label-predict)**2)),\
                                 #c=color[k],transform=axs.transAxes,horizontalalignment='left',fontsize=plt.rcParams['axes.labelsize'])
                min_edge=min(np.min(label_plot),np.min(predict_plot))
                max_edge=max(np.max(label_plot),np.max(predict_plot))
                if abs(min_edge)<0.1 and min_edge/max_edge<delta:
                    min_edge=0
                    max_edge=max_edge+0.15*(max_edge-min_edge)
                elif abs(max_edge)<0.1 and max_edge/min_edge<delta:
                    max_edge=0
                    min_edge=min_edge-0.15*(max_edge-min_edge)
                else:
                    min_edge=min_edge-0.15*(max_edge-min_edge)
                    max_edge=max_edge+0.15*(max_edge-min_edge)
                #if len(self.task_flatten)==1:
                
                if task=="lattice_constant":
                    axs.set_xlabel(r'$\mathrm{True\ lattice\ constant\ (\AA)}$')
                    axs.set_ylabel(r'$\mathrm{Predicted\ lattice\ constant\ (\AA)}$')
                elif task=="E0":
                    axs.set_xlabel(r'$\mathrm{True\ E_0\ (eV/atom)}$')
                    axs.set_ylabel(r'$\mathrm{Predicted\ E_0\ (eV/atom)}$')
                elif task=="Ef":
                    axs.set_xlabel(r'$\mathrm{True\ E_f\ (eV/atom)}$')
                    axs.set_ylabel(r'$\mathrm{Predicted\ E_f\ (eV/atom)}$')
                elif task=="Mag":
                    axs.set_xlabel(r'$\mathrm{True\ m_s\ (\mu_B/atom)}$')
                    axs.set_ylabel(r'$\mathrm{Predicted\ m_s\ (\mu_B/atom)}$')
                elif task=="stability":
                    axs.set_xlabel(r'$\mathrm{True\ E_{hull}\ (eV/atom)}$')
                    axs.set_ylabel(r'$\mathrm{Predicted\ E_{hull}\ (eV/atom)}$')
                R2=r2_score(label,predict)
                axs.set_title(r'$R^2=%0.4f$'%(R2))
                axs.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置X轴主刻度线数量为6
                axs.yaxis.set_major_locator(MaxNLocator(nbins=6))
                axs.xaxis.set_minor_locator(AutoMinorLocator(2))
                axs.yaxis.set_minor_locator(AutoMinorLocator(2))
                axs.tick_params(axis='both', direction='in')
                axs.tick_params(axis='both', which='minor', direction='in')
                axs.set_xlim([min_edge,max_edge])
                axs.set_ylim([min_edge,max_edge])
                axs.set_aspect('equal')
                axs.legend(frameon=False,loc="upper left")
                    #else:
                    #    axs[plot_indice].scatter(np.array(label_plot),np.array(predict_plot),\
                    #                             c=color[k],label=r'$\mathrm{%s}$'%(plot_label[k]),s=scatter_size)
                    #    axs[plot_indice].plot([min_edge,max_edge],[min_edge,max_edge], 'k--', linewidth=plt.rcParams['axes.linewidth'])
                    #    if task=="lattice_constant":
                    #        axs[plot_indice].set_xlabel(r'$\mathrm{True\ lattice\ constant\ (\AA)}$')
                    #        axs[plot_indice].set_ylabel(r'$\mathrm{Predicted\ lattice\ constant\ (\AA)}$')
                    #    elif task=="E0":
                    #        axs[plot_indice].set_xlabel(r'$\mathrm{True\ E_0\ (eV/atom)}$')
                    #        axs[plot_indice].set_ylabel(r'$\mathrm{Predicted\ E_0\ (eV/atom)}$')
                    #    elif task=="Ef":
                    #        axs[plot_indice].set_xlabel(r'$\mathrm{True\ E_f\ (eV/atom)}$')
                    #        axs[plot_indice].set_ylabel(r'$\mathrm{Predicted\ E_f\ (eV/atom)}$')
                    #    elif task=="Mag":
                    #        axs[plot_indice].set_xlabel(r'$\mathrm{True\ m_s\ (\mu B/atom)}$')
                    #        axs[plot_indice].set_ylabel(r'$\mathrm{Predicted\ m_s\ (\mu B/atom)}$')
                    #    elif task=="stability":
                    #        axs[plot_indice].set_xlabel(r'$\mathrm{True\ %s\ (eV/atom)}$'%(task))
                    #        axs[plot_indice].set_ylabel(r'$\mathrm{Predicted\ %s\ (eV/atom)}$'%(task))
                    #    axs[plot_indice].set_title(r'$R^2=%0.4f$'%(R2))
                    #    axs[plot_indice].tick_params(axis='both', direction='in')
                    #    axs[plot_indice].xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置X轴主刻度线数量为5
                    #    axs[plot_indice].yaxis.set_major_locator(MaxNLocator(nbins=6))
                    #    axs[plot_indice].xaxis.set_minor_locator(AutoMinorLocator(2))
                    #    axs[plot_indice].yaxis.set_minor_locator(AutoMinorLocator(2))
                    #    axs[plot_indice].tick_params(axis='both', which='minor', direction='in')
                    #    axs[plot_indice].set_xlim([min_edge,max_edge])
                    #    axs[plot_indice].set_ylim([min_edge,max_edge])
                    #    axs[plot_indice].set_aspect('equal')
                    #    axs[plot_indice].legend(frameon=False)
                plt.tight_layout()
                axs.plot([min_edge,max_edge],[min_edge,max_edge], 'k--', linewidth=plt.rcParams['axes.linewidth'],zorder=0)
                plt.savefig('result/R2_{}.jpg'.format(task),dpi=2400)
                plt.show()
                plot_indice+=1

    def count_more50(self,return_count=False):
        with torch.no_grad():
            count_all=torch.zeros(100)
            for data in self.dataloader[0][0]:
                count_all+=torch.count_nonzero(data[0][:,0,:,:].reshape(-1,100),dim=0).to('cpu')
            
            index=torch.argwhere(count_all>50).reshape(-1)
            if return_count:
                return index,count_all
            else:
                return index
        
    def plot_pca(self):
        index=self.count_more50()
        color=['r','k','b','y']
        tag=['s','p','d','f']
        elements=ChemicalSymbols[index.numpy()]

        upperright_element=['Sr','Ca','Ba','Sc','Hf','Sc','Hf','Cu','Co','Os','W','La']
        upperleft_element=['Ag','Re','K','Hg']
        lowerright_element=['Y','Zr','Fe','Mo','Tc','Ni','Au','S','N','Pd']

        para_dict=torch.load("result/%s_model.pth"%('_'.join(self.task_flatten)))
        embedding=para_dict['embedding.weight'].cpu().detach().numpy()
        X=embedding[index.numpy()]
        pca=PCA(n_components=2)
        pca_result=pca.fit_transform(X)
        explaine=pca.explained_variance_ratio_

        c=[]
        for i in range(len(outer_e)):
            if outer_e[i]=='s':
                c.append(color[0])
            elif outer_e[i]=='p':
                c.append(color[1])
            elif outer_e[i]=='d':
                c.append(color[2])
            elif outer_e[i]=='f':
                c.append(color[3])

        fig=plt.figure(figsize=(4.5,4.5))
        bax = brokenaxes(ylims=((-1.6, 1.3), (2.7, 3.5)), hspace=0.15,despine=False,)
        for i,symbol in enumerate(elements):
            bax.scatter(pca_result[i,0],pca_result[i,1],\
                        c=c[np.argwhere(ChemicalSymbols==symbol)[0][0]],s=(element(symbol).atomic_radius/30)**2)

        for j in range(len(tag)):
            bax.scatter([], [], c=color[j], label=tag[j],s=10)

        bax.legend(frameon=False,loc='upper right')

        #plt.legend()
        #plt.scatter(pca_result[:,0],pca_result[:,1],c=np.array(c4colorbar[:100])[index.numpy()>0],cmap='viridis')
        #plt.colorbar()
        #for j in range(len(set(c4colorbar))):
        #    args=outer_e1==tag[j]
        #    plt.scatter(pca_result[args, 0], pca_result[args, 1], c=color[j], label=tag[j])
        for i in range(len(pca_result)):
            if elements[i] in upperright_element:
                bax.text(pca_result[i,0]+0.02,pca_result[i,1]+0.02,elements[i],fontsize=12)
            elif elements[i] in upperleft_element:
                bax.text(pca_result[i,0]-0.1,pca_result[i,1]+0.07,elements[i],fontsize=12)
            elif elements[i] in lowerright_element:
                bax.text(pca_result[i,0]+0.05,pca_result[i,1]-0.12,elements[i],fontsize=12)

        #bax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置X轴主刻度线数量为6
        #bax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        #bax.xaxis.set_minor_locator(AutoMinorLocator(2))
        #bax.yaxis.set_minor_locator(AutoMinorLocator(2))
        #bax.tick_params(axis='both', direction='in')
        #bax.tick_params(axis='both', which='minor', direction='in')
        for axi,ax in enumerate(bax.axs):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # 设置X轴主刻度线数量为6
            ax.yaxis.set_major_locator(MaxNLocator(nbins=(6 if axi==1 else 2)))
            ax.tick_params(axis='both', direction='in')
            if axi==1:
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis='both', which='minor', direction='in')
        bax.set_xlabel('PC1 (%.2f%%)'%(explaine[0]*100))
        bax.set_ylabel('PC2 (%.2f%%)'%(explaine[1]*100))
        #plt.tight_layout()
        #bax.set_aspect('equal')
        plt.savefig('result/PCA.jpg',dpi=2400)
        plt.show()

    def plot_attention(self):
        element_count=self.count_more50(return_count=True)[1]
        attention_all=torch.zeros(100,100)
        for data in self.dataloader[0][0]:
            attention_all+=torch.sum(self.get_attention(data,self.model)[0].detach().cpu(),dim=(0,1))
        
        total_matrix=attention_all.detach()
        for i in range(100):
            for j in range(100):
                if element_count[i]>=50 and element_count[j]>=50:
                    total_matrix[i,j]=total_matrix[i,j]/(element_count[i]*element_count[j])**0.5
                else:
                    total_matrix[i,j]=0

        fig, ax = plt.subplots(figsize=(4,6))
        cax=ax.matshow((total_matrix/torch.max(total_matrix)).numpy())
        fig.colorbar(cax,location='bottom',pad=0.05)
        ax.set_xticks(np.arange(0,100,10))
        ax.set_yticks(np.arange(0,100,10))
        labels=[ChemicalSymbols[i] for i in range(0,100,10)]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("K")
        ax.set_ylabel("Q")
        ax.xaxis.set_label_position('top')
        plt.savefig('result/attention_all.jpg',dpi=2400)
        plt.show()

        fig, ax = plt.subplots(figsize=(4,6))
        min=19
        max=34
        cax=ax.matshow((total_matrix[min:max,min:max]/torch.max(total_matrix[min:max,min:max])).detach().cpu().numpy())
        cbar=fig.colorbar(cax,location='bottom',pad=0.05)
        cax.set_clim(0,1)
        ax.set_xticks(range(max-min))
        ax.set_yticks(range(max-min))
        labels=ChemicalSymbols[min:max]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("K")
        ax.set_ylabel("Q")
        ax.xaxis.set_label_position('top')
        plt.savefig('result/attention%d_%d.jpg'%(min,max),dpi=2400)
        plt.show()

        fig, ax = plt.subplots(figsize=(4,6))
        min=38
        max=52
        cax=ax.matshow((total_matrix[min:max,min:max]/torch.max(total_matrix[min:max,min:max])).detach().cpu().numpy())
        cbar=fig.colorbar(cax,location='bottom',pad=0.05)
        cax.set_clim(0,1)
        ax.set_xticks(range(max-min))
        ax.set_yticks(range(max-min))
        labels=ChemicalSymbols[min:max]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("K")
        ax.set_ylabel("Q")
        ax.xaxis.set_label_position('top')
        plt.savefig('result/attention%d_%d.jpg'%(min,max),dpi=2400)
        plt.show()

        fig, ax = plt.subplots(figsize=(4,6))
        min=71
        max=83
        cax=ax.matshow((total_matrix[min:max,min:max]/torch.max(total_matrix[min:max,min:max])).detach().cpu().numpy())
        cbar=fig.colorbar(cax,location='bottom',pad=0.05)
        cax.set_clim(0,1)
        ax.set_xticks(range(max-min))
        ax.set_yticks(range(max-min))
        labels=ChemicalSymbols[min:max]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("K")
        ax.set_ylabel("Q")
        ax.xaxis.set_label_position('top')
        plt.savefig('result/attention%d_%d.jpg'%(min,max),dpi=2400)
        plt.show()


    def get_attention(self,data,model):
        model.eval()
        with torch.no_grad():
            out,attention_list = model(data[0],data[2],(data[3]**3)/(2**0.5),return_attention=True)
            return attention_list

class FixedLengthLoader():
    def __init__(self, dataloader, length):
        self.dataloader = dataloader
        self.length = length
        self.iterator = iter(dataloader)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        if self.length <= 0:
            raise StopIteration
        self.length -= 1
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            data = next(self.iterator)
        return data

    def __len__(self):
        return self.length

def random_split(length,p=[0.8,0.2]):
    # random split the dataset into training set and test set
    # input: length of the dataset, p is split proportion
    # output: mask of training set and test set, may contain validation set
    if isinstance(p, float):
        rand_split=np.random.rand(length)
        train_mask=rand_split<p
        test_mask=rand_split>=p
        return train_mask, test_mask
    assert len(p) in [2,3], 'The length of p should be 2 or 3'
    assert np.sum(p)==1, 'The sum of p should be 1'
    if len(p)==2:
        rand_split=np.random.rand(length)
        train_mask=rand_split<p[0]
        test_mask=rand_split>=p[0]
        return train_mask, test_mask
    elif len(p)==3:
        rand_split=np.random.rand(length)
        train_mask=rand_split<p[0]
        val_mask=(rand_split>=p[0])&(rand_split<p[0]+p[1])
        test_mask=rand_split>=p[0]+p[1]
        return train_mask, val_mask, test_mask

def train(train_loader, model, loss_Fn, optimizer, model_type="cnn",device='cuda'):
    model.train()
    loss_all = 0
    for data in train_loader:
        #data = data.to(device)
        output = calculate_result(data,model,model_type)
        if model_type=='gnn':
            loss = loss_Fn(output, data.total_spin)
            loss_all += data.num_graphs * loss.item()
        elif model_type=='cnn':
            loss = loss_Fn(output, data[1])
            loss_all += data[1].shape[0] * loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def test(test_loader,model, loss_Fn, model_type="cnn",device='cuda'):
    model.eval()
    error = 0
    with torch.no_grad():
        for data in test_loader:
            #data = [sub_data.to(device) for sub_data in data]
            output=calculate_result(data,model,model_type)
            if model_type=="gnn":
                error+=loss_Fn(output.data.total_spin).item()*data.num_graphs
            elif model_type=="cnn":
                error+=loss_Fn(output,data[1]).item()*data[1].shape[0]
    return error / len(test_loader.dataset)

def plot_result(model, training_dataloader,test_dataloader,val_dataloader=None,\
                    model_type="cnn",device='cuda',num=100,size=0.5,fig_num=None):
    model.eval()

    training_label=[]
    training_out=[]
    test_label=[]
    test_out=[]
    if val_dataloader is not None:
        val_label=[]
        val_out=[]
    with torch.no_grad():
        for data in training_dataloader:
            output=calculate_result(data,model,model_type)
            training_out.append(output.cpu().detach().numpy())
            if model_type=="gnn":
                training_label.append(data.total_spin.cpu().detach().numpy())
            elif model_type=="cnn":
                training_label.append(data[1].cpu().detach().numpy())

        if val_dataloader is not None:
            for data in val_dataloader:
                output=calculate_result(data,model,model_type)
                val_out.append(output.cpu().detach().numpy())
                if model_type=="gnn":
                    val_label.append(data.total_spin.cpu().detach().numpy())
                elif model_type=="cnn":
                    val_label.append(data[1].cpu().detach().numpy())

        for data in test_dataloader:
            output=calculate_result(data,model,model_type)
            test_out.append(output.cpu().detach().numpy())
            if model_type=="gnn":
                test_label.append(data.total_spin.cpu().detach().numpy())
            elif model_type=="cnn":
                test_label.append(data[1].cpu().detach().numpy())
    
    training_label=np.concatenate(training_label,axis=0)
    training_out=np.concatenate(training_out,axis=0)
    test_label=np.concatenate(test_label,axis=0)
    test_out=np.concatenate(test_out,axis=0)
    if val_dataloader is not None:
        val_label=np.concatenate(val_label,axis=0)
        val_out=np.concatenate(val_out,axis=0)
    #min_edge=min(np.min(training_label),np.min(test_label))
    #max_edge=max(np.max(training_label),np.max(test_label))
    min_edge=torch.min(torch.tensor([np.min(training_label),np.min(test_label)]))
    max_edge=torch.max(torch.tensor([np.max(training_label),np.max(test_label)]))

    random_indices_train=np.array(random.sample(range(len(training_label)),num))
    random_indices_test=np.array(random.sample(range(len(test_label)),num))
    if val_dataloader is not None:
        random_indices_val=np.array(random.sample(range(len(val_label)),num))
    
    if val_dataloader is None:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))
        axs[0].scatter(np.array(training_label[random_indices_train]),np.array(training_out[random_indices_train]))
        axs[0].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')  # Add this line to plot y=x
        R2=r2_score(training_label,training_out)
        axs[0].set_title(r'$\mathrm{Training\ Data},\ R^2=%0.3f$'%R2)
        axs[0].set_xlim([min_edge,max_edge])
        axs[0].set_ylim([min_edge,max_edge])
        axs[1].scatter(np.array(test_label[random_indices_test]),np.array(test_out[random_indices_test]))
        axs[1].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')
        R2=r2_score(test_label,test_out)
        axs[1].set_title(r'$\mathrm{Test\ Data},\ R^2=%0.3f$'%R2)
        axs[1].set_xlim([min_edge,max_edge])
        axs[1].set_ylim([min_edge,max_edge])
    elif val_dataloader is not None:
        fig, axs = plt.subplots(1, 3, figsize=(21, 7))
        axs[0].scatter(np.array(training_label[random_indices_train]),np.array(training_out[random_indices_train]))
        axs[0].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')
        R2=r2_score(training_label,training_out)
        axs[0].set_title(r'$\mathrm{Training\ Data},\ R^2=%0.3f$'%R2)
        axs[0].set_xlim([min_edge,max_edge])
        axs[0].set_ylim([min_edge,max_edge])
        axs[1].scatter(np.array(val_label[random_indices_val]),np.array(val_out[random_indices_val]))
        axs[1].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')
        R2=r2_score(val_label,val_out)
        axs[1].set_title(r'$\mathrm{Validation\ Data},\ R^2=%0.3f$'%R2)
        axs[1].set_xlim([min_edge,max_edge])
        axs[1].set_ylim([min_edge,max_edge])
        axs[2].scatter(np.array(test_label[random_indices_test]),np.array(test_out[random_indices_test]))
        axs[2].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')
        R2=r2_score(test_label,test_out)
        axs[2].set_title(r'$\mathrm{Test\ Data},\ R^2=%0.3f$'%R2)
        axs[2].set_xlim([min_edge,max_edge])
        axs[2].set_ylim([min_edge,max_edge])
    plt.savefig('result.png' if fig_num is None else 'result_%d.png'%fig_num)
    plt.show()

def train_MTL(train_loader, model, loss_fn, optimizer):
    model.train()
    train_loss = []
    train_loss=[0,0]
    for data, target1, spacegroup,target2 in train_loader:
        optimizer.zero_grad()
        output_Mag = model(data,task='Mag',space_group=spacegroup)
        loss_Mag = loss_fn(output_Mag, target1)
        train_loss[0] += loss_Mag.item()
        #loss_Mag.backward()
        #optimizer.step()

        optimizer.zero_grad()
        outout_E = model(data,task='E',space_group=spacegroup)
        loss_E = loss_fn(outout_E, target2)
        loss_E.backward()
        train_loss[1] += loss_E.item()
        optimizer.step()
    return [i/len(train_loader) for i in train_loss]

def test_MTL(test_loader, model, loss_fn):
    test_loss = []
    model.eval()
    test_loss=[0,0]
    with torch.no_grad():
        for data, target1, spacegroup,target2 in test_loader:
            output_Mag = model(data,task='Mag',space_group=spacegroup)
            loss_Mag = loss_fn(output_Mag, target1)
            test_loss[0] += loss_Mag.item()
            outout_E = model(data,task='E',space_group=spacegroup)
            loss_E = loss_fn(outout_E, target2)
            test_loss[1] += loss_E.item()
    return [i/len(test_loader) for i in test_loss]

def plot_mtl_result(model, training_dataloader, test_dataloader, device='cuda',ratio=0.04,size=0.5):
    model.eval()

    training_output_Mag=[]
    training_output_E=[]
    test_output_Mag=[]
    test_output_E=[]
    training_label_Mag=[]
    training_label_E=[]
    test_label_Mag=[]
    test_label_E=[]

    with torch.no_grad():
        for data, target1, spacegroup,target2 in training_dataloader:
            #data = data.to(device)
            output_Mag = model(data,task='Mag',space_group=spacegroup)
            training_output_Mag.append(output_Mag.squeeze().cpu().detach().numpy())
            training_label_Mag.append(target1.squeeze().cpu().detach().numpy())
            output_E = model(data,task='E',space_group=spacegroup)
            training_output_E.append(output_E.squeeze().cpu().detach().numpy())
            training_label_E.append(target2.squeeze().cpu().detach().numpy())

        for data, target1, spacegroup,target2 in test_dataloader:
            #data = data.to(device)
            output_Mag = model(data,task='Mag',space_group=spacegroup)
            test_output_Mag.append(output_Mag.squeeze().cpu().detach().numpy())
            test_label_Mag.append(target1.squeeze().cpu().detach().numpy())
            output_E = model(data,task='E',space_group=spacegroup)
            test_output_E.append(output_E.squeeze().cpu().detach().numpy())
            test_label_E.append(target2.squeeze().cpu().detach().numpy())
    
    training_output_Mag=np.concatenate(training_output_Mag,axis=0)
    training_output_E=np.concatenate(training_output_E,axis=0)
    test_output_Mag=np.concatenate(test_output_Mag,axis=0)
    test_output_E=np.concatenate(test_output_E,axis=0)
    training_label_Mag=np.concatenate(training_label_Mag,axis=0)
    training_label_E=np.concatenate(training_label_E,axis=0)
    test_label_Mag=np.concatenate(test_label_Mag,axis=0)
    test_label_E=np.concatenate(test_label_E,axis=0)

    #min_edge=min(np.min(training_label),np.min(test_label))
    #max_edge=max(np.max(training_label),np.max(test_label))
    min_edge=0
    max_edge=2.5
    R2=r2_score(test_label_Mag,test_output_Mag)
    training_show,_=random_split(len(training_output_Mag),ratio)
    test_show,_=random_split(len(test_output_Mag),ratio*4)

    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs=axs.flatten()

    axs[0].scatter(np.array(training_label_Mag[training_show]),np.array(training_output_Mag[training_show]), s=size)
    axs[0].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')  # Add this line to plot y=x
    axs[0].set_title('Training Data')
    axs[0].set_xlim([min_edge,max_edge])
    axs[0].set_ylim([min_edge,max_edge])
    axs[1].scatter(np.array(test_label_Mag[test_show]),np.array(test_output_Mag[test_show]), s=size)
    axs[1].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')
    axs[1].set_title(r'$Test Data,\ R^2=%0.4f$'%R2)
    axs[1].set_xlim([min_edge,max_edge])
    axs[1].set_ylim([min_edge,max_edge])

    min_edge=-10
    max_edge=-2
    R2=r2_score(test_label_E,test_output_E)
    axs[2].scatter(np.array(training_label_E[training_show]),np.array(training_output_E[training_show]), s=size)
    axs[2].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')  # Add this line to plot y=x
    axs[2].set_title('Training Data')
    axs[2].set_xlim([min_edge,max_edge])
    axs[2].set_ylim([min_edge,max_edge])
    axs[3].scatter(np.array(test_label_E[test_show]),np.array(test_output_E[test_show]), s=size)
    axs[3].plot([min_edge,max_edge],[min_edge,max_edge], 'b-')
    axs[3].set_title(r'$Test Data,\ R^2=%0.4f$'%R2)
    axs[3].set_xlim([min_edge,max_edge])
    axs[3].set_ylim([min_edge,max_edge])

    plt.savefig('result.png')
    plt.show()