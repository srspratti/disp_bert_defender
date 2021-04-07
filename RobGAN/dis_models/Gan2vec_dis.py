import torch
import torch.nn as nn
import torch.nn.functional as F
from .resblocks import Block, OptimizedBlock
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size=64):
        super(Discriminator, self).__init__()

        self.embed_size = embed_size

        self.recurrent = nn.Sequential(
            nn.LSTM(
                embed_size,
                hidden_size,
                num_layers=3,
                batch_first=True
            ),
        )

        self.mbd = MinibatchDiscrimination(hidden_size, hidden_size)
        self.decider = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, (_, x) = self.recurrent(x)
        x = x[-1]

        x = self.mbd(x)
        return self.decider(x)

'''
    Impliments Minibatch Discrimination to avoid same-looking output
    Shamelessly stolen from https://gist.github.com/t-ae/
'''

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims=64, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        # Outputs Batch x Out*Kernel
        matrices = x.mm(self.T.view(self.in_features, -1))

        # Transforms to Batch x Out x Kernel
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        # Now we quickly find distance from each X to every other
        # X by viewing it as a 1 x Batch x Out x Kernel mat and a
        #                      Batch x 1 x Out x Kernel mat
        # That way the difference along the kernel dimension is
        # equivilant to the dist from x to every other sample
        M = matrices.unsqueeze(0)
        M_T = M.permute(1, 0, 2, 3)

        # Simple distance formula
        norm = torch.abs(M - M_T).sum(3)  # Batch x Batch x Out
        expnorm = torch.exp(-norm)

        # Add all distances together, and remove self distance (minus 1)
        o_b = (expnorm.sum(0) - 1)  # Batch x Out
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x


# class ResNetAC(nn.Module):
#     def __init__(self, ch=64, n_classes=0, activation=F.relu, bn=False):
#         super(ResNetAC, self).__init__()
#         self.activation = activation
#         self.block1 = OptimizedBlock(3, ch, bn=bn)
#         self.block2 = Block(ch, ch * 2, activation=activation, downsample=True, bn=bn)
#         self.block3 = Block(ch * 2, ch * 4, activation=activation, downsample=True, bn=bn)
#         self.block4 = Block(ch * 4, ch * 8, activation=activation, downsample=True, bn=bn)
#         self.block5 = Block(ch * 8, ch * 16, activation=activation, downsample=True, bn=bn)
#         self.block6 = Block(ch * 16, ch * 16, activation=activation, downsample=False, bn=bn)
#         self.l7 = nn.Linear(ch * 16, 1)
#         nn.init.xavier_uniform_(self.l7.weight, gain=1.0)
#         if n_classes > 0:
#             self.l_y = nn.Linear(ch * 16, n_classes)
#             nn.init.xavier_uniform_(self.l_y.weight, gain=1.0)
#
#     def forward(self, x):
#         h = x
#         h = self.block1(h)
#         h = self.block2(h)
#         h = self.block3(h)
#         h = self.block4(h)
#         h = self.block5(h)
#         h = self.block6(h)
#         h = self.activation(h)
#         # global sum pooling (h, w)
#         #TODO try to use global avg pooling instead
#         h = h.view(h.size(0), h.size(1), -1)
#         h = torch.sum(h, 2)
#         output = self.l7(h)
#         w_y = self.l_y(h)
#         return output.view(-1), w_y





