import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, max_len=20, min_len=3, num_layers=1):
        super(Generator, self).__init__()

        self.out_size = out_size
        self.num_layers = num_layers
        self.MAX_LEN = max_len
        self.MIN_LEN = min_len
        self.one_hot_size = max_len - min_len

        self.recurrent = nn.LSTM(
            out_size + self.one_hot_size,
            out_size + self.one_hot_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(out_size + self.one_hot_size, out_size)

    '''
    Given batch of starter words, generate a sequence of outputs
    '''

    def forward(self, batch, sentence_len=5):
        h_n = Variable(
            torch.zeros(
                self.num_layers,
                batch.size(0),
                self.out_size + self.one_hot_size
            ).normal_()
        )

        c_n = Variable(
            torch.zeros(
                self.num_layers,
                batch.size(0),
                self.out_size + self.one_hot_size
            )  # .normal_()
        )

        # Tell the encoder how long the sentence will be
        one_hot = torch.zeros(batch.size(0), 1, self.one_hot_size)
        one_hot[:, :, self.MAX_LEN - sentence_len] = 1.0
        x_n = torch.cat([one_hot, batch], dim=-1)

        sentence = [batch]

        for _ in range(sentence_len):
            x_n, (h_n, c_n) = self.recurrent(x_n, (h_n, c_n))

            # Run output through one more linear layer w no activation
            x = x_n[:, 0, :]
            x = self.linear(x)
            sentence.append(x.unsqueeze(1))

        h_n = torch.cat(sentence, dim=1)
        return h_n

    def generate(self, batch, sentence_len=5):
        with torch.no_grad():
            return self.forward(batch, sentence_len=sentence_len)

# class ResNetGenerator(nn.Module):
#     def __init__(self, ch=64, dim_z=128, bottom_width=4, activation=F.relu, \
#             n_classes=0, distribution="normal"):
#         super(ResNetGenerator, self).__init__()
#         self.bottom_width = bottom_width
#         self.activation = activation
#         self.distribution = distribution
#         self.dim_z = dim_z
#         self.n_classes = n_classes
#         self.l1 = nn.Linear(dim_z, (bottom_width ** 2) * ch * 16)
#         nn.init.xavier_uniform_(self.l1.weight, 1.0)
#         self.block2 = Block(ch * 16, ch * 16, activation=activation, upsample=True, n_classes=n_classes)
#         self.block3 = Block(ch * 16, ch * 8, activation=activation, upsample=True, n_classes=n_classes)
#         self.block4 = Block(ch * 8, ch * 4, activation=activation, upsample=True, n_classes=n_classes)
#         self.block5 = Block(ch * 4, ch * 2, activation=activation, upsample=True, n_classes=n_classes)
#         self.block6 = Block(ch * 2, ch * 1, activation=activation, upsample=True, n_classes=n_classes)
#         self.b7 = nn.BatchNorm2d(ch)
#         nn.init.constant_(self.b7.weight, 1.0) #XXX this is different from default initialization method
#         self.l7 = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)
#         nn.init.xavier_uniform_(self.l7.weight, 1.0)
#
#     def forward(self, z, y):
#         h = z
#         h = self.l1(h)
#         h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
#         h = self.block2(h, y)
#         h = self.block3(h, y)
#         h = self.block4(h, y)
#         h = self.block5(h, y)
#         h = self.block6(h, y)
#         h = self.b7(h)
#         h = self.activation(h)
#         h = self.l7(h)
#         h = F.tanh(h)
#         return h


from .resblocks import Block
