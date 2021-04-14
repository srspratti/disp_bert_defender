import torch 

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from bert_model import BertForDiscriminator, BertConfig, WEIGHTS_NAME, CONFIG_NAME

class Generator(nn.Module):
    def __init__(self, latent_size, out_size, max_len=20, min_len=3, num_layers=1):
        super(Generator, self).__init__()

        print("latent_size: ", latent_size)
        print("out_size: ", out_size)
        self.out_size = out_size
        self.num_layers = num_layers
        self.MAX_LEN = max_len
        self.MIN_LEN = min_len 
        #self.MAX_LEN = 1
        #self.MIN_LEN = 1 
        self.one_hot_size = max_len-min_len
        #self.one_hot_size = 0

        self.recurrent = nn.LSTM( 
            out_size+self.one_hot_size,
            out_size+self.one_hot_size,
            num_layers=num_layers,
            batch_first=True
        )

        #recurrent = LSTM(81,81,1)
        self.linear = nn.Linear(out_size+self.one_hot_size, out_size)

    '''
    Given batch of starter words, generate a sequence of outputs
    '''
    def forward_old(self, batch, sentence_len=1):
        h_n = Variable(
            torch.zeros(
                self.num_layers, 
                batch.size(0), 
                self.out_size+self.one_hot_size
            ).normal_()
        )

        c_n = Variable(
            torch.zeros(
                self.num_layers, 
                batch.size(0), 
                self.out_size+self.one_hot_size
            )#.normal_()
        )

        # Tell the encoder how long the sentence will be 
        one_hot = torch.zeros(batch.size(0), 1, self.one_hot_size)
        one_hot[:, :, self.MAX_LEN-sentence_len] = 1.0
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
    
    # Current Forward method of the model
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

        h_n_old = Variable(
            torch.zeros(
                self.num_layers,
                batch.size(0),
                self.out_size
            ).normal_()
        )

        c_n_old = Variable(
            torch.zeros(
                self.num_layers,
                batch.size(0),
                self.out_size
            )  # .normal_()
        )



        # Tell the encoder how long the sentence will be
        one_hot = torch.zeros(batch.size(0), 1, self.one_hot_size)
        #self.out_size
        #one_hot = torch.zeros(batch.size(0), self.out_size, self.one_hot_size)
        #one_hot_size
        #out_size
        print("one_hot_size type: ", type(self.one_hot_size))
        print(" one_hot_size shape : ", self.one_hot_size)
        print("out_size type: ", type(self.out_size))
        print(" out_size shape : ", self.out_size)
        print("h_n type: ", type(h_n))
        print("h_n shape : ", h_n.shape)
        print("c_n type: ", type(c_n))
        print("c_n shape : ", c_n.shape)
        print("batch type: ", type(batch))
        print("batch shape : ", batch.shape)
        #one_hot = torch.zeros(batch.size(0), 1, 1)
        print("one_hot shape: ", one_hot.shape)
        one_hot[:, :, self.MAX_LEN - sentence_len] = 1.0
        print("one_hot shape: after ", one_hot.shape)
        x_n = torch.cat([one_hot, batch], dim=-1)
        print("x_n shape : ", x_n.shape)

        #one_hot = torch.zeros(batch.size(0), 1, self.one_hot_size)
        #one_hot[:, :, self.MAX_LEN - sentence_len] = 1.0
        

        sentence = [batch]

        print("sentence : Type ", type(sentence))
        print("sentence : Len ", len(sentence))

        for _ in range(sentence_len):
            print("sentence_len: ", sentence_len)

            x_n, (h_n, c_n) = self.recurrent(x_n, (h_n, c_n))

            # Run output through one more linear layer w no activation
            x = x_n[:, 0, :]
            x = self.linear(x)
            sentence.append(x.unsqueeze(1))

        h_n = torch.cat(sentence, dim=1)
        print(" h_n shape : ", h_n.shape)
        return h_n

    def generate(self, batch, sentence_len=5):
        with torch.no_grad():
            return self.forward(batch, sentence_len=sentence_len)


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

        #self.mbd = MinibatchDiscrimination(hidden_size, hidden_size)
        self.decider = nn.Sequential(
            nn.Linear(hidden_size*2, 1),
            nn.Sigmoid()
        )

        #self.linear_adv = nn.Linear(2 * hdim, output_dim)

        # to-do : Can we have 2 deciders in nn ?
        # TODO - 2-b : char_vocab_size , hdim and output_dim
        char_vocab_size=""
        hdim=""
        output_dim=""
        self.decider_multi = nn.sequential(
            nn.LSTM(3*char_vocab_size, hdim, 1, batch_first=True,bidirectional=True),
            nn.Linear(2*hdim, output_dim))



    def forward(self, x):
        _, (_, x) = self.recurrent(x)
        x = x[-1]

        x = self.mbd(x)
        #self.decider_multi.LSTM(x)
        #return self.decider(x)
        def get_inp_lens_from_x(x): # TODO - 3 : get inp and lens from x
            return x
        inp, lens = get_inp_lens_from_x(x)
        packed_input = pack_padded_sequence(inp, lens, batch_first=True)
        packed_output, _ = self.decider_multi.LSTM(packed_input)
        h, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.decider_multi.Linear(h)  # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2)
        #return out  # out is batch_size  x class_size x  max_seq_len

        return self.decider(x), out

    """ size(inp) --> BATCH_SIZE x MAX_SEQ_LEN x EMB_DIM 
        """
    """
    def forward(self, inp, lens):
        packed_input = pack_padded_sequence(inp, lens, batch_first=True)
        packed_output, _ = self.lstm(packed_input)
        h, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.linear(h)  # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2)
        return out  # out is batch_size  x class_size x  max_seq_len
    """

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
        o_b = (expnorm.sum(0) - 1)   # Batch x Out 
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x