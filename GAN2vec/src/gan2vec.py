import torch 

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Gan2vec_RobGAN_utils.defenses.scRNN.utils import *

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

        #recurrent = LSTM(145,145,1)
        # recurrent = Linear(145,128)
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
    def __init__(self, embed_size,char_vocab_size, encoder, hidden_size=64,max_seq_length=128):
        super(Discriminator, self).__init__()

        self.embed_size = embed_size
        self.encoder = encoder
        self.max_seq_length=max_seq_length

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

        #self.linear_adv = nn.Linear(2 * hdim, output_dim)
        # to-do : Can we have 2 deciders in nn ?
        self.recurrent = nn.Sequential(
            nn.LSTM(
                embed_size,
                hidden_size,
                num_layers=3,
                batch_first=True
            ),
        )

        # TODO - 2-b : char_vocab_size , hdim and output_dim
        #char_vocab_size=CHAR_VOCAB
        print("char_vocab_size: ", char_vocab_size)
        hdim=50
        output_dim=self.max_seq_length
        #self.decider_multi = nn.Sequential(
        #    nn.LSTM(3*char_vocab_size, hdim, 1, batch_first=True,bidirectional=True),
        #    nn.Linear(2*hdim, output_dim))

        self.lstm = nn.LSTM(3*char_vocab_size, hdim, 1, batch_first=True,bidirectional=True)
        #self.lstm = nn.LSTM(char_vocab_size, hdim, 1, batch_first=True, bidirectional=True)
        #self.lstm = nn.LSTM(char_vocab_size, hdim, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hdim, output_dim)

    # TODO : Convert a tensor to packed sentence
    def cvrt_tsr_line_representation(self, packed_input):

        Xtype = torch.FloatTensor
        ytype = torch.LongTensor
        # packed_input tensor to a line format
        packed_input_seq = packed_input[0].detach().numpy()
        st = [
            self.encoder.most_similar([self.packed_input[i]], topn=1)[0]
            for i in range(self.packed_input.shape[0])
        ]

        st, sim = list(zip(*st))

        #packed_input <=> fake which is of size [ 256, 6 , 128 ] : 256 = batch_size , 6 - sentence length , 128 - word dimension length #
        packed_input_seq = packed_input[0].detach().numpy()

        for i in range(len(packed_input[0])):
            packed_input_seq = packed_input[i].detach().numpy()
            print("packed_input_seq : type ", type(packed_input_seq))
            #print("packed_input_seq : type ", packed_input_seq.)
            for it in enumerate(packed_input_seq):
                line = ""
                SEQ_LEN = len(line.split())
                #SEQ_LEN = max_
                line = line.lower()
                # TODO -mscll : Create a separate GAN2vec and RobGAN Utils
                X, _ = get_line_representation(line)
                tx = Variable(torch.from_numpy(np.array([X]))).type(Xtype)
        packed_input = pack_padded_sequence(tx, [SEQ_LEN], batch_first=True)

        return packed_input

    def get_inp_lens_from_x(packed_output):  # TODO - 3 : get inp and lens from x
        ins, len = pad_packed_sequence(packed_output, batch_first=True)
        return ins, len

    def forward(self, x):

        print("x as forward input : type ", type(x))
        print("x as forward input : Shape ", x.shape)
        #print("x as forward input : type ", type(x))

        self.packed_input = x
        #inp, lens = self.get_inp_lens_from_x(x)

        _, (_, x) = self.recurrent(x)
        x = x[-1]


        x = self.mbd(x)
        #self.decider_multi.LSTM(x)
        #return self.decider(x)

        #packed_input = pack_padded_sequence(inp, lens, batch_first=True)
        # TODO : to-check : whether the below statement is true or not ? If true , we can directly use 'x' as packed_input into the LSTM layer
        #packed_input = x
        #print("type of packed_input: x", type(self.packed_input))
        #print("shape of packed_input: ", self.packed_input.shape)
        #packed_output, _ = self.decider_multi.LSTM(packed_input)

        #packed_input_line_rep = self.cvrt_tsr_line_representation(self.packed_input)

        print("packed_input type : ", type(self.packed_input))
        print("packed_input type : ", type(self.packed_input))
       # lens = self.packed_input[:]
        print("packed_input : shape ", self.packed_input.shape[0])
        print("packed_input : type of  ", type(self.packed_input[0]))
        print("packed_input : shape of  ", self.packed_input[0].shape)
        #print("packed_input : value element 0 ", self.packed_input[0])

        self.packed_input = self.packed_input[0].detach().numpy()

        print("packed_input : shape after ", self.packed_input.shape[0])
        #print("packed_input : type of  ", type(self.packed_input[0]))
        print("packed_input : shape of  after ", self.packed_input[0].shape)
        # TODO : why the size here is 768 , 128 * 6 ??
        print("packed_input : shape of  after ", self.packed_input.size)
        #print("packed_input : shape of  after ", self.packed_input)

        st = [
            self.encoder.most_similar([self.packed_input[i]], topn=1)[0]
            for i in range(self.packed_input.shape[0])
        ]

        st, sim = list(zip(*st))

        print("st: ", st)

        #self.packed_input
        #torch.tensor_split(self.packed_input,,dim=1)
        # TODO : Need to insert a batch_size
        #lens = [6]*256
        #lens = torch.tensor([6]*256)



        packed_input_lstm = pack_padded_sequence(self.packed_input, lens, batch_first=True)
        packed_output, _ = self.lstm(packed_input_lstm)
        h, _ = pad_packed_sequence(packed_output, batch_first=True)
        #out = self.decider_multi.Linear(h)  # out is batch_size x max_seq_len x class_size
        out = self.linear(h)  # out is batch_size x max_seq_len x class_size
        out = out.transpose(dim0=1, dim1=2)
        #return out  # out is batch_size  x class_size x  max_seq_len

        return self.decider(x), out

        # out value here should be equal to Bert Discriminator logits in loss, logits = model(flaw_ids, flaw_mask, flaw_labels)

        # Another Approach for the Discriminator ( inspired from the bert Discriminator )
        #self.bert = BertModel(config)
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #self.discriminator = nn.Linear(config.hidden_size, 2)
        #self.loss_fct = CrossEntropyLoss()
        #self.apply(self.init_bert_weights)


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