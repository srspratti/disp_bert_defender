import torch
import pickle
import os
import time

from torch import nn
from torch.autograd import Variable
from random import randint
from torch.optim import Adam
#from gan2vec import Discriminator, Generator
# from gan2vec_conv import ConvGenerator
from GAN2vec_gen_dis import Discriminator, Generator
from torch.nn.utils.rnn import pack_padded_sequence
from gensim.models import Word2Vec

#DATA_DIR = '/Users/sreeramasanjeevpratti/PycharmProjects/disp_bert_defender/GAN2vec/data'
DATA_DIR = '/tmp/pycharm_project_196/GAN2vec/data' # For debugger
IN_TEXT = 'cleaned_haiku.data'
IN_W2V = 'w2v_haiku.model'

text = encoder = None


def get_data():
    global text, encoder
    if text:
        return

    print("CWD : ", os.getcwd())
    print("path value : ", os.path.join(DATA_DIR, IN_TEXT))
    with open(os.path.join(DATA_DIR, IN_TEXT), 'rb') as f:
        # text = pickle.load(f)[:256]
        text = pickle.load(f)[:5]
    encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V))
    # print("text in get_data(): ", text)
    # print("encoder value in get_data(): ", encoder)


def get_lines(start, end):
    get_data()

    seq_lens = []
    sentences = []
    longest = 0
    # print("start: ", start)
    # print("end: ", end)
    for l in text[start:end]:
        # print("l in text: ", l)
        seq_lens.append(len(l))
        longest = len(l) if len(l) > longest else longest

        sentence = []
        for w in l:
            # print("w in l: ", w)
            # print("encoder.wv[w]", type(encoder.wv[w]))
            # print("encoder.wv[w] len", len(encoder.wv[w]))
            # print("torch.tensor(encoder.wv[w]) : ", torch.tensor(encoder.wv[w]))
            sentence.append(torch.tensor(encoder.wv[w]))
            # print("sentence: ", type(sentence))
            # print("sentence: ", len(sentence))

        # print("sentences len : ", len(sentences))
        # print("sentences type : ", type(sentence))
        sentences.append(torch.stack(sentence).unsqueeze(0))

    # Pad input
    # print("sentences of 0 ", sentences[0])
    # print("sentences of 0 type : ", type(sentences[0]))
    # print("sentences of 0 len : ", len(sentences[0]))
    d_size = sentences[0].size(2)
    # print("d_size: ", d_size)
    for i in range(len(sentences)):
        sl = sentences[i].size(1)

        if sl < longest:
            sentences[i] = torch.cat(
                [sentences[i], torch.zeros(1, longest - sl, d_size)],
                dim=1
            )

    # Need to squish sentences into [0,1] domain
    seq = torch.cat(sentences, dim=0)
    # seq = torch.sigmoid(seq)
    print("seq: type ", type(seq))
    print("seq: len ", len(seq))
    print("seq:  ", seq)
    start_words = seq[:, 0:1, :]
    packer = pack_padded_sequence(
        seq,
        seq_lens,
        batch_first=True,
        enforce_sorted=False
    )

    return packer, start_words


def get_closest(sentences):
    scores = []
    wv = encoder.wv
    for s in sentences.detach().numpy():
        st = [
            wv[wv.most_similar([s[i]], topn=1)[0][0]]
            for i in range(s.shape[0])
        ]
        scores.append(torch.tensor(st))

    return torch.stack(scores, dim=0)


def train(epochs, batch_size=5, latent_size=256, K=1):
    get_data()
    # print("text type : ", type(text))
    # print("text: ", text)
    # print("text len : ", len(text))
    num_samples = len(text)
    # print("num_samples: ", num_samples)

    G = Generator(64, 64)
    D = Discriminator(64)

    l2 = nn.MSELoss()
    loss = nn.BCELoss()
    opt_d = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
    opt_g = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))

    for e in range(epochs):
        i = 0
        while batch_size * i < num_samples:
            stime = time.time()

            start = batch_size * i
            end = min(batch_size * (i + 1), num_samples)
            bs = end - start

            # Use lable smoothing
            tl = torch.full((bs, 1), 0.9)
            fl = torch.full((bs, 1), 0.1)

            # Train descriminator
            opt_d.zero_grad()
            real, greal = get_lines(start, end)
            print("real: ", real)
            print("real: type:  ", type(real))
            print("real: shape:  ", len(real))
            print("greal: ", greal)
            print("greal: shape : ", greal.shape)
            print("greal: type : ", type(greal))
            fake = G(greal)
            print("fake: ", fake)
            print("fake: shape : ", fake.shape)
            print("fake: type : ", type(fake))

            r_loss = loss(D(real), tl)
            f_loss = loss(D(fake), fl)

            r_loss.backward()
            f_loss.backward()
            d_loss = (r_loss.mean().item() + f_loss.mean().item()) / 2
            opt_d.step()

            # Train generator
            for _ in range(K):
                opt_g.zero_grad()

                # GAN fooling ability
                fake = G(greal)
                g_loss = loss(D(fake), tl)
                g_loss.backward()
                opt_g.step()

            g_loss = g_loss.item()

            print(
                '[%d] D Loss: %0.3f  G Loss %0.3f  (%0.1fs)' %
                (e, d_loss, g_loss, time.time() - stime)
            )

            i += 1

        if e % 10 == 0:
            torch.save(G, 'generator.model')
    torch.save(D, 'Discriminator.model')
    torch.save(G, 'generator.model')


torch.set_num_threads(16)
if __name__ == '__main__':
    # train(1000, batch_size=256)
    train(2, batch_size=5)
