import torch
import os 

# from random import randint
# from gensim.models import Word2Vec
# from GAN2vec.src.train import get_lines
# # from GAN2vec.src.gan2vec import Generator, Discriminator
from GAN2vec_RobGAN_train import *

# text, text_orig, encoder, labels = get_data()
# encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V)).wv
Dis_saved = torch.load('Discriminator.model')

ipt = ''
while('q' not in ipt):

    print("Dis_saved: ", Dis_saved)
    rnd = randint(0, 256)
    sentences_packed, _ = get_lines(rnd, rnd+2)

    #s = G.generate(sw)
    sentence_label, sentence_word_labels = Dis_saved(sentences_packed)
    #
    # s = s[0].detach().numpy()
    # print(s.shape)
    #
    # st = [
    #     encoder.most_similar([s[i]], topn=1)[0]
    #     for i in range(s.shape[0])
    # ]
    #
    # st, sim = list(zip(*st))
    #
    # print(' '.join(st))
    # print('\t'.join(['%0.4f' % i for i in sim]))
    # #print(s)
    ipt = input()