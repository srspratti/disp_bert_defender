import torch
import os 

# from random import randint
from gensim.models import Word2Vec
# from GAN2vec.src.train import get_lines
# # from GAN2vec.src.gan2vec import Generator, Discriminator
from GAN2vec_RobGAN_train import *

data_dir_path = os.getcwd() + '/data/sst-2/test.tsv'

def create_vec_model_save():
    task_name = 'sst-2'
    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    #data_dir_path = os.getcwd() + '/data/sst-2/train.tsv'
    text, text_orig, encoder, labels = get_data_encoder(data_dir_path, label_list)
    return text, text_orig, encoder, labels, processor, label_list


def load_disc():
    return torch.load('Discriminator.model')


def load_encoder():
    model_path = os.getcwd() + '/data/sst-2/sst-2_gensim_word2vec.model'
    return gensim.models.Word2Vec.load()


def discriminator_test():
    i=1
    while (i <= 10):

        # Loading trained Discriminator

        Dis_saved = load_disc()
        print("Dis_saved: ", Dis_saved)

        # Loading Text and Encoder
        text, _, encoder, _, processor,label_list= create_vec_model_save()

        rnd = randint(0, 10)
        #sentences_packed, _ = get_lines_encoder(rnd, rnd + 2, text, encoder)
        sentences_packed, _ = adversarial_attacks_for_dis(start=rnd, end=rnd+2, encoder=encoder, text=text,
                                                          processor=processor, label_list=label_list,
                                                          data_dir=data_dir_path)
        sentence_label, sentence_word_labels = Dis_saved(sentences_packed)
        print("sentence_label type: ", type(sentence_label))
        print("sentence_word_labels type: ", type(sentence_word_labels))

        # s = G.generate(sw)

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

        i=+1

def main():
    #discriminator_test(data_dir_path)
    discriminator_test()

if __name__ == '__main__':
    main()