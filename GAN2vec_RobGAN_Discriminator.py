import torch
import os 

# from random import randint
from gensim.models import Word2Vec
# from GAN2vec.src.train import get_lines
# # from GAN2vec.src.gan2vec import Generator, Discriminator
from tqdm import tqdm, trange
from GAN2vec_RobGAN_train import *

data_dir_path = os.getcwd() + '/data/sst-2/train.tsv'
bert_model = 'bert-base-uncased'
do_lower_case = True

#tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

def create_vec_model_save():
    task_name = 'sst-2'
    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    #data_dir_path = os.getcwd() + '/data/sst-2/train.tsv'
    text, text_orig, encoder, labels = get_data_encoder(data_dir_path, label_list)

    create_vocab(data_dir_path, text_orig)
    print("w2i={}, i2w={}, CHAR_VOCAB={}".format(w2i, i2w, CHAR_VOCAB))

    return text, text_orig, encoder, labels, processor, label_list


def load_disc():
    return torch.load('Discriminator.model')


def load_encoder():
    model_path = os.getcwd() + '/data/sst-2/sst-2_gensim_word2vec.model'
    return gensim.models.Word2Vec.load()


def discriminator_test(num_batches):
    num = 0
    for num in range(num_batches):

        # Loading trained Discriminator

        Dis_saved = load_disc()
        print("Dis_saved: ", Dis_saved)

        # Loading Text and Encoder
        text, _, encoder, _, processor,label_list= create_vec_model_save()

        rnd = randint(1, 100)
        print("rnd: ", rnd)
        #sentences_packed, _ = get_lines_encoder(rnd, rnd + 2, text, encoder)
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        sentences_packed, sentence_flaw_labels_truth = adversarial_attacks_for_dis(start=rnd, end=rnd+7, encoder=encoder, text=text,
                                                          processor=processor, label_list=label_list,
                                                          data_dir=data_dir_path, tokenizer=tokenizer)
        sentence_label, sentence_word_labels = Dis_saved(sentences_packed)
        print("sentence_label type: ", type(sentence_label))
        print("sentence_word_labels type: ", type(sentence_word_labels))
        print("sentence_label type: ", sentence_label.shape)
        print("sentence_word_labels type: ", sentence_word_labels.shape)
        print("sentence_word_labels value: ", sentence_word_labels[0,:,:])
        #sentence_flaw_labels_truth
        #print("sentence_flaw_labels_truth value: ", sentence_flaw_labels_truth[0, :, :])
        print("sentence_flaw_labels_truth Shape: ", sentence_flaw_labels_truth.shape)
        print("Batch done.....")

        # TODO : Need to add Eval metric - Precision , Recall and F1
        # TODO : Need to output the format for embedding estimator to recover the tokens



def main():
    #discriminator_test(data_dir_path)
    discriminator_test(1)

if __name__ == '__main__':
    main()