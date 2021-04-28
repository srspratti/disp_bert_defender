import torch
import os 

# from random import randint
from gensim.models import Word2Vec
# from GAN2vec.src.train import get_lines
# # from GAN2vec.src.gan2vec import Generator, Discriminator
from tqdm import tqdm, trange
from GAN2vec_RobGAN_train import *

data_dir_path_train = os.getcwd() + '/data/sst-2/train.tsv'
data_dir_path = os.getcwd() + '/data/sst-2/add_1/enum_attacks_disp/disc_enum_add_base.tsv'
bert_model = 'bert-base-uncased'
do_lower_case = True

#tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)

def create_vec_model_from_data_save():
    task_name = 'sst-2'
    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    #data_dir_path = os.getcwd() + '/data/sst-2/train.tsv'
    text, text_orig, encoder, labels = get_data_encoder(data_dir_path, label_list)
    text, text_orig, encoder, labels, flaw_labels = get_data_flaw_labels_encoder(data_dir_path, label_list)

    #text_train, text_orig_train, encoder_train, labels_train = get_data_encoder(data_dir_path_train, label_list)
    #create_vocab(data_dir_path, text_orig)
    #create_vocab(data_dir_path_train, text_orig_train)
    print("w2i={}, i2w={}, CHAR_VOCAB={}".format(w2i, i2w, CHAR_VOCAB))

    return text, text_orig, encoder, labels, processor, label_list, flaw_labels

def get_encoder_from_train():
    task_name = 'sst-2'
    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    #data_dir_path = os.getcwd() + '/data/sst-2/train.tsv'
    #text, text_orig, encoder, labels = get_data_encoder(data_dir_path, label_list)

    text_train, text_orig_train, encoder_train, labels_train = get_data_encoder(data_dir_path_train, label_list)
    #create_vocab(data_dir_path, text_orig)
    create_vocab(data_dir_path_train, text_orig_train)
    print("w2i={}, i2w={}, CHAR_VOCAB={}".format(w2i, i2w, CHAR_VOCAB))

    return encoder_train


def get_data_and_labels():
    task_name = 'sst-2'
    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    #data_dir_path = os.getcwd() + '/data/sst-2/train.tsv'
    text, text_orig, encoder, labels = get_data_encoder(data_dir_path, label_list) # ecnoder from the attacks data provided.
    # Still need to try to load the existing vec model and test it.

    #create_vocab(data_dir_path, text_orig)
    #print("w2i={}, i2w={}, CHAR_VOCAB={}".format(w2i, i2w, CHAR_VOCAB))

    return text, text_orig, encoder, labels, processor, label_list


def load_disc():
    return torch.load('Discriminator.model')


def load_encoder():
    model_path = os.getcwd() + '/data/sst-2/sst-2_gensim_word2vec.model'
    return gensim.models.Word2Vec.load(model_path)


def discriminator_test(num_batches):
    num = 0
    for num in range(num_batches):

        # Loading trained Discriminator

        Dis_saved = load_disc()
        print("Dis_saved: ", Dis_saved)

        # Loading Text and Encoder
        text, text_orig, _, _, processor,label_list, flaw_ids = create_vec_model_from_data_save()
        #text, _, _, _, processor, label_list = get_data_and_labels()
        #encoder = load_encoder()
        encoder = get_encoder_from_train()

        rnd = randint(1, 100)
        print("rnd: ", rnd)
        # print("text : ", text)
        # print("text_orig: ", text_orig)
        #sentences_packed, _ = get_lines_encoder(rnd, rnd + 2, text, encoder)
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        #sentences_packed, sentence_flaw_labels_truth = adversarial_attacks_for_dis(start=rnd, end=rnd+8, encoder=encoder, text=text,
        #                                                   processor=processor, label_list=label_list,
        #                                                   data_dir=data_dir_path, tokenizer=tokenizer)

        sentences_packed = get_packed_sentences(text=text_orig, start=rnd, end=(rnd+8), encoder=encoder)
        sentence_label, sentence_word_labels = Dis_saved(sentences_packed)
        print("sentence_label type: ", type(sentence_label))
        print("sentence_word_labels type: ", type(sentence_word_labels))
        print("sentence_label type: ", sentence_label.shape)
        print("sentence_word_labels type: ", sentence_word_labels.shape)
        print("sentence_word_labels value: ", sentence_word_labels[0,:,:])
        #sentence_flaw_labels_truth
        #print("sentence_flaw_labels_truth value: ", sentence_flaw_labels_truth[0, :, :])
        #print("sentence_flaw_labels_truth Shape: ", sentence_flaw_labels_truth.shape)
        print("flaw_labels length : ", len(flaw_ids))
        print("flaw_labels length of 1st element : ", flaw_ids[0])
        print("Batch done.....")

        flaw_logits_t = torch.argmax(sentence_word_labels, dim=1)

        print("Type of flaw_logits: ", type(flaw_logits_t))
        print("shape of flaw_logits: ", flaw_logits_t.size())
        print("Length of flaw_logits: ", len(flaw_logits_t))
        print("flaw_logits: ", flaw_logits_t)

        # Convert flaw_logits into list
        flaw_logits = flaw_logits_t.tolist()

        print("flaw_logits type : ", type(flaw_logits))
        print("flaw_logits len : ", len(flaw_logits))
        print("flaw_logits len of 1st element: ", len(flaw_logits[0]))


        # TODO : Need to add Eval metric - Precision , Recall and F1

        # Calculating the flaw_labels from flaw_ids

        true_logits = []
        predictions, truths = [], []

        # print("length of flaw_ids: ",len(flaw_ids))

        for i in range(len(flaw_ids)):
            tmp = [0] * len(flaw_logits[i])

            # print("tmp: ",tmp) # ne line
            # print("printing i:",i)
            # print("len of tmp: ",len(tmp))
            # print("length of flaw_ids of i : ",len(flaw_ids[i]))
            # print("flaw_ids[i]: ",flaw_ids[i])

            for j in range(len(flaw_ids[0])):
                # print("flaw_ids[i][j] : ",flaw_ids[i][j])
                # print("tmp value: ", tmp)
                # print("tmp len: ", len(tmp))
                if flaw_ids[i][j] == 0: break
                if flaw_ids[i][j] >= len(tmp): continue
                tmp[flaw_ids[i][j]] = 1

            true_logits.append(tmp)
            # print('true_logits: ', true_logits)

        tmp_eval_accuracy = accuracy_2d(flaw_logits, true_logits)
        eval_accuracy += tmp_eval_accuracy

        predictions += true_logits  # Original
        truths += flaw_logits  # Original




        #assert flaw_logits.shape == sentence_flaw_labels_truth.shape
        #dis_eval_Accuracy = accuracy_2d(flaw_logits, sentence_flaw_labels_truth)

        #print("dis_eval_Accuracy : ", dis_eval_Accuracy)

        # TODO : Need to output the format for embedding estimator to recover the tokens



def main():
    #discriminator_test(data_dir_path)
    discriminator_test(1)

if __name__ == '__main__':
    main()