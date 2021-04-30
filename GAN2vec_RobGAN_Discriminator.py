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


def discriminator_test(num_batches, max_seq_length):
    num = 0
    max_seq_len = max_seq_length
    all_eval_accuracy = 0
    all_eval_f1score = 0
    all_eval_recall = 0
    all_eval_precision = 0
    all_flaw_logits = []
    for num in range(num_batches):

        # Loading trained Discriminator

        Dis_saved = load_disc()
        print("Dis_saved: ", Dis_saved)

        # Loading Text and Encoder
        text, text_orig, _, _, processor,label_list, all_flaw_ids = create_vec_model_from_data_save()
        #text, _, _, _, processor, label_list = get_data_and_labels()
        #encoder = load_encoder()
        encoder = get_encoder_from_train()

        #rnd = randint(1, 100)
        rnd = 1
        #rnd = 59
        print("rnd: ", rnd)
        # print("text : ", text)
        # print("text_orig: ", text_orig)
        #sentences_packed, _ = get_lines_encoder(rnd, rnd + 2, text, encoder)
        tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
        #sentences_packed, sentence_flaw_labels_truth = adversarial_attacks_for_dis(start=rnd, end=rnd+8, encoder=encoder, text=text,
        #                                                   processor=processor, label_list=label_list,
        #                                                   data_dir=data_dir_path, tokenizer=tokenizer)

        sentences_packed = get_packed_sentences(text=text_orig, start=rnd, end=(rnd+256), encoder=encoder)
        sentence_label, sentence_word_labels = Dis_saved(sentences_packed)
        print("sentence_label type: ", type(sentence_label))
        print("sentence_word_labels type: ", type(sentence_word_labels))
        print("sentence_label type: ", sentence_label.shape)
        print("sentence_word_labels type: ", sentence_word_labels.shape)
        print("sentence_word_labels value: ", sentence_word_labels[0,:,:])
        #sentence_flaw_labels_truth
        #print("sentence_flaw_labels_truth value: ", sentence_flaw_labels_truth[0, :, :])
        #print("sentence_flaw_labels_truth Shape: ", sentence_flaw_labels_truth.shape)
        print("flaw_labels length : ", len(all_flaw_ids))
        print("flaw_labels length of 1st element : ", all_flaw_ids[0])
        print("all flaw_ ids : ", all_flaw_ids)
        print("Batch done.....")

        flaw_logits_t = torch.argmax(sentence_word_labels, dim=1)

        print("Type of flaw_logits: ", type(flaw_logits_t))
        print("shape of flaw_logits: ", flaw_logits_t.size())
        print("Length of flaw_logits: ", len(flaw_logits_t))
        print("flaw_logits: ", flaw_logits_t)

        # Convert flaw_logits into list
        flaw_logits = flaw_logits_t.tolist()
        all_flaw_logits.append(flaw_logits)

        print("flaw_logits type : ", type(flaw_logits))
        print("flaw_logits len : ", len(flaw_logits))
        print("flaw_logits len of 1st element: ", len(flaw_logits[0]))


        # TODO : Need to add Eval metric - Precision , Recall and F1
        # Converting flaw_ids of batch from all_flaw_ids
        srt = (rnd-1)
        ed = ( rnd+7)
        print("srt: ", srt)
        print("ed: ", ed)

        flaw_ids = all_flaw_ids[srt:ed]
        print("len of flaw_ids : ", len(flaw_ids))
        # Calculating the flaw_labels from flaw_ids
        all_flaw_labels = []
        for idx in range((rnd+8)-(rnd)):
            print("sample_idx: ", idx)
            # idx = (sample_idx-1)
            index = []
            flaw_labels = [0]*max_seq_len
            print("flaw_ids[idx] : ", flaw_ids[idx])
            print("isBlank(flaw_ids[idx]) : ", isBlank(flaw_ids[idx]))
            if isBlank(flaw_ids[idx]):
                print("Here in isBlank block: ")
                flaw_labels = [0]*max_seq_len
                all_flaw_labels.append(flaw_labels)
            else:
                clean_flaw_id = flaw_ids[idx].strip('"').split(',')
                print("clean_flaw_id : ", clean_flaw_id)
                #for val_idx, flaw_ids_val in enumerate(flaw_ids[idx]):
                for val_idx, flaw_ids_val in enumerate(clean_flaw_id):
                    # if flaw_ids_val > max_seq_len:
                    #    index.append(val_idx)
                    print("flaw_ids_val : ", flaw_ids_val)
                    if int(flaw_ids_val) <= max_seq_len : flaw_labels[(int(flaw_ids_val) - 1)] = 1
                all_flaw_labels.append(flaw_labels)
            print("flaw_labels : ", flaw_labels)
            print("flaw_logits : ", flaw_logits[idx])
            #sample_eval_accuracy = accuracy_2d(flaw_logits[idx], flaw_labels)

        print("all_flaw_logits len: ", len(all_flaw_logits))
        print("all_flaw_labels len: ", len(all_flaw_labels))
        batch_accuracy = accuracy_2d(flaw_logits, all_flaw_labels)
        all_eval_accuracy = all_eval_accuracy + batch_accuracy
        all_flaw_labels_t = torch.tensor(all_flaw_labels, dtype=torch.long)
        all_flaw_logits_t = torch.tensor(all_flaw_logits, dtype=torch.long)
        all_flaw_logits_t_s = torch.squeeze(all_flaw_logits_t, 0)
        print("all_flaw_labels_t shape : ", all_flaw_labels_t.shape)
        print("all_flaw_logits_t shape : ", all_flaw_logits_t_s.shape)
        """
        batch_f1score, batch_recall, batch_precision = f1_3d(all_flaw_logits_t_s, all_flaw_labels_t)
        all_eval_f1score = all_eval_f1score + batch_f1score
        all_eval_recall = all_eval_recall + batch_recall
        all_eval_precision = all_eval_precision + batch_precision
        """

    print("accuracy is : ", all_eval_accuracy/num_batches)
    # print("Precision is : ", all_eval_precision / num_batches)
    # print("Recall is : ", all_eval_recall / num_batches)
    # print("F1score is : ", all_eval_f1score / num_batches)

        # TODO : Need to output the format for embedding estimator to recover the tokens

def main():
    #discriminator_test(data_dir_path)
    discriminator_test(1,6)

if __name__ == '__main__':
    main()