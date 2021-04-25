""" helper functions for
    - data loading
    - representation building
    - vocabulary loading
"""

from collections import defaultdict
import numpy as np
import pickle
import random
import os
import csv
import sys
from random import shuffle
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from gensim.models import Word2Vec
from gensim.models import FastText
from bert_utils import *


CHAR_VOCAB = []
CHAR_VOCAB_BG = []
w2i = defaultdict(lambda: 0.0)
w2i_bg = defaultdict(lambda: 0.0)
i2w = defaultdict(lambda: "UNK")
i2w_bg = defaultdict(lambda: "UNK")

#TODO: think of an open vocabulary system
WORD_LIMIT = 9999 # remaining 1 for <PAD> (this is inclusive of UNK)
task_name = ""
TARGET_PAD_IDX = -1
INPUT_PAD_IDX = 0

keyboard_mappings = None

class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None, flaw_labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.flaw_labels = flaw_labels

def set_word_limit(word_limit, task=""):
    global WORD_LIMIT
    global task_name
    WORD_LIMIT = word_limit
    task_name = task

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines

def _create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        # print("line :", line)
        # print("line[0]: ", line[0])
        # print("line[1]", line[1])
        flaw_labels = None
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, i)
        text_a = line[0]
        label = line[1]
        if len(line) > 2: flaw_labels = line[2]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
    return examples


def get_train_examples(DATA_DIR):
    """See base class."""
    if 'tsv' in DATA_DIR:
        return _create_examples(_read_tsv(DATA_DIR), "train")
    else:
        return _create_examples(_read_tsv(os.path.join(DATA_DIR, "train.tsv")), "train")

def get_text_from_train_examples(train_examples):
    all_text_train_examples = []
    for example in train_examples:
        all_text_train_examples.append(example.text_a)
    return all_text_train_examples

def get_text_and_labels_train_examples(train_examples, label_list):
    all_text_train_examples=[]
    all_labels_train_examples=[]
    label_map = {label: i for i, label in enumerate(label_list)}
    for example in train_examples:
        all_text_train_examples.append(example.text_a)
        all_labels_train_examples.append(label_map[example.label])
    return all_text_train_examples, all_labels_train_examples

# from gensim.models import KeyedVectors
# word_vectors.save('vectors.kv')
# reloaded_word_vectors = KeyedVectors.load('vectors.kv')

def get_data_encoder(data_dir, label_list):
    train_examples = get_train_examples(data_dir)
    text, labels = get_text_and_labels_train_examples(train_examples, label_list)
    text_new = [txt.split() for txt in text]
    encoder = FastText(text_new, min_count=1, size=128)
    #encoder = Word2Vec.load(os.path.join('/tmp/pycharm_project_196/GAN2vec/data/w2v_haiku.model'))
    save_path = os.getcwd() + '/data/sst-2'
    #Word2Vec.save('/sst-2_gensim_word2vec.model')
    encoder.save('/sst-2_gensim_word2vec.model')
    # model = gensim.models.Word2Vec.load("modelName.model")
    return text_new, text, encoder, labels
#adversarial_attacks_for_dis(start=rnd, end=rnd+2, encoder=encoder, text=text, processor=processor, label_list=label_list)
def adversarial_attacks_for_dis(start , end, encoder, text, processor, label_list, data_dir): # parameters : text
    # ........code here................

    # text_batch + labels
    # from train.tsv = > text_a and labels

    # text_a and labels
    # from train.tsv = get_train_examples(_create_examples(read_tsv)))
    #processor.get_train_examples_for_attacks(args.data_dir, start, end)
    test_examples = processor.get_train_examples(data_dir)
    features_for_attacks, w2i_disp, i2w_disp, vocab_size = convert_examples_to_features_gan2vec(test_examples,
                                                                                                label_list,
                                                                                                tokenizer=None,
                                                                                                max_seq_length=6)

    all_tokens = torch.tensor([f.token_ids for f in features_for_attacks], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features_for_attacks], dtype=torch.long)

    # print("all_tokens type : ", type(all_tokens))
    # print("all_label_id type : ", type(all_label_id))
    # print("all_tokens type : ", all_tokens.shape)
    # print("all_label_id type : ", all_label_id.shape)
    # assert len(all_tokens) ==

    data_for_attacks = TensorDataset(all_tokens, all_label_id)
    sampler_for_attacks = RandomSampler(data_for_attacks)  # for NO GPU
    # train_sampler = DistributedSampler(train_data)

    # dataloader_for_attack = DataLoader(data_for_attacks, sampler=sampler_for_attacks)
    # TOD : Removed Sampler, need to verify : Need to remove it for random_attacks also
    dataloader_for_attack = DataLoader(data_for_attacks)

    all_batch_flaw_tokens = []
    all_batch_flaw_labels = []
    all_batch_flaw_labels_truth = []
    flaw_labels_lst = []
    for step, batch in enumerate(tqdm(dataloader_for_attack, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        tokens, _ = batch  # , label_id, ngram_ids, ngram_labels, ngram_masks

        # module1: learn a discriminator
        tokens = tokens.to('cpu').numpy()

        # features_with_flaws, all_flaw_tokens, all_token_idx, all_truth_tokens = convert_examples_to_features_flaw_attacks(
        #    tokens,args.max_seq_length, args.max_ngram_length, tokenizer, i2w,embeddings = None, emb_index = None, words = None)

        features_with_flaws, all_flaw_tokens, all_token_idx, all_truth_tokens, all_flaw_labels_truth = convert_examples_to_features_flaw_attacks_gr(
            tokens, args.max_seq_length, args.max_ngram_length, i2w, tokenizer, embeddings=None, emb_index=None,
            words=None)

        all_token_idx = ",".join([str(id) for tok in all_token_idx for id in tok])
        all_truth_tokens_flat = ' '.join([str(id) for tok in all_truth_tokens for id in tok])

        flaw_ids = torch.tensor([f.flaw_ids for f in features_with_flaws])
        flaw_labels = torch.tensor([f.flaw_labels for f in features_with_flaws])

        print("all_flaw_tokens : before before ", all_flaw_tokens)
        all_flaw_tokens = ' '.join([str(y) for x in all_flaw_tokens for y in x])

        flaw_ids_ar = flaw_ids.detach().cpu().numpy()
        flaw_ids_lst = flaw_ids.tolist()
        flaw_labels_ar = flaw_labels.detach().cpu().numpy()
        flaw_labels_lst = flaw_labels.tolist()
        print("flaw_labels_lst : ", flaw_labels_lst)
        print("all_flaw_tokens : before ", all_flaw_tokens)
        all_flaw_tokens = all_flaw_tokens.strip("''").strip("``")
        all_truth_tokens_flat = all_truth_tokens_flat.strip("''").strip("``")
        # print("all_flaw_tokens: ",all_flaw_tokens)
        print("all_flaw_tokens : ", all_flaw_tokens)
        print("all_truth_tokens_flat : ", all_truth_tokens_flat)
        print("all_flaw_labels_truth : ", all_flaw_labels_truth)
        print("+++++++++++++++++++++++++++++++++++")
        all_batch_flaw_tokens.append(all_flaw_tokens)
        all_batch_flaw_labels.append(flaw_labels_lst)
        all_batch_flaw_labels_truth.append(all_flaw_labels_truth)

    # print("all_flaw_tokens type ", type(all_flaw_tokens))
    # print("all_flaw_tokens type ", len(all_flaw_tokens))
    # print("all_batch_flaw_tokens type ", type(all_batch_flaw_tokens))
    # print("all_batch_flaw_tokens len ", len(all_batch_flaw_tokens))
    # print("all_batch_flaw_labels type ", type(all_batch_flaw_labels))
    # print("all_batch_flaw_labels len ", len(all_batch_flaw_labels))
    batch_tx = []
    BATCH_SEQ_LEN = []
    Xtype = torch.FloatTensor
    # for line in all_batch_flaw_tokens:
    #     print("line: length : SBPLSHP : before:  ", len(line))
    #     SEQ_LEN = len(line.split())
    #     line = line.lower()
    #     print("line: length : SBPLSHP : after: ", len(line))
    #     # TODO - mscll. : Create a separate GAN2vec and RobGAN Utils
    #
    #     X = get_line_representation(line)
    #     #tx = Variable(torch.from_numpy(np.array([X]))).type(Xtype)
    #
    #     # batch_tx.append(tx)
    #     batch_tx.append(X)
    #     print("X Length ", len(X))
    #
    #
    #
    #     BATCH_SEQ_LEN.append(SEQ_LEN)
    #     # print("X :", type(X))
    #     # print("tx :", type(tx))
    #
    # # print("batch_tx : ", len(batch_tx))
    # # print("BATCH_SEQ_LEN : ", BATCH_SEQ_LEN)
    # print("BATCH_SEQ_LEN : ", BATCH_SEQ_LEN)
    # X_t = torch.tensor(batch_tx, dtype=torch.float)
    # # packed_input = pack_padded_sequence(tx, [SEQ_LEN], batch_first=True)
    # # print("X_t = torch.tensor(batch_tx, dtype=torch.float) shape: ",X_t.shape)
    for line in all_batch_flaw_tokens:
        print("line: length : SBPLSHP : before:  ", len(line))
        SEQ_LEN = len(line.split())
        line = line.lower()
        print("line: length : SBPLSHP : after: ", len(line))
        # TODO - mscll. : Create a separate GAN2vec and RobGAN Utils

        X = get_target_representation(line, encoder)
        # tx = Variable(torch.from_numpy(np.array([X]))).type(Xtype)

        # batch_tx.append(tx)
        batch_tx.append(X)
        print("X Length ", len(X))

        BATCH_SEQ_LEN.append(SEQ_LEN)
        # print("X :", type(X))
        # print("tx :", type(tx))

    # print("batch_tx : ", len(batch_tx))
    # print("BATCH_SEQ_LEN : ", BATCH_SEQ_LEN)
    print("BATCH_SEQ_LEN : ", BATCH_SEQ_LEN)
    X_t = torch.tensor(batch_tx, dtype=torch.float)
    print("X_t shape: ", X_t.shape)
    # packed_input = pack_padded_sequence(tx, [SEQ_LEN], batch_first=True)
    # print("X_t = torch.tensor(batch_tx, dtype=torch.float) shape: ",X_t.shape)
    real_adv = pack_padded_sequence(X_t, BATCH_SEQ_LEN, batch_first=True)
    all_batch_flaw_labels_truth_t = torch.tensor(all_batch_flaw_labels_truth, dtype=torch.long)

    print("all_batch_flaw_labels_truth len : ", len(all_batch_flaw_labels_truth))
    print("all_batch_flaw_labels_truth 0 element len : ", len(all_batch_flaw_labels_truth[0]))
    print("all_batch_flaw_labels_truth 0 element value : ", all_batch_flaw_labels_truth[0])
    # print("all_batch_flaw_labels_truth 0 element len : ", all_batch_flaw_labels_truth[0])
    all_batch_flaw_labels_truth_t_s = torch.squeeze(all_batch_flaw_labels_truth_t)
    # flaw_ids_or_flaw_labels
    # return real_adv, all_flaw_labels_truth
    return X_t, all_batch_flaw_labels_truth_t_s


def get_lines_encoder(start, end, text, encoder):
    # text, encoder = get_data()
    text = text
    encoder = encoder

    seq_lens = []
    sentences = []
    longest = 0
    # print("printing start: ", start)
    # print("printing end: ", end)
    text_batch = []
    for i in range((end - start)):
        text_batch.append(text[i])
    # print("Printing Text Batch: ", text_batch)
    # print("Printing Text Batch: len ", len(text_batch))
    for l in text_batch:
        # print("l in : ",l)
        seq_lens.append(len(l))
        longest = len(l) if len(l) > longest else longest
        # longest = args.max_seq_length
        # TODO : Might need to look into the max_seq_length
        # longest = 6

        sentence = []
        # print("encoder : ", encoder)
        # for txt in l.split():
        for txt in l:
            # print(" txt : ", txt)
            # print("encoder.wv[txt]) :", encoder.wv[txt])
            # print("encoder.wv[txt]) type :", type(encoder.wv[txt]))
            # print("encoder.wv[txt]) shape :", encoder.wv[txt].shape)
            sentence.append(torch.tensor(encoder.wv[txt]))
            # print(" sentence len : ", len(sentence))
            # print(" sentence type : ", type(sentence))

        # print("sentence type of : ", type(sentence))
        # print("sentences len : ", len(sentences))
        # print("sentences type : ", type(sentence))
        sentences.append(torch.stack(sentence).unsqueeze(0))

    # Pad input
    d_size = sentences[0].size(2)
    print("sentences: ", type(sentences))
    print("sentences len: ", len(sentences))
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
    # print("seq: len ", len(seq))
    # print("seq:  ", seq)
    print("seq:  shape ", seq.shape)
    print("Seq_lens: ", seq_lens)
    start_words = seq[:, 0:1, :]
    # start_words = seq[:, :, :]

    # for idx in range(len(seq_lens)):
    #  start_words = seq[:,0:(seq_lens[idx]-1), :]
    packer = pack_padded_sequence(
        seq,
        seq_lens,
        batch_first=True,
        enforce_sorted=False
    )

    # print("packer type of : ", type(packer))
    # print("start words type of : ", type(start_words))
    # print("packer type of : ", type(packer))
    # print("packer type of : shape ", packer.shape)
    # print("start words type of : ", type(start_words))
    # print("start words type of : shape ", start_words.shape)
    return packer, start_words

def get_lines(filename):
    f = open(filename)
    lines = f.readlines()
    if "|||" in lines[0]:
        # remove the tag
        clean_lines = [line.split("|||")[1].strip().lower() for line in lines]
    else:
        clean_lines = [line.strip().lower() for line in lines]
    return clean_lines


def create_vocab_old(filename, background_train=False, cv_path=""):
    global w2i, i2w, CHAR_VOCAB
    lines = get_lines(filename)
    for line in lines:
        for word in line.split():

            # add all its char in vocab
            for char in word:
                if char not in CHAR_VOCAB:
                    CHAR_VOCAB.append(char)

            w2i[word] += 1.0

    if background_train:
        CHAR_VOCAB = pickle.load(open(cv_path, 'rb'))
    word_list = sorted(w2i.items(), key=lambda x:x[1], reverse=True)
    word_list = word_list[:WORD_LIMIT] # only need top few words

    # remaining words are UNKs ... sorry!
    w2i = defaultdict(lambda: WORD_LIMIT) # default id is UNK ID
    w2i['<PAD>'] = INPUT_PAD_IDX # INPUT_PAD_IDX is 0
    i2w[INPUT_PAD_IDX] = '<PAD>'
    for idx in range(WORD_LIMIT-1):
        w2i[word_list[idx][0]] = idx+1
        i2w[idx+1] = word_list[idx][0]

    pickle.dump(dict(w2i), open("vocab/" + task_name + "w2i_" + str(WORD_LIMIT) + ".p", 'wb'))
    pickle.dump(dict(i2w), open("vocab/" + task_name + "i2w_" + str(WORD_LIMIT) + ".p", 'wb')) # don't think its needed
    pickle.dump(CHAR_VOCAB, open("vocab/" + task_name + "CHAR_VOCAB_ " + str(WORD_LIMIT) + ".p", 'wb'))
    return


def load_vocab_dicts(wi_path, iw_path, cv_path, use_background=False):
    wi = pickle.load(open(wi_path, 'rb'))
    iw = pickle.load(open(iw_path, 'rb'))
    cv = pickle.load(open(cv_path, 'rb'))
    if use_background:
        convert_vocab_dicts_bg(wi, iw, cv)
    else:
        convert_vocab_dicts(wi, iw, cv)

""" converts vocabulary dictionaries into defaultdicts
"""
def convert_vocab_dicts(wi, iw, cv):
    global w2i, i2w, CHAR_VOCAB
    CHAR_VOCAB = cv
    w2i = defaultdict(lambda: WORD_LIMIT)
    for w in wi:
        w2i[w] = wi[w]

    for i in iw:
        i2w[i] = iw[i]
    return

def convert_vocab_dicts_bg(wi, iw, cv):
    global w2i_bg, i2w_bg, CHAR_VOCAB_BG
    CHAR_VOCAB_BG = cv
    w2i_bg = defaultdict(lambda: WORD_LIMIT)
    for w in wi:
        w2i_bg[w] = wi[w]

    for i in iw:
        i2w_bg[i] = iw[i]
    return


def get_target_representation(line, encoder):
    #print("w2i: ", encoder['the'])
    return [encoder[word] for word in line.split()]

def pad_input_sequence(X, max_len):
    assert (len(X) <= max_len)
    while len(X) != max_len:
        X.append([INPUT_PAD_IDX for _ in range(len(X[0]))])
    return X

def pad_target_sequence(y, max_len):
    assert (len(y) <= max_len)
    while len(y) != max_len:
        y.append(TARGET_PAD_IDX)
    return y

def get_batched_input_data(lines, batch_size, rep_list=['swap'], probs=[1.0]):
    #shuffle(lines)
    total_len = len(lines)
    output = []
    for batch_start in range(0, len(lines) - batch_size, batch_size):

        input_lines = []
        modified_lines = []
        X = []
        y = []
        lens = []
        max_len = max([len(line.split()) \
                for line in lines[batch_start: batch_start + batch_size]])

        for line in lines[batch_start: batch_start + batch_size]:
            X_i, modified_line_i = get_line_representation(line, rep_list, probs)
            assert (len(line.split()) == len(modified_line_i.split()))
            y_i = get_target_representation(line)
            # pad X_i, and y_i
            X_i = pad_input_sequence(X_i, max_len)
            y_i = pad_target_sequence(y_i, max_len)
            # append input lines, modified lines, X_i, y_i, lens
            input_lines.append(line)
            modified_lines.append(modified_line_i)
            X.append(X_i)
            y.append(y_i)
            lens.append(len(modified_line_i.split()))

        output.append((input_lines, modified_lines, np.array(X), np.array(y), lens))
    return output

def get_target_representation_old(line):
    return [w2i[word] for word in line.split()]

def get_vector_representation(word):
    return w2i[word]

def get_line_representation(line):
    rep = []
    #modified_words = []
    for word in line.split():
        #print("word: ", word)
        word_rep, _ = get_swap_word_representation(word)
        #print("word_rep: ", word_rep)
        new_word = word
        rep.append(word_rep)
        # modified_words.append(new_word)
    return rep

def loss_nll(bin_output, bin_label, multi_output, multi_label, lam=0.5):
    L1 = F.binary_cross_entropy_with_logits(bin_output, bin_label)
    L2 = F.cross_entropy(multi_output, multi_label)
    return lam * L1 + (1.0 - lam) * L2

def create_vocab(data_dir,text, background_train=False, cv_path=""):

        #train_examples = get_train_examples(data_dir)
        global w2i, i2w, CHAR_VOCAB
        #lines = get_lines(data_dir)
        #print("Text : ", text)
        for line in text:
            #print("Line: ", line)
            for word in line.split():

                # add all its char in vocab
                for char in word:
                    if char not in CHAR_VOCAB:
                        CHAR_VOCAB.append(char)

                w2i[word] += 1.0

        if background_train:
            CHAR_VOCAB = pickle.load(open(cv_path, 'rb'))
        word_list = sorted(w2i.items(), key=lambda x: x[1], reverse=True)
        word_list = word_list[:WORD_LIMIT]  # only need top few words

        # remaining words are UNKs ... sorry!
        w2i = defaultdict(lambda: WORD_LIMIT)  # default id is UNK ID
        w2i['<PAD>'] = INPUT_PAD_IDX  # INPUT_PAD_IDX is 0
        i2w[INPUT_PAD_IDX] = '<PAD>'
        for idx in range(WORD_LIMIT - 1):
            w2i[word_list[idx][0]] = idx + 1
            i2w[idx + 1] = word_list[idx][0]

        pickle.dump(dict(w2i), open("./GAN2vec_RobGAN_data_oup/vocab/" + task_name + "w2i_" + str(WORD_LIMIT) + ".p", 'wb'))
        pickle.dump(dict(i2w),
                    open("./GAN2vec_RobGAN_data_oup/vocab/" + task_name + "i2w_" + str(WORD_LIMIT) + ".p", 'wb'))  # don't think its needed
        pickle.dump(CHAR_VOCAB, open("./GAN2vec_RobGAN_data_oup/vocab/" + task_name + "CHAR_VOCAB_ " + str(WORD_LIMIT) + ".p", 'wb'))
        return


def get_line_representation_old(line, rep_list=['swap'], probs=[1.0]):
    rep = []
    modified_words = []
    for word in line.split():
        rep_type = np.random.choice(rep_list, 1, p=probs)[0]
        if 'swap' in rep_type:
            word_rep, new_word = get_swap_word_representation(word)
        elif 'drop' in rep_type:
            word_rep, new_word = get_drop_word_representation(word, 1.0)
        elif 'add' in rep_type:
            word_rep, new_word = get_add_word_representation(word)
        elif 'key' in rep_type:
            word_rep, new_word = get_keyboard_word_representation(word)
        elif 'none' in rep_type or 'normal' in rep_type:
            word_rep, _ = get_swap_word_representation(word)
            new_word = word
        else:
            #TODO: give a more ceremonious error...
            raise NotImplementedError
        rep.append(word_rep)
        modified_words.append(new_word)
    return rep, " ".join(modified_words)


""" word representation from individual chars
    one hot (first char) + bag of chars (middle chars) + one hot (last char)
"""
def get_swap_word_representation(word):

    # dirty case
    if len(word) == 1 or len(word) == 2:
        rep = one_hot(word[0]) + zero_vector() + one_hot(word[-1])
        return rep, word

    #print("char_vocab: ", CHAR_VOCAB)
    # print("one_hot(word[0]) :", one_hot(word[0]))
    # print(" bag_of_chars(word[1:-1]) :", bag_of_chars(word[1:-1]))
    # print(" one_hot(word[-1]) :", one_hot(word[-1]))


    rep = one_hot(word[0]) + bag_of_chars(word[1:-1]) + one_hot(word[-1])
    if len(word) > 3:
        idx = random.randint(1, len(word)-3)
        word = word[:idx] + word[idx + 1] + word[idx] + word[idx+2:]

    return rep, word




""" word representation from individual chars (except that one of the internal
    chars might be dropped with a probability prob
"""
def get_drop_word_representation(word, prob=0.5):
    p = random.random()
    if len(word) >= 5 and p < prob:
        idx = random.randint(1, len(word)-2)
        word = word[:idx] + word[idx+1:]
        rep, _ = get_swap_word_representation(word) # don't care about the returned word
    elif p > prob:
        rep, word = get_swap_word_representation(word)
    else:
        rep, _ = get_swap_word_representation(word) # don't care about the returned word
    return rep, word


def get_add_word_representation(word):
    if len(word) >= 3:
        idx = random.randint(1, len(word)-1)
        random_char = _get_random_char()
        word = word[:idx] + random_char + word[idx:]
        rep, _ = get_swap_word_representation(word) # don't care about the returned word
    else:
        rep, _ = get_swap_word_representation(word) # don't care about the returned word
    return rep, word

def get_keyboard_word_representation(word):
    if len(word) >=3:
        idx = random.randint(1, len(word)-2)
        keyboard_neighbor = _get_keyboard_neighbor(word[idx])
        word = word[:idx] + keyboard_neighbor + word[idx+1:]
        rep, _ = get_swap_word_representation(word) # don't care about the returned word
    else:
        rep, _ = get_swap_word_representation(word) # don't care about the returned word
    return rep, word


""" word representation from bag of chars
"""
def get_boc_word_representation(word):
    return zero_vector() + bag_of_chars(word) + zero_vector()


def one_hot(char):
    return [1.0 if ch == char else 0.0 for ch in CHAR_VOCAB]


def bag_of_chars(chars):
    return [float(chars.count(ch)) for ch in CHAR_VOCAB]


def zero_vector():
    return [0.0 for _ in CHAR_VOCAB]


#TODO: is that all the characters we need??
def _get_random_char():
    alphabets = "abcdefghijklmnopqrstuvwxyz"
    alphabets = [i for i in alphabets]
    return np.random.choice(alphabets, 1)[0]


def _get_keyboard_neighbor(ch):
    global keyboard_mappings
    if keyboard_mappings is None or len(keyboard_mappings) != 26:
        keyboard_mappings = defaultdict(lambda: [])
        keyboard = ["qwertyuiop", "asdfghjkl*", "zxcvbnm***"]
        row = len(keyboard)
        col = len(keyboard[0])

        dx = [-1, 1, 0, 0]
        dy = [0, 0, -1, 1]

        for i in range(row):
            for j in range(col):
                for k in range(4):
                    x_, y_ = i + dx[k], j + dy[k]
                    if (x_ >= 0 and x_ < row) and (y_ >= 0 and y_ < col):
                        if keyboard[x_][y_] == '*': continue
                        if keyboard[i][j] == '*': continue
                        keyboard_mappings[keyboard[i][j]].append(keyboard[x_][y_])

    if ch not in keyboard_mappings: return ch
    return np.random.choice(keyboard_mappings[ch], 1)[0]
