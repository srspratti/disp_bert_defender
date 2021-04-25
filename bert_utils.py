import csv
import logging
import sys
import numpy as np
import os
import random
import string
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from nltk.tokenize import word_tokenize
import pprint
import io
import torch
import hnswlib
from enumerate_attacks import *

# copy-test


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vectors(fname):
    # print("fname value ****::::****=> ",fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    emb_dict = {}
    for line in fin:
        # print("line value ****::::****=> ",line)
        tokens = line.rstrip().split(' ')
        #    print("tokens value ****::::****=> ",tokens)
        #   print("tokens length ****::::****=> ",len(tokens))
        if len(tokens) == 2: continue
        emb_dict[tokens[0]] = list(map(float, tokens[1:]))
        # print("tokens[0] value ****::::****=> ",tokens[0])
    #   print("tokens[1:] value ****::::****=> ",tokens[1:])
    # print("emp_dict[tokens[0]] value ****::::****=> ", emb_dict[tokens[0]])
    vocab_list = list(emb_dict.keys())
    emb_vec = list(emb_dict.values())
    # print("vocab_list type ****::::****=> ",type(vocab_list))
    # print("vocab_list length ****::::****=> ",len(vocab_list))
    # print("vocab_list value 0 element ****::::****=> ",vocab_list[0])
    # print("emp_vec type ****::::****=> ",type(emb_vec))
    # print("emb_vec length ****::::****=> ",len(emb_vec))
    # print("emb_vec[0] value 0 element ****::::****=> ",emb_vec[0])
    return emb_dict, emb_vec, vocab_list, len(vocab_list)


def write_vocab_info(fname, vocab_size, vocab_list):
    with open(fname, 'w') as g:
        g.write(str(vocab_size) + '\n')
        for vocab in vocab_list:
            g.write(vocab + '\n')


def load_vocab_info(fname):
    f = open(fname, 'r')
    contents = f.readlines()
    return int(contents[0].replace('\n', '')), [x.replace('\n', '') for x in contents[1:]]


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None, flaw_labels=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.flaw_labels = flaw_labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputFeatures_disc_train(object):

    def __init__(self, token_ids, label_id):
        self.token_ids = token_ids
        self.label_id = label_id


class InputFeatures_disc_eval(object):

    def __init__(self, token_ids, input_ids, input_mask, flaw_labels, flaw_ids, label_id, chunks):
        self.token_ids = token_ids
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.flaw_labels = flaw_labels
        self.flaw_ids = flaw_ids
        self.label_id = label_id
        self.chunks = chunks


# class InputFeatures(object):

#     def __init__(self, input_ids, input_mask):

#         self.input_ids = input_ids
#         self.input_mask = input_mask

class InputFeatures_ngram(object):

    def __init__(self, tokens, label_id, ngram_ids, ngram_labels, ngram_masks):
        self.tokens = tokens
        self.label_id = label_id
        self.ngram_ids = ngram_ids
        self.ngram_labels = ngram_labels
        self.ngram_masks = ngram_masks


class InputFeatures_gnrt_train(object):

    def __init__(self, ngram_ids, ngram_labels, ngram_masks, ngram_embeddings):
        self.ngram_ids = ngram_ids
        self.ngram_labels = ngram_labels
        self.ngram_masks = ngram_masks
        self.ngram_embeddings = ngram_embeddings


class InputFeatures_gnrt_eval(object):

    def __init__(self, token_ids, ngram_ids, ngram_labels, ngram_masks, flaw_labels, label_id):
        self.token_ids = token_ids
        self.ngram_ids = ngram_ids
        self.ngram_labels = ngram_labels
        self.ngram_mask = ngram_masks
        self.flaw_labels = flaw_labels
        self.label_id = label_id


class InputFeatures_flaw(object):

    def __init__(self, flaw_ids, flaw_mask, flaw_labels):
        self.flaw_mask = flaw_mask
        self.flaw_ids = flaw_ids
        self.flaw_labels = flaw_labels


class InputFeatures_flaw_attacks(object):

    def __init__(self, flaw_ids, flaw_mask, flaw_labels, flaw_tokens):
        self.flaw_mask = flaw_mask
        self.flaw_ids = flaw_ids
        self.flaw_labels = flaw_labels
        self.flaw_tokens = flaw_tokens


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = label_map[example.label]
        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def convert_examples_to_features_disc_train(examples, label_list, max_seq_length, tokenizer, w2i={}, i2w={}, index=1):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        token_ids = []
        tokens = word_tokenize(example.text_a)
        if len(tokens) > max_seq_length: # TODO : just verify with the master version whether this line is ocrrect or not ?
            tokens = tokens[:max_seq_length]
        for token in tokens:
            if token not in w2i:
                w2i[token] = index
                i2w[index] = token
                index += 1
            token_ids.append(w2i[token])
        token_ids += [0] * (max_seq_length - len(token_ids))
        label_id = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))

        features.append(
            InputFeatures_disc_train(token_ids=token_ids,
                                     label_id=label_id))
    return features, w2i, i2w, index

def convert_examples_to_features_gan2vec(examples, label_list, max_seq_length, tokenizer, w2i={}, i2w={}, index=1):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        token_ids = []
        tokens = word_tokenize(example.text_a)
        print("Printing tokens truth: ", tokens)
        #print("length of tokens: ", len(tokens))
        if len(tokens) > max_seq_length: # TODO : Might need to change it back, used it for Gan2vec and RobGAN
            tokens = tokens[:max_seq_length]
        # TODO : Padding for sentences less than max_sequence_length
        # if len(tokens) < max_seq_length:
        #     tokens = tokens + tokens
        #print("length of tokens: after max_seq_length: ", len(tokens))
        for token in tokens:
            if token not in w2i:
                w2i[token] = index
                i2w[index] = token
                index += 1
            token_ids.append(w2i[token])
        token_ids += [0] * (max_seq_length - len(token_ids))
        #print("length of token_ids: ", len(token_ids))
        label_id = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))

        features.append(
            InputFeatures_disc_train(token_ids=token_ids,
                                     label_id=label_id))
    return features, w2i, i2w, index


def convert_examples_to_features_disc_eval(examples, label_list, max_seq_length, tokenizer, w2i={}, i2w={}, index=1):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    label_map = {label: i for i, label in enumerate(label_list)}
    for (ex_index, example) in enumerate(examples):

        flaw_ids = []
        # print("ex_index; ",ex_index)
        tokens = word_tokenize(example.text_a)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        if example.flaw_labels is not None:
            # print("example.flaw_labels is not None: block ")
            if example.flaw_labels == '':
                flaw_ids = [-1]
            else:
                flaw_ids = [int(x) for x in (example.flaw_labels).strip('"').strip(' ').split(',')]
                # print("flaw_ids strip split block: ")
                # flaw_ids = [int(x) for x in (example.flaw_labels).strip('"').split(',')]

        # flaw_ids: the index of flaw words on word-level
        # flaw_labels: the index of flaw words on wordpiece-level
        # usually, flaw_ids will have more flaw words than flaw_labels, since their max length both equals max_seq_length but flaw_labels need more space

        token_ids = []
        input_pieces, chunks, flaw_labels = [], [], []
        flaw_ids_cut = []

        for i, tok in enumerate(tokens):

            print("i in enumerate tokens: ", i)

            if tok not in w2i:
                w2i[tok] = index
                i2w[index] = tok
                index += 1
            token_ids.append(w2i[tok])

            word_pieces = tokenizer.tokenize(tok)
            input_pieces += word_pieces
            chunks.append(len(word_pieces))

            if i in flaw_ids:
                flaw_ids_cut.append(i)
                flaw_labels += [1] * len(word_pieces)
            else:
                flaw_labels += [0] * len(word_pieces)

            if len(input_pieces) > max_seq_length - 2:
                input_pieces = input_pieces[:(max_seq_length - 2)]
                flaw_labels = flaw_labels[:(max_seq_length - 2)]
                chunks[-1] = max_seq_length - 2 - sum(chunks[:-1])
                break

        if len(chunks) > max_seq_length - 2:
            chunks = chunks[:max_seq_length - 2]

        input_pieces = ["[CLS]"] + input_pieces + ["[SEP]"]
        flaw_labels = [0] + flaw_labels + [0]
        chunks = [1] + chunks + [1]

        input_ids = tokenizer.convert_tokens_to_ids(input_pieces)
        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        flaw_labels += padding
        chunks += [0] * (max_seq_length - len(chunks))
        token_ids += [0] * (max_seq_length - len(token_ids))
        flaw_ids += [0] * (max_seq_length - len(flaw_ids))
        flaw_ids_cut += [0] * (max_seq_length - len(flaw_ids_cut))

        label_id = label_map[example.label]

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % example.text_a)
            logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("flaw_labels: %s" % " ".join([str(x) for x in flaw_labels]))
            logger.info("flaw_ids: %s" % " ".join([str(x) for x in flaw_ids]))
            logger.info("flaw_ids_cut: %s" % " ".join([str(x) for x in flaw_ids_cut]))
            logger.info("chunks: %s" % " ".join([str(x) for x in chunks]))

        features.append(
            InputFeatures_disc_eval(token_ids=token_ids,
                                    input_ids=input_ids,
                                    input_mask=input_mask,
                                    flaw_labels=flaw_labels,
                                    flaw_ids=flaw_ids_cut,
                                    label_id=label_id,
                                    chunks=chunks))
    return features, w2i, i2w, index


def convert_tokens_to_ngram(words, max_seq_length, max_ngram_length, tokenizer, w2i, flaw_labels=None, N=2,
                            train=False):
    # padding for collecting n-grams
    words_pad = ["[CLS]"] * N + words + ["[SEP]"] * N
    # features: ngram_input_ids (batch_size, sequence_length, ngram_length)
    # labels: the word-level ids of the token to predict (batch_size, sequence_length)
    # masks: ngram_input_mask (batch_size, sequence_length, ngram_length)
    features, labels, masks = [], [], []

    for i in range(len(words)):
        if len(words) > max_seq_length and train and random.random() > 0.25:
            continue
        # two situations the ngram would be created:
        # 1. no flaw labels are given, should generate ngrams for all the tokens to train 
        # 2. flaw labels are given, should generate ngrams for those flaw tokens to test
        if (flaw_labels is not None and i in flaw_labels) or (flaw_labels is None):
            tokens = words_pad[i:(i + 1 + 2 * N)]
            labels.append(w2i[tokens[N]])
            # mask the middle for prediction
            tokens[N] = "[MASK]"
            tokens = ' '.join(tokens)
            word_pieces = tokenizer.tokenize(tokens)
            word_ids = tokenizer.convert_tokens_to_ids(word_pieces)

            if len(word_ids) > max_ngram_length:
                word_ids = word_ids[:max_ngram_length]

            mask_ids = [1] * len(word_ids)
            padding = [0] * (max_ngram_length - len(word_ids))
            word_ids += padding
            mask_ids += padding
            features.append(word_ids)
            masks.append(mask_ids)

    if len(features) > max_seq_length:
        features = features[:max_seq_length]
    if len(masks) > max_seq_length:
        masks = masks[:max_seq_length]
    if len(labels) > max_seq_length:
        labels = labels[:max_seq_length]

    padding = [[0] * max_ngram_length] * (max_seq_length - len(features))
    features += padding
    masks += padding
    labels += [0] * (max_seq_length - len(labels))

    return features, labels, masks


def convert_examples_to_features_gnrt_train(examples, label_list, max_seq_length, max_ngram_length, tokenizer,
                                            embeddings, w2i={}, i2w={}, index=1):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = word_tokenize(example.text_a)
        # if len(tokens) > max_seq_length:
        #     tokens = tokens[:max_ngram_length]

        token_ids = []
        for token in tokens:
            if token not in w2i:
                w2i[token] = index
                i2w[index] = token
                index += 1
            token_ids.append(w2i[token])

        ngram_ids, ngram_labels, ngram_masks = convert_tokens_to_ngram(tokens,
                                                                       max_seq_length,
                                                                       max_ngram_length,
                                                                       tokenizer,
                                                                       w2i,
                                                                       train=True)

        ngram_embeddings = look_up_embeddings(ngram_labels, embeddings, i2w)

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
            logger.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
            logger.info("ngram_labels: %s" % " ".join([str(x) for x in ngram_labels]))
            # logger.info("ngram_masks: %s" % " ".join([str(x) for x in ngram_masks]))
            # logger.info("ngram_embeddings: %s" % " ".join([str(x) for x in ngram_embeddings]))

        features.append(
            InputFeatures_gnrt_train(ngram_ids=ngram_ids,
                                     ngram_labels=ngram_labels,
                                     ngram_masks=ngram_masks,
                                     ngram_embeddings=ngram_embeddings))
    return features, w2i, i2w, index


def convert_examples_to_features_gnrt_eval(examples, label_list, max_seq_length, max_ngram_length, tokenizer, w2i={},
                                           i2w={}, index=1):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    label_map = {label: i for i, label in enumerate(label_list)}
    for (ex_index, example) in enumerate(examples):

        tokens = word_tokenize(example.text_a)
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]

        if example.flaw_labels is not None:
            # flaw_labels = [int(x) for x in (example.flaw_labels).split(',')]
            flaw_labels = [int(x) for x in (example.flaw_labels).strip('"').strip(' ').split(',')]
        # else:
        #     flaw_labels = list(range(len(tokens)))   

        token_ids = []
        for token in tokens:
            if token not in w2i:
                w2i[token] = index
                i2w[index] = token
                index += 1
            token_ids.append(w2i[token])

        ngram_ids, ngram_labels, ngram_masks = convert_tokens_to_ngram(tokens,
                                                                       max_seq_length,
                                                                       max_ngram_length,
                                                                       tokenizer,
                                                                       w2i,
                                                                       flaw_labels)
        flaw_labels += [0] * (max_seq_length - len(flaw_labels))
        token_ids += [0] * (max_seq_length - len(token_ids))
        label_id = label_map[example.label]

        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("token_ids: %s" % " ".join([str(x) for x in token_ids]))
            logger.info("flaw_labels: %s" % " ".join([str(x) for x in flaw_labels]))
            # logger.info("ngram_ids: %s" % " ".join([str(x) for x in ngram_ids]))
            logger.info("ngram_labels: %s" % " ".join([str(x) for x in ngram_labels]))
            # logger.info("ngram_masks: %s" % " ".join([str(x) for x in ngram_masks]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures_gnrt_eval(token_ids=token_ids,
                                    ngram_ids=ngram_ids,
                                    ngram_labels=ngram_labels,
                                    ngram_masks=ngram_masks,
                                    flaw_labels=flaw_labels,
                                    label_id=label_id))

    return features, w2i, i2w, index


def convert_examples_to_features_adv(examples, max_seq_length, tokenizer, i2w, embeddings=None, emb_index=None,
                                     words=None):
    features = []

    for (ex_index, example) in enumerate(examples):

        tokens = example
        flaw_tokens, flaw_pieces = [], []

        for tok_id in tokens:

            if tok_id == 0: break

            tok = i2w[tok_id]

            _, tok_flaw = random_attack(tok, embeddings, emb_index, words)
            word_pieces = tokenizer.tokenize(tok_flaw)

            flaw_pieces += word_pieces
            flaw_tokens.append(tok_flaw)

            if len(flaw_pieces) > max_seq_length - 2:
                flaw_pieces = flaw_pieces[:(max_seq_length - 2)]
                break

        flaw_pieces = ["[CLS]"] + flaw_pieces + ["[SEP]"]
        flaw_ids = tokenizer.convert_tokens_to_ids(flaw_pieces)
        flaw_mask = [1] * len(flaw_ids)

        padding = [0] * (max_seq_length - len(flaw_ids))
        flaw_ids += padding
        flaw_mask += padding

        assert len(flaw_ids) == max_seq_length
        assert len(flaw_mask) == max_seq_length

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("flaw_tokens: %s" % " ".join([str(x) for x in flaw_tokens]))
            logger.info("flaw_ids: %s" % " ".join([str(x) for x in flaw_ids]))
            logger.info("flaw_mask: %s" % " ".join([str(x) for x in flaw_mask]))

        features.append(
            InputFeatures_flaw(flaw_ids=flaw_ids,
                               flaw_mask=flaw_mask,
                               flaw_labels=None))

    return features


def convert_examples_to_features_flaw(examples, max_seq_length, max_ngram_length, tokenizer, i2w, embeddings=None,
                                      emb_index=None, words=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    # print("examples: ", examples)

    for (ex_index, example) in enumerate(examples):

        tokens = example
        # print("example: ", example[0])
        flaw_labels = []
        flaw_tokens, flaw_pieces = [], []

        for tok_id in tokens:

            # print("tok_id : ",tok_id)

            if tok_id == 0: break

            tok = i2w[tok_id]

            label, tok_flaw = random_attack(tok, embeddings, emb_index, words)  # embeddings
            word_pieces = tokenizer.tokenize(tok_flaw)
            print("label: ", label)

            flaw_labels += [label] * len(word_pieces)
            flaw_pieces += word_pieces

            flaw_tokens.append(tok_flaw)

            if len(flaw_pieces) > max_seq_length - 2:
                flaw_pieces = flaw_pieces[:(max_seq_length - 2)]
                flaw_labels = flaw_labels[:(max_seq_length - 2)]
                break

        flaw_pieces = ["[CLS]"] + flaw_pieces + ["[SEP]"]
        flaw_labels = [0] + flaw_labels + [0]

        flaw_ids = tokenizer.convert_tokens_to_ids(flaw_pieces)
        flaw_mask = [1] * len(flaw_ids)

        padding = [0] * (max_seq_length - len(flaw_ids))
        flaw_ids += padding
        flaw_mask += padding
        flaw_labels += padding

        assert len(flaw_ids) == max_seq_length
        assert len(flaw_mask) == max_seq_length
        assert len(flaw_labels) == max_seq_length

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("flaw_tokens: %s" % " ".join([str(x) for x in flaw_tokens]))
            logger.info("flaw_ids: %s" % " ".join([str(x) for x in flaw_ids]))
            logger.info("flaw_mask: %s" % " ".join([str(x) for x in flaw_mask]))
            logger.info("flaw_labels: %s" % " ".join([str(x) for x in flaw_labels]))

        features.append(
            InputFeatures_flaw(flaw_ids=flaw_ids,
                               flaw_mask=flaw_mask,
                               flaw_labels=flaw_labels))

    return features


def convert_examples_to_features_flaw_attacks(examples, max_seq_length, max_ngram_length, tokenizer, i2w, embeddings=None,
                                      emb_index=None, words=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    all_flaw_tokens = []
    all_token_idx = []
    all_truth_tokens = []


    for (ex_index, example) in enumerate(examples):

        tokens = example

        flaw_labels = []
        flaw_pieces = []
        token_ids_seq = []
        flaw_tokens_seq = []
        truth_tokens_seq = []

        for idx, tok_id in enumerate(tokens):

            # print("tok_id : ",tok_id)

            if tok_id == 0: break

            tok = i2w[tok_id]

            truth_tokens_seq.append(tok)

            label, tok_flaw = random_attack(tok, embeddings, emb_index, words)  # embeddings
            word_pieces = tokenizer.tokenize(tok_flaw)

            flaw_labels += [label] * len(word_pieces)
            flaw_pieces += word_pieces

            #print("label: ", label)
            flaw_tokens_seq.append(tok_flaw)
            if label == 1:
                token_ids_seq.append(int(idx))
           # token_ids_seq.append(idx)
            # print("idx: ", idx)
            # print("tok_flaw: ", tok_flaw)
            # print("flaw_tokens_seq: ", flaw_tokens_seq)
            # print("token_ids_seq: ", token_ids_seq)

            if len(flaw_pieces) > max_seq_length - 2:
                flaw_pieces = flaw_pieces[:(max_seq_length - 2)]
                flaw_labels = flaw_labels[:(max_seq_length - 2)]
                break

        all_flaw_tokens.append(flaw_tokens_seq)
        all_token_idx.append(token_ids_seq)
        all_truth_tokens.append(truth_tokens_seq)
        flaw_pieces = ["[CLS]"] + flaw_pieces + ["[SEP]"]
        flaw_labels = [0] + flaw_labels + [0]

        flaw_ids = tokenizer.convert_tokens_to_ids(flaw_pieces)
        flaw_mask = [1] * len(flaw_ids)

        padding = [0] * (max_seq_length - len(flaw_ids))
        flaw_ids += padding
        flaw_mask += padding
        flaw_labels += padding

        assert len(flaw_ids) == max_seq_length
        assert len(flaw_mask) == max_seq_length
        assert len(flaw_labels) == max_seq_length

        features.append(InputFeatures_flaw(flaw_ids=flaw_ids, flaw_mask=flaw_mask, flaw_labels=flaw_labels))

    return features, all_flaw_tokens, all_token_idx, all_truth_tokens

def convert_examples_to_features_flaw_attacks_disp(examples, max_seq_length, max_ngram_length, tokenizer, i2w,attack_type, embeddings=None,
                                      emb_index=None, words=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    all_flaw_tokens = []
    all_token_idx = []
    all_truth_tokens = []


    for (ex_index, example) in enumerate(examples):

        tokens = example

        flaw_labels = []
        flaw_pieces = []
        token_ids_seq = []
        flaw_tokens_seq = []
        truth_tokens_seq = []

        for idx, tok_id in enumerate(tokens):

            # print("tok_id : ",tok_id)

            if tok_id == 0: break

            tok = i2w[tok_id]

            truth_tokens_seq.append(tok)

            #label, tok_flaw = random_attack(tok, embeddings, emb_index, words)
            label, tok_flaw = valid_disp_attack(tok, attack_type)
            word_pieces = tokenizer.tokenize(tok_flaw)

            flaw_labels += [label] * len(word_pieces)
            flaw_pieces += word_pieces

            #print("label: ", label)
            flaw_tokens_seq.append(tok_flaw)
            if label == 1:
                token_ids_seq.append(int(idx))
           # token_ids_seq.append(idx)
            # print("idx: ", idx)
            # print("tok_flaw: ", tok_flaw)
            # print("flaw_tokens_seq: ", flaw_tokens_seq)
            # print("token_ids_seq: ", token_ids_seq)

            if len(flaw_pieces) > max_seq_length - 2:
                flaw_pieces = flaw_pieces[:(max_seq_length - 2)]
                flaw_labels = flaw_labels[:(max_seq_length - 2)]
                break

        all_flaw_tokens.append(flaw_tokens_seq)
        all_token_idx.append(token_ids_seq)
        all_truth_tokens.append(truth_tokens_seq)
        flaw_pieces = ["[CLS]"] + flaw_pieces + ["[SEP]"]
        flaw_labels = [0] + flaw_labels + [0]

        flaw_ids = tokenizer.convert_tokens_to_ids(flaw_pieces)
        flaw_mask = [1] * len(flaw_ids)

        padding = [0] * (max_seq_length - len(flaw_ids))
        flaw_ids += padding
        flaw_mask += padding
        flaw_labels += padding

        assert len(flaw_ids) == max_seq_length
        assert len(flaw_mask) == max_seq_length
        assert len(flaw_labels) == max_seq_length

        features.append(InputFeatures_flaw(flaw_ids=flaw_ids, flaw_mask=flaw_mask, flaw_labels=flaw_labels))

    return features, all_flaw_tokens, all_token_idx, all_truth_tokens




def convert_examples_to_features_flaw_attacks_gr(examples, max_seq_length, max_ngram_length, i2w,tokenizer=None, embeddings=None,
                                      emb_index=None, words=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    all_flaw_tokens = []
    all_token_idx = []
    all_truth_tokens = []
    all_flaw_labels_truth = []


    for (ex_index, example) in enumerate(examples):

        tokens = example

        flaw_labels = []
        flaw_labels_truth = []
        flaw_pieces = []
        token_ids_seq = []
        flaw_tokens_seq = []
        truth_tokens_seq = []
        print("length of the tokens: ", len(tokens))
        tokens_print = [tok for tok in tokens]
        print("tokens : ", tokens_print)
        for idx, tok_id in enumerate(tokens):

            print("tok_id : ",tok_id)

            # TODO : Important, critical addition to the code : adding new words ( common words such as 'the' , 'is' , 'a' etc..)
            #if tok_id == 0: break # Uncomment to remove the below to-do

            #if tok_id == 0 or tok_id == 1 or tok_id == 359 : tok_id = np.random.choice([2,47,3,4]) # TODO : change -1
            if tok_id == 0 or tok_id == 1 or tok_id == 359 : tok_id = 2  # TODO : change -1
            #if tok_id == 1 : tok_id = np.random.choice([2]) # TODO : change -2

            tok = i2w[tok_id]

            #print("tok_id : ", tok_id)
            #print("idx is {} : tok_id is {} and tok is {}".format(idx, tok_id,tok))

            truth_tokens_seq.append(tok)

            label, tok_flaw = random_attack(tok, embeddings, emb_index, words)  # embeddings
            word_pieces = tokenizer.tokenize(tok_flaw)
            #word_pieces = word_tokenize(tok_flaw)

            print("idx is {} : tok_id = {} : tok = {} : tok_flaw = {} : label = {} ".format(idx, tok_id, tok, tok_flaw,
                                                                                            label))

            flaw_labels += [label] * len(word_pieces)
            flaw_pieces += word_pieces

            #print("label: ", label)
            flaw_tokens_seq.append(tok_flaw)
            if label == 1:
                token_ids_seq.append(int(idx))
                flaw_label_word = 1
                flaw_labels_truth.append(flaw_label_word)
            else:
                non_flaw_word = 0
                flaw_labels_truth.append(non_flaw_word)
           # token_ids_seq.append(idx)
            # print("idx: ", idx)
            # print("tok_flaw: ", tok_flaw)
            # print("flaw_tokens_seq: ", flaw_tokens_seq)
            # print("token_ids_seq: ", token_ids_seq)

            print("word_pieces : ", word_pieces)
            print("len of flaw_pieces: ", len(flaw_pieces))
            if len(flaw_pieces) > max_seq_length - 2:
                print("s") # TODO : Just include this check while running random_attacks and check whether the control comes here or not ?
                flaw_pieces = flaw_pieces[:(max_seq_length - 2)]
                flaw_labels = flaw_labels[:(max_seq_length - 2)]
                #break

        print("all_flaw_tokens are {} and all_truth_tokens are {}".format(len(flaw_tokens_seq),len(truth_tokens_seq)))
        all_flaw_labels_truth.append(flaw_labels_truth)
        all_flaw_tokens.append(flaw_tokens_seq)
        all_token_idx.append(token_ids_seq)
        all_truth_tokens.append(truth_tokens_seq)
        flaw_pieces = ["[CLS]"] + flaw_pieces + ["[SEP]"]
        flaw_labels = [0] + flaw_labels + [0]

        # TODO : Why do we need flaw_ids
        flaw_ids = tokenizer.convert_tokens_to_ids(flaw_pieces)
        flaw_mask = [1] * len(flaw_ids)

        padding = [0] * (max_seq_length - len(flaw_ids))
        flaw_ids += padding
        flaw_mask += padding
        flaw_labels += padding

        assert len(flaw_ids) == max_seq_length
        assert len(flaw_mask) == max_seq_length
        assert len(flaw_labels) == max_seq_length
        # if (len(all_flaw_tokens) != max_seq_length):
        #     print("all_flaw_tokens: ", all_flaw_tokens)
        # if (len(all_truth_tokens) != max_seq_length):
        #     print("all_truth_tokens which is less than 6: ", all_truth_tokens)
        # #     print("all_flaw_tokens ex_index : ", ex_index)
        #assert len(all_flaw_tokens) == max_seq_length


        features.append(InputFeatures_flaw(flaw_ids=flaw_ids, flaw_mask=flaw_mask, flaw_labels=flaw_labels))

    return features, all_flaw_tokens, all_token_idx, all_truth_tokens, all_flaw_labels_truth


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",")
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(line)  # list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class IMDBProcessor(DataProcessor):
    """Processor for the IMDB-Binary data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "train")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "test.csv")), "dev")

    def get_disc_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "disc_test.csv")), "dev")

    def get_gnrt_dev_examples(self, data_dir):
        """See base class."""
        if 'csv' in data_dir:
            return self._create_examples(
                self._read_csv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_csv(os.path.join(data_dir, "disc_outputs.csv")), "dev")

    def get_clf_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(data_dir, "gnrt_outputs.csv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            flaw_labels = None
            text_a = line[1]
            label = line[0]
            if len(line) == 3: flaw_labels = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples


class SST2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "train")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_train_examples_for_attacks(self, data_dir, start, end):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples_batch_adversarial_attacks(start, end, self._read_tsv(data_dir), "train")
        else:
            return self._create_examples_batch_adversarial_attacks(start, end, self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # User Added Method - for generating random attacks similar to DISP
    def get_dev_examples_for_attacks(self, data_dir):
        #         if 'tsv' in data_dir:
        #             return self.create_examples(self._read_tsv(data_dir), "dev_attacks")
        #         else:
        #             return self.create_examples(self._read_tsv(os.path.join(data_dir, "dev_attacks.tsv")), "dev")
        if 'tsv' in data_dir:
            return self._create_examples(self._read_tsv(data_dir), "dev_attacks")
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_attacks.tsv")), "dev")

    def get_disc_dev_examples(self, data_dir):
        """See base class."""
        # print("data_dir value : ", data_dir)
        if 'tsv' in data_dir:
            # print("tsv in data_dir")
            return self._create_examples(
                self._read_tsv(data_dir), "dev")
        else:
            # print("tsv not in data_dir")
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "disc_dev.tsv")), "dev")

    def get_gnrt_dev_examples(self, data_dir):
        """See base class."""
        if 'tsv' in data_dir:
            return self._create_examples(
                self._read_tsv(data_dir), "dev")
        else:
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "disc_outputs.tsv")), "dev")

    def get_clf_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "gnrt_outputs.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #print("line :", line)
            #print("line[0]: ", line[0])
            #print("line[1]", line[1])
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


    def _create_examples_batch_adversarial_attacks(self, start, end , lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        i = int(start)
        for line_idx in range(end-start):
        #for (i, line) in enumerate(lines):
            flaw_labels = None
            #if i == 0:
            #    continue
            guid = "%s-%s" % (set_type, i)
            text_a = lines[line_idx][0]
            label = lines[line_idx][1]
            if len(lines[line_idx]) > 2: flaw_labels = lines[line_idx][2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples




    def create_examples_for_attacks(self, lines):
        "Create examples for attacks"
        sentences = []
        labels = []
        for (i, line) in enumerate(lines):
            flaw_labels = None
            if i == 0:
                continue
            text_a = line[0]
            label = line[1]
            # if len(line) == 3: flaw_labels = line[2]
            sentences.append(text_a)
            labels.append(label)
        return sentences, labels


def attack_char(token):
    sign = random.random()
    length = len(token)
    index = random.choice(range(length))
    letter = random.choice(string.ascii_letters)
    if sign < 0.33:  # swap
        token = token[:index] + letter + token[index + 1:]
    elif sign < 0.66:  # add
        token = token[:index] + letter + token[index:]
    else:  # drop
        token = token[:index] + token[index + 1:]
    return token


# Project : Added New Method
def get_pad(batch, num_dim):
    pad = [[0] * (num_dim - len(batch))]
    flat_pad = [item for sublist in pad for item in sublist]
    batch_pad = list(np.append(batch, flat_pad))
    return batch_pad


# Project : Modified Method
def load_embeddings_and_save_index(labels, emb_vec, index_path):
    '''
        labels: size-N numpy array of integer word IDs for embeddings.
        embeddings: size N*d numpy array of vectors.
        index_path: path to store the index.
    '''

    labels_ls = list(labels)
    emb_vec_ar = np.asarray(emb_vec, dtype=object)
    num_words = len(labels)
    num_dim = len(emb_vec[0])
    assert (len(labels) == len(emb_vec))
    p = hnswlib.Index(space='l2', dim=num_dim)
    p.init_index(max_elements=num_words, ef_construction=200, M=100)
    p.set_num_threads(1)
    max_cols = num_dim
    max_rows = num_words
    emb_vec_padded_less_dim_before = [get_pad(batch, num_dim) if len(batch) != 300 else batch for batch in emb_vec]
    emb_vec_padded_ar = np.asarray(emb_vec_padded_less_dim_before, dtype=object)
    p.add_items(emb_vec_padded_ar, labels)
    p.save_index(index_path)
    return p


def load_embedding_index(index_path, vocab_size, num_dim=300):
    p = hnswlib.Index(space='l2', dim=num_dim)
    p.load_index(index_path, max_elements=vocab_size)
    return p


def query_most_similar_word_id_from_embedding(p, emb, n):
    finding = p.knn_query([emb], k=n)
    return finding[0][0]


def attack_word(tok, p, emb_dict,
                vocab_list):  # TODO: attack tokens from the word-level: find a similar word and replace
    '''
        tok: string to be attacked.
        p: the object loaded by load_embedding_index or generated by load_embeddings_and_save_index.
        emb_dict: a dict, transferring a word to the corresponding embedding vector.
        emb_vocab: a list or a dict, transferring the word id to the corresponding word string.
    '''
    # embeddings: dict
    # tok: string
    if tok in emb_dict:
        tok_emb = emb_dict[tok]
    else:
        return vocab_list[random.choice(range(len(vocab_list)))]
        # pass

    most_similar_word_id = query_most_similar_word_id_from_embedding(p, tok_emb, 20)
    index = random.choice(range(len(most_similar_word_id)))
    return vocab_list[most_similar_word_id[index]]


def random_attack(tok, emb_dict, p, vocab_list):
    prob = np.random.random()
    #prob = 0.14
    #print("Printing prob: ",prob)
    # attack token with 15% probability
    if prob < 0.15:
        prob /= 0.15
        # 60% randomly attack token from character level
        if prob < 0.8:  # 0.8
            tok_flaw = attack_char(tok)
            # 40% randomly attack token from word level
        else:
            if emb_dict is not None:
                tok_flaw = attack_word(tok, p, emb_dict, vocab_list)
                #print(tok+' '+tok_flaw)
                return 1, tok_flaw
            else:
                return 0, tok
        return 1, tok_flaw
    else:
        return 0, tok


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def accuracy_2d(out, labels):
    tmp1, tmp2 = [], []
    for l in out:
        tmp1 += l
    for l in labels:
        tmp2 += l
    return np.sum([1 if tmp1[i] == tmp2[i] else 0 for i in range(len(tmp2))]) / len(tmp2)


def f1_3d(out, labels):
    num = out.shape[-1]
    out = np.reshape(out, [-1, num])
    labels = np.reshape(labels, [-1])
    outputs = np.argmax(out, axis=1)
    return f1_score(labels, outputs), recall_score(labels, outputs), precision_score(labels, outputs)


def f1_2d(labels, out):
    tmp1, tmp2 = [], []
    for l in out:
        tmp1 += l
    for l in labels:
        tmp2 += l
    return f1_score(tmp2, tmp1), recall_score(tmp2, tmp1), precision_score(tmp2, tmp1)


def look_up_embeddings(ngram_labels, embeddings, i2w):
    # ngram_labels: (batch_size, sequence_length); padded
    # ngram_embeddings: (batch_size, sequence_length, embedding_size)

    embs = []
    for label in ngram_labels:
        if label == 0:
            embs.append([0] * 300)
        else:
            word = i2w[label]
            if word in embeddings:
                embs.append(list(embeddings[word]))
            else:
                pass

    return embs


def look_up_words(ngram_logits, ngram_masks, vocab_list, p):  # embeddings
    # ngram_logits: (sequence_size, embedding_size)
    # ngram_labels: (sequence_size)
    # ngram_masks: (sequence_size)
    ngram_labels = []
    for i, emb in enumerate(ngram_logits):  # TODO: find the most similar words from the embeddings
        if sum(ngram_masks[i]) == 0: break
        most_similar_word_id = query_most_similar_word_id_from_embedding(p, emb, 1)[0]

        ngram_labels.append(vocab_list[most_similar_word_id])

    return ngram_labels


def multiplyList(myList):
    # Multiply elements one by one 
    result = 1
    for x in myList:
        result = result * x
    return result


def logit_converter(logits, chunks):
    # logits: (batch, sequence_length); padded
    # flaw_logits: (batch, sequence_length); padded
    #     print("type chunks: ", type(chunks))
    #     print("len of chunks: ", len(chunks))
    #     print("type of logits: ", type(logits))
    #     print("len of logits : ", len(logits))
    max_seq_length = len(chunks[0])
    max_batch_size = len(chunks)
    flaw_logits = []
    #     print("max_seq_length sbplshp :  ", max_seq_length)
    #     print("max_batch_size sbplshp : ", max_batch_size)

    for i in range(max_batch_size):
        flaw_logit = []
        index = 1
        for j in range(1, max_seq_length - 1):  # index = 1; (1,max_seq_length-1): [CLS] and [SEP]
            com = chunks[i][j]
            if com == 0: break
            flaw_logit.append(multiplyList(logits[i][index:(index + com)]))
            index += com
        flaw_logits.append(flaw_logit)

    return flaw_logits


def replace_token(token_ids, flaw_labels, correct_tokens, i2w):
    tokens = [i2w[x] for x in token_ids if x != 0]
    if -1 in flaw_labels:
        pass
    else:
        # flaw_labels = [x for x in flaw_labels if x != 0]
        while len(flaw_labels) > 1 and flaw_labels[-1] == 0: flaw_labels = flaw_labels[:-1]

        # print("flaw_labels:{}".format(flaw_labels))
        # print("tokens:{}".format(tokens))
        # print("correct_tokens:{}".format(correct_tokens))

        try:
            for i in range(len(flaw_labels)):
                tokens[flaw_labels[i]] = correct_tokens[i]
        except:
            # print("############# out of bound ###############")
            pass

    return tokens


processors = {
    "sst-2": SST2Processor,
    "imdb": IMDBProcessor,
    "cola": SST2Processor,
}

num_labels_task = {
    "sst-2": 2,
    "imdb": 2,
    "cola": 2,
}
