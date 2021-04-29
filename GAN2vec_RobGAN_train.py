from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
#copy-test
import numpy as np
import pandas as pd
import torch
import pickle
import time
import gensim.models.wrappers.fasttext
#pydevd_pycharm.settrace('localhost', port=$SERVER_PORT, stdoutToServer=True, stderrToServer=True)

from torch import nn
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm, trange
from random import randint
from GAN2vec.src.gan2vec import Discriminator, Generator
# from gan2vec_conv import ConvGenerator
from gensim.models import Word2Vec
from gensim.models import FastText
#from gensim.models.wrappers import FastText
from collections import defaultdict

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert_model import BertForDiscriminator, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from Gan2vec_RobGAN_utils.defenses.scRNN.model import ScRNN
from Gan2vec_RobGAN_utils.biLstm import BiLSTM
#from RobGAN.miscs.loss import loss_nll
#from RobGAN.miscs.loss import loss_nll
from Gan2vec_utils import *

from bert_utils import *

# CHAR_VOCAB = []
# CHAR_VOCAB_BG = []
# w2i = defaultdict(lambda: 0.0)
# w2i_bg = defaultdict(lambda: 0.0)
# i2w = defaultdict(lambda: "UNK")
# i2w_bg = defaultdict(lambda: "UNK")

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--sample",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--word_embedding_file",
                        default='./emb/wiki-news-300d-1M.vec',
                        type=str,
                        help="The input directory of word embeddings.")
    parser.add_argument("--index_path",
                        default='./emb/p_index.bin',
                        type=str,
                        help="The input directory of word embedding index.")
    parser.add_argument("--word_embedding_info",
                        default='./emb/vocab_info.txt',
                        type=str,
                        help="The input directory of word embedding info.")
    parser.add_argument("--data_file",
                        default='',
                        type=str,
                        help="The input directory of input data file.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_ngram_length",
                        default=16,
                        type=int,
                        help="The maximum total ngram sequence")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--embedding_size",
                        default=300,
                        type=int,
                        help="Total batch size for embeddings.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_eval_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of eval epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--single',
                        action='store_true',
                        help="Whether only evaluate a single epoch")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=args.do_lower_case)


    # Parameters directly assignment in the code
    #DATA_DIR = 'data'
    DATA_DIR = './data/sst-2/train.tsv' # For debugger
    IN_TEXT = 'cleaned_haiku.data'
    IN_W2V = 'w2v_haiku.model'

    # to-do: think of an open vocabulary system
    WORD_LIMIT = 9999  # remaining 1 for <PAD> (this is inclusive of UNK)
    #task_name = ""
    TARGET_PAD_IDX = -1
    INPUT_PAD_IDX = 0

    keyboard_mappings = None

    text = encoder = None
    print("args.num_train_epochs  :", args.num_train_epochs )

    def _create_examples(lines, set_type):
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

    def _create_examples_clean(lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            flaw_labels = None
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_raw = line[0]
            text_a = clean_text(text_raw)
            label = line[1]
            if len(line) > 2: flaw_labels = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
        return examples

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

    def get_train_examples(DATA_DIR):
        """See base class."""
        if 'tsv' in DATA_DIR:
            return _create_examples(_read_tsv(DATA_DIR), "train")
        else:
            return _create_examples(
                _read_tsv(os.path.join(DATA_DIR, "train.tsv")), "train")

    def get_train_examples_clean(DATA_DIR):
        """See base class."""
        if 'tsv' in DATA_DIR:
            return _create_examples_clean(_read_tsv(DATA_DIR), "train")
        else:
            return _create_examples_clean(
                _read_tsv(os.path.join(DATA_DIR, "train.tsv")), "train")

    def get_text_from_train_examples(train_examples):
        all_text_train_examples=[]
        for example in train_examples:
            all_text_train_examples.append(example.text_a)

        return all_text_train_examples

    def get_text_and_labels_train_examples(train_examples):
        all_text_train_examples=[]
        all_labels_train_examples=[]
        label_map = {label: i for i, label in enumerate(label_list)}
        for example in train_examples:
            all_text_train_examples.append(example.text_a)
            all_labels_train_examples.append(label_map[example.label])

        return all_text_train_examples, all_labels_train_examples

    text = None
    encoder = None

    def get_data():

        #train_examples = get_train_examples(args.data_dir)
        train_examples = get_train_examples_clean(args.data_dir)
        text, labels = get_text_and_labels_train_examples(train_examples)

        logger.info("Loading word embeddings ...in Gensim format ")

        text_new = [txt.split() for txt in text]

        encoder = FastText(text_new, min_count=1, size=128)

        return text_new, text, encoder, labels

    def get_lines(start, end, text, encoder):

        text = text
        encoder = encoder

        seq_lens = []
        sentences = []
        longest = 0
        text_batch = []
        for i in range((end-start)):
            text_batch.append(text[i])
        for l in text_batch :
            seq_lens.append(len(l))
            longest = len(l) if len(l) > longest else longest
            sentence = []
            for txt in l:
                sentence.append(torch.tensor(encoder.wv[txt]))
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
        start_words = seq[:, 0:1, :]
        packer = pack_padded_sequence(
            seq,
            seq_lens,
            batch_first=True,
            enforce_sorted=False
        )
        return packer, start_words

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

    """ word representation from individual chars
        one hot (first char) + bag of chars (middle chars) + one hot (last char)
    """

    def get_swap_word_representation(word):

        # dirty case
        if len(word) == 1 or len(word) == 2:
            rep = one_hot(word[0]) + zero_vector() + one_hot(word[-1]) # Return value used
            return rep, word

        rep = one_hot(word[0]) + bag_of_chars(word[1:-1]) + one_hot(word[-1]) # Return value used
        if len(word) > 3:
            idx = random.randint(1, len(word) - 3)
            word = word[:idx] + word[idx + 1] + word[idx] + word[idx + 2:] # return value not used

        return rep, word

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


    torch.set_num_threads(16)
    device = torch.device("cpu")

    sample_task = args.sample.lower()

    def adversarial_attacks(start , end, encoder):

        train_examples_batch = processor.get_train_examples_for_attacks(args.data_dir, start , end)
        features_for_attacks, w2i_disp, i2w_disp, vocab_size = convert_examples_to_features_gan2vec(train_examples_batch, label_list,tokenizer=None,max_seq_length=6)

        all_tokens = torch.tensor([f.token_ids for f in features_for_attacks], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in features_for_attacks], dtype=torch.long)

        data_for_attacks = TensorDataset(all_tokens, all_label_id)
        sampler_for_attacks = SequentialSampler(data_for_attacks)

        dataloader_for_attack = DataLoader(data_for_attacks, sampler=sampler_for_attacks)

        all_batch_flaw_tokens = []
        all_batch_flaw_labels = []
        all_batch_flaw_labels_truth = []
        flaw_labels_lst = []
        for step, batch in enumerate(tqdm(dataloader_for_attack, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            tokens, _ = batch  # , label_id, ngram_ids, ngram_labels, ngram_masks

            tokens = tokens.to('cpu').numpy()

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
            all_flaw_tokens = all_flaw_tokens.strip("''").strip("``")
            all_truth_tokens_flat = all_truth_tokens_flat.strip("''").strip("``")
            all_batch_flaw_tokens.append(all_flaw_tokens)
            all_batch_flaw_labels.append(flaw_labels_lst)
            all_batch_flaw_labels_truth.append(all_flaw_labels_truth)

        batch_tx = []
        BATCH_SEQ_LEN = []
        Xtype = torch.FloatTensor
        for line in all_batch_flaw_tokens:
            SEQ_LEN = len(line.split())
            line = line.lower()
            X = get_target_representation(line, encoder)
            batch_tx.append(X)
            BATCH_SEQ_LEN.append(SEQ_LEN)
        X_t = torch.tensor(batch_tx, dtype=torch.float)
        real_adv = pack_padded_sequence(X_t, BATCH_SEQ_LEN, batch_first=True)
        all_batch_flaw_labels_truth_t = torch.tensor(all_batch_flaw_labels_truth, dtype=torch.long)
        all_batch_flaw_labels_truth_t_s = torch.squeeze(all_batch_flaw_labels_truth_t)
        return X_t, all_batch_flaw_labels_truth_t_s

    def get_loss():
        return loss_nll, loss_nll

    def train(epochs, batch_size=256, latent_size=256, K=1):
        text, text_orig, encoder, labels = get_data()
        num_samples = len(text)
        create_vocab(args.data_dir,text_orig)
        G = Generator(128, 128)
        D = Discriminator(128, len(CHAR_VOCAB), encoder)

        l2 = nn.MSELoss()
        loss = get_loss()
        opt_d = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
        opt_g = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))

        max_seq_len = args.max_seq_length
        for e in range(epochs):
            i = 0

            while batch_size * i < num_samples:
                stime = time.time()

                start = batch_size * i
                end = min(batch_size * (i + 1), num_samples)
                bs = end - start
                # Fixed labels
                test_seq_length = 6
                zeros = torch.zeros(bs, test_seq_length, dtype=torch.long)
                ones = torch.ones(bs, test_seq_length, dtype=torch.long)

                tl = torch.full((bs, 1), 0.9)
                fl = torch.full((bs, 1), 0.1)

                real, greal = get_lines(start, end, text, encoder)

                # Train Generator as per RobGAN
                for _ in range(K):
                    opt_g.zero_grad()

                    # GAN fooling ability
                    fake = G(greal)
                    d_fake_bin, d_fake_multi=D(fake)
                    g_loss = loss_nll(d_fake_bin, tl, d_fake_multi, zeros, lam=0.5)
                    g_loss.backward()
                    opt_g.step()

                g_loss = g_loss.item()

                # Train descriminator
                opt_d.zero_grad()
                fake = G(greal)

                d_fake_bin_d, d_fake_multi_d = D(fake)
                d_f_loss = loss_nll(d_fake_bin_d, fl, d_fake_multi_d, ones, lam=0.5)
                real_adv, flaw_labels = adversarial_attacks(start, end, encoder)

                d_adv_bin, d_adv_multi = D(real_adv)
                d_adv_loss = loss_nll(d_adv_bin, tl, d_adv_multi, flaw_labels, lam=0.5)
                d_loss_total = d_adv_loss + d_f_loss

                d_loss_total.backward()
                opt_d.step()

                i += 1

        torch.save(D, 'Discriminator.model')
        torch.save(G, 'generator.model')

    if sample_task == 'developing':
        train(10, batch_size=256)

if __name__ == '__main__':
    main()
