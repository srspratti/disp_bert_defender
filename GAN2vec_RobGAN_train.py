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

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert_model import BertForDiscriminator, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from Gan2vec_RobGAN_utils.defenses.scRNN.model import ScRNN
from Gan2vec_RobGAN_utils.biLstm import BiLSTM

from bert_utils import *

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

    # Parameters directly assignment in the code
    #DATA_DIR = 'data'
    DATA_DIR = './data/sst-2/train.tsv' # For debugger
    IN_TEXT = 'cleaned_haiku.data'
    IN_W2V = 'w2v_haiku.model'

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

    def get_text_from_train_examples(train_examples):
        all_text_train_examples=[]
        for example in train_examples:
            all_text_train_examples.append(example.text_a)

        return all_text_train_examples

    text = None
    encoder = None

    def get_data():


        train_examples = get_train_examples(args.data_dir)
        text = get_text_from_train_examples(train_examples)
        #print("text: ", text)
        #print("text type : ", type(text))
        #print("text len : ", len(text))

        # logger.info("Loading word embeddings ... ")
        # emb_dict, emb_vec, vocab_list, emb_vocab_size = load_vectors(args.word_embedding_file)
        # if not os.path.exists(args.index_path):
        #
        #     write_vocab_info(args.word_embedding_info, emb_vocab_size, vocab_list)
        #     encoder = load_embeddings_and_save_index(range(emb_vocab_size), emb_vec, args.index_path)
        # else:
        #     # emb_vocab_size, vocab_list = load_vocab_info(args.word_embedding_info)
        #     encoder = load_embedding_index(args.index_path, emb_vocab_size, num_dim=args.embedding_size)

        #encoder = Word2Vec.load(os.path.join(DATA_DIR, IN_W2V))
        #from gensim.models import FastText
        #sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
        logger.info("Loading word embeddings ...in Gensim format ")
        #encoder = FastText.load_fasttext_format(args.word_embedding_file)
        # TODO : Convert the text into the Gensim format
        #text_new = [ tk for txt in text for tk in txt]
        text_new = [txt.split() for txt in text]
        # encoder = FastText(text_new, min_count=1)
        #word_embedding_file
        # encoder = Word2Vec.load(os.path.join('/tmp/pycharm_project_196/GAN2vec/data/w2v_haiku.model'))
        #gensim.models.KeyedVectors.load_word2vec_format
        #encoder = gensim.models.wrappers.fasttext.FastTextKeyedVectors.load_word2vec_format(args.word_embedding_file)
        #encoder = gensim.models.KeyedVectors.load_word2vec_format(args.word_embedding_file)
        encoder = Word2Vec(text_new, min_count=1, size = 128)
        # print("encoder: ", encoder)
        # print("text_new : ", text_new)
        # print("text_new type : ", type(text_new))
        # print("text_new len : ", len(text_new))
        # for ii in range(len(text)):
        #     sentences = [x for x in text if x != ['']]
        #     text[ii] = sentences
        # all_sentences = []
        # for txt in text:
        #     all_sentences += txt
        #sentences
        #print("all_sentences: ", all_sentences)
        # print("text : ", text)
        # print("text type : ", type(text))
        # print("text len : ", len(text))
        #encoder = FastText(all_sentences, min_count=1)
        #encoder = FastText(text, min_count=1)
        # encoder = FastText(text_new, min_count=1)
        # #encoder = Word2Vec.load(os.path.join('/tmp/pycharm_project_196/GAN2vec/data/w2v_haiku.model'))
        # print("encoder: ", encoder)
        return text_new, encoder
    
    def get_lines_old(start, end, text, encoder):
        #text, encoder = get_data()
        text = text
        encoder = encoder

        seq_lens = []
        sentences = []
        longest = 0
        #print("printing start: ", start)
        #print("printing end: ", end)
        text_batch = []
        for i in range((end-start)):
            text_batch.append(text[i])
        #print("Printing Text Batch: ", text_batch)
        #print("Printing Text Batch: len ", len(text_batch))
        for l in text_batch :
            #print("l in : ",l)
            seq_lens.append(len(l))
            longest = len(l) if len(l) > longest else longest
            #longest = args.max_seq_length

            sentence = []
            #print("encoder : ", encoder)
            #for txt in l.split():
            for txt in l:
                #print(" txt : ", txt)
                #print("encoder.wv[txt]) :", encoder.wv[txt])
                #print("encoder.wv[txt]) type :", type(encoder.wv[txt]))
                #print("encoder.wv[txt]) shape :", encoder.wv[txt].shape)
                sentence.append(torch.tensor(encoder.wv[txt]))
                #print(" sentence len : ", len(sentence))
                #print(" sentence type : ", type(sentence))

            #print("sentence type of : ", type(sentence))
            #print("sentences len : ", len(sentences))
            #print("sentences type : ", type(sentence))
            sentences.append(torch.stack(sentence).unsqueeze(0))

        # Pad input
        d_size = sentences[0].size(2)
        #print("sentences: ", type(sentences))
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
        #print("seq: type ", type(seq))
        #print("seq: len ", len(seq))
        #print("seq:  ", seq)
        #print("seq:  shape ", seq.shape)
        start_words = seq[:, 0:1, :]
        packer = pack_padded_sequence(
            seq,
            seq_lens,
            batch_first=True,
            enforce_sorted=False
        )

        # print("packer type of : ", type(packer))
        # print("start words type of : ", type(start_words))
        #print("packer type of : ", type(packer))
        #print("packer type of : shape ", packer.shape)
        #print("start words type of : ", type(start_words))
        #print("start words type of : shape ", start_words.shape)
        return packer, start_words

    def get_lines_old_old(start, end):
        text, encoder = get_data()

        seq_lens = []
        sentences = []
        longest = 0
        print("printing start: ", start)
        print("printing end: ", end)
        text_batch = []
        for i in range((end-start)):
            text_batch.append(text[i])
        print("Printing Text Batch: ", text_batch)
        for l in text_batch :
            print("l in : ",l)
            seq_lens.append(len(l))
            longest = len(l) if len(l) > longest else longest
            #longest = args.max_seq_length

            sentence = []
            print("encoder : ", encoder)
            #for txt in l.split():
            for txt in l:
                #print(" txt : ", txt)
                #print("encoder.wv[txt]) :", encoder.wv[txt])
                #print("encoder.wv[txt]) type :", type(encoder.wv[txt]))
                #print("encoder.wv[txt]) shape :", encoder.wv[txt].shape)
                sentence.append(torch.tensor(encoder.wv[txt]))
                print(" sentence len : ", len(sentence))
                print(" sentence type : ", type(sentence))

            #print("sentence type of : ", type(sentence))
            print("sentences len : ", len(sentences))
            print("sentences type : ", type(sentence))
            sentences.append(torch.stack(sentence).unsqueeze(0))

        # Pad input
        d_size = sentences[0].size(2)
        print("sentences: ", type(sentences))
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
        print("seq:  shape ", seq.shape)
        start_words = seq[:, 0:1, :]
        packer = pack_padded_sequence(
            seq,
            seq_lens,
            batch_first=True,
            enforce_sorted=False
        )

        print("packer type of : ", type(packer))
        print("packer type of : shape ", packer.shape)
        print("start words type of : ", type(start_words))
        print("start words type of : shape ", start_words.shape)
        return packer, start_words

    def get_lines(start, end, text, encoder):
        #text, encoder = get_data()
        text = text
        encoder = encoder

        seq_lens = []
        sentences = []
        longest = 0
        #print("printing start: ", start)
        #print("printing end: ", end)
        text_batch = []
        for i in range((end-start)):
            text_batch.append(text[i])
        #print("Printing Text Batch: ", text_batch)
        #print("Printing Text Batch: len ", len(text_batch))
        for l in text_batch :
            #print("l in : ",l)
            seq_lens.append(len(l))
            longest = len(l) if len(l) > longest else longest
            #longest = args.max_seq_length

            sentence = []
            #print("encoder : ", encoder)
            #for txt in l.split():
            for txt in l:
                #print(" txt : ", txt)
                #print("encoder.wv[txt]) :", encoder.wv[txt])
                #print("encoder.wv[txt]) type :", type(encoder.wv[txt]))
                #print("encoder.wv[txt]) shape :", encoder.wv[txt].shape)
                sentence.append(torch.tensor(encoder.wv[txt]))
                #print(" sentence len : ", len(sentence))
                #print(" sentence type : ", type(sentence))

            #print("sentence type of : ", type(sentence))
            #print("sentences len : ", len(sentences))
            #print("sentences type : ", type(sentence))
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
        #print("seq: len ", len(seq))
        #print("seq:  ", seq)
        print("seq:  shape ", seq.shape)
        print("Seq_lens: ", seq_lens)
        start_words = seq[:, 0:1, :]
        #start_words = seq[:, :, :]
        
        #for idx in range(len(seq_lens)):
        #  start_words = seq[:,0:(seq_lens[idx]-1), :]
        packer = pack_padded_sequence(
            seq,
            seq_lens,
            batch_first=True,
            enforce_sorted=False
        )

        # print("packer type of : ", type(packer))
        # print("start words type of : ", type(start_words))
        #print("packer type of : ", type(packer))
        #print("packer type of : shape ", packer.shape)
        #print("start words type of : ", type(start_words))
        #print("start words type of : shape ", start_words.shape)
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


    torch.set_num_threads(16)

    sample_task = args.sample.lower()
    print("sample_task :", sample_task)

    def train_old_old(epochs, batch_size=3, latent_size=256, K=1):
        #text, encoder = get_data()
        num_samples = len(text)


        G = Generator(64, 64)
        D = Discriminator(64)

        l2 = nn.MSELoss()
        loss = nn.BCELoss()
        loss_ce = nn.CrossEntropyLoss()
        opt_d = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
        opt_g = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))

        for e in range(int(args.num_train_epochs)):
            i = 0
            while batch_size * i < num_samples:
                stime = time.time()

                start = batch_size * i
                end = min(batch_size * (i + 1), num_samples)
                bs = end - start

                # Use lable smoothing
                tl = torch.full((bs, 1), 0.9)
                fl = torch.full((bs, 1), 0.1)

                real, greal = get_lines(start, end)

                print("real: ", real)
                #print("greal: ", greal)

                # Train Generator as per RobGAN
                # Train generator
                for _ in range(K):
                    opt_g.zero_grad()

                    # GAN fooling ability
                    fake = G(greal)
                    g_loss = loss(D(fake), tl)
                    g_loss.backward()
                    opt_g.step()

                g_loss = g_loss.item()

                # Train descriminator
                opt_d.zero_grad()
                fake = G(greal)

                r_loss = loss(D(real), tl)
                f_loss = loss(D(fake), fl)

                # Adversarial attack
                # gadv = get_adv_lines(start, end) # TODO : adversarial attacks def

                # a_loss = loss_ce(D(gadv),t1)  # TODO : defining a loss function

                # r_loss.backward()
                # f_loss.backward()
                # d_loss = (r_loss.mean().item() + f_loss.mean().item()) / 2

                # TODO : Adding up all the loses and including it in backprop
                # d_total_loss = r_loss + f_loss + a_loss
                # d_total_loss.backward()

                opt_d.step()

                i += 1

            if e % 10 == 0:
                torch.save(G, 'generator.model')
        torch.save(G, 'generator.model')

    def train_old(epochs, batch_size=256, latent_size=256, K=1):
        text, encoder = get_data()
        num_samples = len(text)

        #get_data()
        # print("text type : ", type(text))
        # print("text: ", text)
        # print("text len : ", len(text))
        #num_samples = len(text)
        # print("num_samples: ", num_samples)

        G = Generator(64, 64)
        D = Discriminator(64)

        l2 = nn.MSELoss()
        loss = nn.BCELoss()
        opt_d = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
        opt_g = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
        
        #print("batch: ", batch_size)
        #print("num of samples: ", num_samples)
        #print("num of epochs: ", epochs)
        for e in range(epochs):
            i = 0
            while batch_size * i < num_samples:
                stime = time.time()

                start = batch_size * i
                end = min(batch_size * (i + 1), num_samples)
                bs = end - start

                # print("start: ", start)
                # print("end: ", end)
                # print("bs: ", bs)

                # Use lable smoothing
                tl = torch.full((bs, 1), 0.9)
                fl = torch.full((bs, 1), 0.1)

                # Train descriminator
                opt_d.zero_grad()
                #real, greal = get_lines(start, end)
                #real, greal = get_lines(0, 2)
                real, greal = get_lines(start, end, text, encoder)
                # print("real: ", real)
                # print("real: type:  ", type(real))
                # print("real: shape:  ", len(real))
                # print("greal: ", greal)
                # print("greal: shape : ", greal.shape)
                # print("greal: type : ", type(greal))
                
                fake = G(greal)
                # print("fake: ", fake)
                # print("fake: shape : ", fake.shape)
                # print("fake: type : ", type(fake))

                # print("D(real): ", D(real))
                # print("t1 : ", tl)
                # print("t1 : shape ", tl.shape)
                # print("f1 : ", fl)
                # print("f1 : shape ", fl.shape)
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

    def train(epochs, batch_size=256, latent_size=256, K=1):
        text, encoder = get_data()
        num_samples = len(text)

        # get_data()
        # print("text type : ", type(text))
        # print("text: ", text)
        # print("text len : ", len(text))
        # num_samples = len(text)
        # print("num_samples: ", num_samples)

        G = Generator(128, 128)
        D = Discriminator(128)

        #G = BiLSTM
        #D = BertForDiscriminator
        #D = ScRNN

        l2 = nn.MSELoss()
        loss = nn.BCELoss()
        #loss = nn.CrossEntropyLoss()
        opt_d = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))
        opt_g = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))

        # print("batch: ", batch_size)
        # print("num of samples: ", num_samples)
        # print("num of epochs: ", epochs)
        max_seq_len = args.max_seq_length
        for e in range(epochs):
            i = 0
            while batch_size * i < num_samples:
                stime = time.time()

                start = batch_size * i
                end = min(batch_size * (i + 1), num_samples)
                bs = end - start

                # print("start: ", start)
                # print("end: ", end)
                # print("bs: ", bs)

                # Use lable smoothing
                tl = torch.full((bs, 1), 0.9)
                fl = torch.full((bs, 1), 0.1)

                # Label smoothing for word-level
                #tl = torch.full((bs, max_seq_len), 0.9)
                #fl = torch.full((bs, max_seq_len), 0.1)

                # Train descriminator
                opt_d.zero_grad()
                # real, greal = get_lines(start, end)
                # real, greal = get_lines(0, 2)
                real, greal = get_lines(start, end, text, encoder)
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

                print("D(real): ", D(real))
                print("t1 : ", tl)
                print("t1 : shape ", tl.shape)
                print("f1 : ", fl)
                print("f1 : shape ", fl.shape)
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

    if sample_task == 'developing':
        train(100, batch_size=256)

if __name__ == '__main__':
    main()
    #get_sample_data()
    #train(1000, batch_size=256)
    #train(2, batch_size=5)
