from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
#copy-test
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from bert_model import BertForDiscriminator, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear

from bert_utils import *

def _read_tsv(cls, input_file, quotechar=None):
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
                return lines


# def get_dev_examples_for_attacks(self, data_dir):
#     if 'tsv' in data_dir:
#         return create_examples(self._read_tsv(data_dir), "dev_attacks")
#     else:
#         return create_examples(self._read_tsv(os.path.join(data_dir, "dev_attacks.tsv")), "dev")

#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         if 'tsv' in data_dir:
#             return self._create_examples(
#                 self._read_tsv(data_dir), "dev")
#         else:
#             return self._create_examples(
#                 self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")    

        
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             flaw_labels = None
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = line[0]
#             label = line[1]
#             if len(line) == 3: flaw_labels = line[2]
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
#         return examples
    
# def create_examples_for_attacks(self,lines):
#     "Create examples for attacks"
#     sentences = []
#     labels=[]
#     for (i, line) in enumerate(lines):
#         flaw_labels = None
#         if i == 0:
#             continue
#             text_a = line[0]
#             label = line[1]
#             #if len(line) == 3: flaw_labels = line[2]
#             sentences.append(text_a)
#             labels.append(label)
#         return sentences,labels
    
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             guid = "%s-%s" % (set_type, i)
#             flaw_labels = None
#             text_a = line[1]
#             label = line[0]
#             if len(line) == 3: flaw_labels = line[2]
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=None, label=label, flaw_labels=flaw_labels))
#         return examples
    
def convert_examples_to_features_disc_train(examples, label_list, max_seq_length, tokenizer, w2i={}, i2w={}, index=1):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        token_ids = []
        tokens = word_tokenize(example.text_a) # where in the code text_a was initialized with sentence of the examples
        if len(tokens) > max_seq_length:
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
    return features,w2i,i2w,index

# def convert_examples_to_features_flaw_attacks(examples, max_seq_length, max_ngram_length, tokenizer, i2w, embeddings=None,
#                                       emb_index=None, words=None):
#     """Loads a data file into a list of `InputBatch`s."""

#     features = []
#     print("examples: ", examples)

#     for (ex_index, example) in enumerate(examples):

#         tokens = example
#         print("example: ", example[0])
#         flaw_labels = []
#         flaw_tokens, flaw_pieces = [], []

#         for tok_id in tokens:
            
#             print("tok_id : ",tok_id)

#             if tok_id == 0: break

#             tok = i2w[tok_id]

#             label, tok_flaw = random_attack(tok, embeddings, emb_index, words)  # embeddings
#             word_pieces = tokenizer.tokenize(tok_flaw)

#             flaw_labels += [label] * len(word_pieces)
#             flaw_pieces += word_pieces

#             flaw_tokens.append(tok_flaw)

#             if len(flaw_pieces) > max_seq_length - 2:
#                 flaw_pieces = flaw_pieces[:(max_seq_length - 2)]
#                 flaw_labels = flaw_labels[:(max_seq_length - 2)]
#                 break

#         flaw_pieces = ["[CLS]"] + flaw_pieces + ["[SEP]"]
#         flaw_labels = [0] + flaw_labels + [0]

#         flaw_ids = tokenizer.convert_tokens_to_ids(flaw_pieces)
#         flaw_mask = [1] * len(flaw_ids)

#         padding = [0] * (max_seq_length - len(flaw_ids))
#         flaw_ids += padding
#         flaw_mask += padding
#         flaw_labels += padding

#         assert len(flaw_ids) == max_seq_length
#         assert len(flaw_mask) == max_seq_length
#         assert len(flaw_labels) == max_seq_length
#     return features

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default='/data/sst-2/dev_attacks.tsv',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='/data/sst-2/add_1/',
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
    parser.add_argument("--embedding_size",
                        default=300,
                        type=int,
                        help="Total batch size for embeddings.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task e.g. sst-2 or imdb to generate attacks")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--output_dir_attacks",
                        action='store_true',
                        help="The output directory where the .tsv with flaw_ids and falw_labels need to be created")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    args = parser.parse_args()

    ## Execution Code
    
    
    # GPU or CPU ? 
    # Comment the if else block for no CUDA
#   if args.local_rank == -1 or args.no_cuda:
#         device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
#         n_gpu = torch.cuda.device_count()
#    else:
#         torch.cuda.set_device(args.local_rank)
#         device = torch.device("cuda", args.local_rank)
#         n_gpu = 1
#         # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
#         torch.distributed.init_process_group(backend='nccl')    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # uncomment this for no GPU 
    #logger.info("device: {} , distributed training: {}, 16-bits training: {}".format(
     #   device, bool(args.local_rank != -1), args.fp16))


    # Prepare model
#     cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
#                                                                    'distributed_{}'.format(args.local_rank))
#     model = BertForDiscriminator.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)
#     model.to(device)
    
    
    #data_dir= "./data/sst-2/dev_attacks.tsv"
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    examples_for_attacks = None
    all_tokens = []
    all_label_id = []
    examples_for_attacks = processor.get_dev_examples_for_attacks(args.data_dir)
    features_for_attacks, w2i, i2w, vocab_size = convert_examples_to_features_disc_train(examples_for_attacks,
                                                                                   label_list,
                                                                                  args.max_seq_length, tokenizer)
    all_tokens = torch.tensor([f.token_ids for f in features_for_attacks], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in features_for_attacks], dtype=torch.long)
     

    data_for_attacks = TensorDataset(all_tokens, all_label_id)
    sampler_for_attacks = RandomSampler(data_for_attacks)  # for NO GPU
    # train_sampler = DistributedSampler(train_data)

    dataloader_for_attack = DataLoader(data_for_attacks, sampler=sampler_for_attacks)

    logger.info("Loading word embeddings for generating attacks... ")
    emb_dict, emb_vec, vocab_list, emb_vocab_size = load_vectors(args.word_embedding_file)
    if not os.path.exists(args.index_path):

        write_vocab_info(args.word_embedding_info, emb_vocab_size, vocab_list)
        p = load_embeddings_and_save_index(range(emb_vocab_size), emb_vec, args.index_path)
    else:
        # emb_vocab_size, vocab_list = load_vocab_info(args.word_embedding_info)
        p = load_embedding_index(args.index_path, emb_vocab_size, num_dim=args.embedding_size)
    # emb_dict, emb_vec, vocab_list, emb_vocab_size, p = None, None, None, None, None


    # examples_for_attacks = None
    # w2i, i2w, vocab_size = {}, {}, 1
    dir_path="./data/sst-2/add_1/"
    output_file = os.path.join(dir_path, "disc_for_attacks_outputs.tsv")
    print("output_file", output_file)
    flaw_ids = []
    flaw_labels = []
    all_tokens=list(all_tokens.detach().cpu().numpy())
    all_label_id=list(all_label_id.detach().cpu().numpy())
    with open(output_file, "w") as csv_file:
        for step, batch in enumerate(tqdm(dataloader_for_attack, desc="attacks")):
            
            print("STEP: SBPLSHP: ", step)
            batch = tuple(t.to(device) for t in batch)
            tokens,_ = batch #, label_id, ngram_ids, ngram_labels, ngram_masks
            tokens = tokens.to('cpu').numpy() 
            
            features_with_flaws = convert_examples_to_features_flaw_attacks(tokens,
                                                                            args.max_seq_length, args.max_ngram_length,tokenizer, i2w,
                                                                            embeddings=None, emb_index=None, words=None)
            flaw_ids = torch.tensor([f.flaw_ids for f in features_with_flaws])
            flaw_labels = torch.tensor([f.flaw_labels for f in features_with_flaws])
            print("flaw_ids: ", flaw_ids.shape)

        #for indx, item in enumerate(range(len(flaw_ids))):
            writer = csv.writer(csv_file, delimiter='\t')
            #writer.writerow(["sentence", "label"])
            flaw_ids_ar=flaw_ids.detach().cpu().numpy()
            flaw_ids_lst=flaw_ids.tolist()
            writer.writerow([all_tokens[step],all_label_id[step], flaw_ids_lst]) # need to write the token
            print("SBPLSHP all_tokens type : ", type(all_tokens))
            print("SBPLSHP all_tokens Len : ", len(all_tokens))
            print("SBPLSHP all_tokens step: ", all_tokens[step])
            print("all_label_id : type : ", type(all_label_id))
            print("all_label_id : Len : ", len(all_label_id))
            print("all_label_id :  : ", all_label_id)            
            print("flaw_ids : type : ", type(flaw_ids))
            print("flaw_ids : Len : ", len(flaw_ids))
            print("flaw_ids : : ", flaw_ids)            
            
#                         output_file = os.path.join(args.data_dir, "epoch"+str(epoch)+"disc_outputs.tsv")
#             with open(output_file,"w") as csv_file:
#                 writer = csv.writer(csv_file, delimiter='\t')
#                 writer.writerow(["sentence", "label", "ids"])

if __name__ == "__main__":
    main()



