#!/usr/bin/env python3
import csv
import sys, os
import random
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import hnswlib
import numpy as np

stops = set(stopwords.words('english'))
puncs = set(string.punctuation)
forbids = set(stopwords.words('english') + list(string.punctuation))
 
try:
    import ujson as json
except:
    print('Do not find ujson, use json instead', file=sys.stderr)
    import json

available_chars = string.ascii_lowercase # string.ascii_letters # + string.digits

wordbase = [x.strip().replace(' ', '') for x in open('wordnetWords.txt', 'r')]
print('{} words.'.format(len(wordbase)))

emb_index = hnswlib.Index(space='l2', dim=300)
emb_words = []
emb_word_id = {}


#
# def valid_for_attack(x):
#     if attack_type == 'embed' and (x not in emb_word_id): return False
#     return (x.lower() not in forbids) and (not is_number(x)) and (len(x) >= 3)

def valid_for_attack(x, attack_type):
    if attack_type == 'embed' and (x not in emb_word_id): return False
    return (x.lower() not in forbids) and (not is_number(x)) and (len(x) >= 3)

def attack(text, attack_type):
    try:
        if attack_type == 'add':
            idx = random.randint(0, len(text))
            return text[:idx] + random.choice(available_chars) + text[idx:]
        elif attack_type == 'drop':
            idx = random.randint(0, len(text) - 1)
            return text[:idx] + text[idx+1:]
        elif attack_type == 'swap':
            idx = random.randint(0, len(text) - 2)
            return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
        elif attack_type == 'rand':
            return random.choice(wordbase)
        elif attack_type == 'embed':
            emb = emb_index.get_items([emb_word_id[text]])
            word_ids, _ =  emb_index.knn_query(emb, k=20)
            candidates = [emb_words[i] for i in word_ids[0] if emb_words[i] != text and (emb_words[i] not in forbids)]
            if len(candidates) == 0:
                return None
            return random.choice(candidates)

    except:
        return None
    return None

def attack_disp(text, attack_type):
    prob = np.random.random()
    flaw_label = 0
    if prob < 0.15:
        try:
            if attack_type == 'add': # Insertion
                flaw_label = 1
                idx = random.randint(0, len(text))
                return flaw_label, text[:idx] + random.choice(available_chars) + text[idx:]
            elif attack_type == 'drop': # deletion
                flaw_label = 1
                idx = random.randint(0, len(text) - 1)
                return flaw_label, text[:idx] + text[idx+1:]
            elif attack_type == 'swap': # swap
                flaw_label = 1
                idx = random.randint(0, len(text) - 2)
                return flaw_label, text[:idx] + text[idx+1] + text[idx] + text[idx+2:]
            elif attack_type == 'rand': # random
                flaw_label = 1
                return flaw_label, random.choice(wordbase)
            elif attack_type == 'embed': # embed
                flaw_label = 1
                emb = emb_index.get_items([emb_word_id[text]])
                word_ids, _ =  emb_index.knn_query(emb, k=20)
                candidates = [emb_words[i] for i in word_ids[0] if emb_words[i] != text and (emb_words[i] not in forbids)]
                if len(candidates) == 0:
                    return None
                return flaw_label, random.choice(candidates)
        except:
            return flaw_label, text
    else:
        return flaw_label, text

def valid_disp_attack(word, attack_type):
    if valid_for_attack(word, attack_type):
        return attack_disp(word, attack_type)
    else:
        return 0, word

def is_number(x):
    try:
        float(x)
        return True
    except:
        return False

if __name__ == '__main__':
    random.seed(252)
    types =  ['add', 'drop', 'swap', 'rand', 'embed']
    if len(sys.argv) < 1 + 3:
        print('--usage ./generate_char_attacks.py data_dir type times\nTypes: {}'.format(', '.join(types)), file=sys.stderr)
        sys.exit(0)

    # is_SST = 'SST' in sys.argv[1]
    # input_file = sys.argv[1] + '/test.csv'
    # if is_SST:
    #     input_file = sys.argv[1] + '/dev.tsv'

    # TODO : changed the path
    input_file = os.getcwd() + sys.argv[1] + '/dev_attacks.tsv'
    attack_type = sys.argv[2]
    times = int(sys.argv[3])

    if attack_type == 'embed':
        print('Load embeddings')
        emb_index.load_index('emb.index', max_elements = 2000000)
        with open('emb.word', 'r') as fp:
            for line in fp:
                emb_words.append(line.strip())
                emb_word_id[emb_words[-1]] = len(emb_words) - 1
        print('Done')

    output_dir = os.getcwd() + sys.argv[1] + '/{}_{}_enum/'.format(attack_type, times)
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir + '/disc_for_attacks_enum_attack.csv'
    with open(output_file, 'w') as wp:
        with open(output_file + '.num', 'w') as wpn:
            with open(input_file, 'r') as fp:
                writer = csv.writer(wp, delimiter='\t')
                reader = csv.reader(fp, delimiter='\t')
                for line in reader:
                    text, label = list(line)
                    if text == 'sentence' and label == 'label':
                        writer.writerow([text] + [str(label)])
                        continue
                    tokens = word_tokenize(text)
                    # def valid_for_attack(x):
                    #     if attack_type == 'embed' and (x not in emb_word_id): return False
                    #     return  (x.lower() not in forbids) and (not is_number(x)) and (len(x) >= 3)
                    available_positions = [i for i in range(len(tokens)) if valid_for_attack(tokens[i])]
                    random.shuffle(available_positions)
                    print(10, file=wpn)
                    for i in range(10):
                        try:
                            p = available_positions[i % len(available_positions)]
                            print("p : ", p)
                        except:
                            print(line)
                            sys.exit(0)
                        w = attack(tokens[p], attack_type)
                        attacked_pos = [str(p) if w else '-1']
                        # TODO : Might need to change the above line based on the output
                        #attacked_pos = [str(p) if w else '-1']
                        if w == None:
                            w = tokens[p]
                        new_tokens = [w if j == p else tokens[j] for j in range(len(tokens))]
                        new_text = ' '.join(new_tokens)
                        attacked_pos_lst = ','.join(attacked_pos)
                        writer.writerow([new_text] + [str(label)] + [','.join(attacked_pos)])
                        print("new_text is {} label is {} and attacked_pos is {} ".format(new_text,str(label), attacked_pos_lst))