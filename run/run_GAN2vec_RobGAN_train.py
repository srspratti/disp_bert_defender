import os
"""
Classifier : Eval on Test Data ( Non-Perturbed text)

dev_attacks.tsv
"""

#command = 'python bert_classifier.py  --task_name sst-2  --do_eval  --do_lower_case  --data_dir data/sst-2/add_1/dev_attacks.tsv  --bert_model bert-base-uncased  --max_seq_length 64  --eval_batch_size 8 --learning_rate 5e-5  --output_dir ./tmp/sst2-class/ > GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_dev_attacks_v2.txt'

# command = 'python bert_classifier.py ' \
#           '--task_name sst-2 ' \
#           '--do_eval ' \
#           '--do_lower_case ' \
#           '--data_dir data/sst-2/add_1/dev_attacks.tsv ' \
#           '--bert_model bert-base-uncased ' \
#           '--max_seq_length 64 '\
#           '--eval_batch_size 8 ' \
#           '--learning_rate 5e-5 '\
#           '--output_dir ./tmp/sst2-class/ '\
#           '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_dev_attacks_v2.txt '

command = 'python GAN2vec_RobGAN_train.py  ' \
          '--sample developing  ' \
          '--task_name sst-2  ' \
          '--do_train  ' \
          '--do_lower_case  ' \
          '--data_dir data/sst-2/train.tsv  ' \
          '--max_seq_length 6  ' \
          '--train_batch_size 256  ' \
          '--learning_rate 2e-5  ' \
          '--num_train_epochs 100  ' \
          '--output_dir ./tmp/disc ' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_GAN2vec_RobGAN_train_v1.txt '

os.system(command)