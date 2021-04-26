import os
"""
Classifier : Eval on Test Data ( Recovered Perturbed text)

dev_attacks.tsv attacks are add ( insertion ) 
"""

#command = 'python bert_classifier.py  --task_name sst-2  --do_eval  --do_lower_case  --data_dir data/sst-2/add_1/dev_attacks.tsv  --bert_model bert-base-uncased  --max_seq_length 64  --eval_batch_size 8 --learning_rate 5e-5  --output_dir ./tmp/sst2-class/ > GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_dev_attacks_v2.txt'

command = 'python bert_classifier.py ' \
          '--task_name sst-2 ' \
          '--do_eval ' \
          '--do_lower_case ' \
          '--data_dir data/sst-2/add_1/enum_attacks_disp/gnrt_eval_outputs_rand.tsv ' \
          '--bert_model bert-base-uncased ' \
          '--max_seq_length 64 '\
          '--eval_batch_size 8 ' \
          '--learning_rate 5e-5 '\
          '--output_dir ./tmp/sst2-class/ '\
          '--attack_type rand ' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_rand_recovered.txt '

os.system(command)