import os

# dev_attacks.tsv drop ( deletion )

#command = 'python bert_classifier.py  --task_name sst-2  --do_train  --do_lower_case  --data_dir data/sst-2/  --bert_model bert-base-uncased  --max_seq_length 64  --train_batch_size 8  --learning_rate 5e-5  --output_dir ./tmp/sst2-class/  --num_train_epochs 2 &> > GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_train.txt'

command = 'python bert_enumerate_attacks_disp.py ' \
          ' --task_name sst-2 ' \
          ' --do_lower_case  ' \
          '--data_dir data/sst-2/add_1/dev_attacks.tsv  ' \
          '--bert_model bert-base-uncased ' \
          '--max_seq_length 128  ' \
          '--output_dir ./data/sst-2/add_1 ' \
          '--attack_type drop ' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_enumerate_attacks_drop.txt '

os.system(command)