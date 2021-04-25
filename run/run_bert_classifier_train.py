import os

#command = 'python bert_classifier.py  --task_name sst-2  --do_train  --do_lower_case  --data_dir data/sst-2/  --bert_model bert-base-uncased  --max_seq_length 64  --train_batch_size 8  --learning_rate 5e-5  --output_dir ./tmp/sst2-class/  --num_train_epochs 2 &> > GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_train.txt'

command = 'python bert_classifier.py ' \
          '--task_name sst-2 ' \
          '--do_train ' \
          '--do_lower_case ' \
          '--data_dir data/sst-2/add_1/ ' \
          '--bert_model bert-base-uncased ' \
          '--max_seq_length 64 '\
          '--train_batch_size 8 ' \
          '--learning_rate 5e-5 '\
          '--output_dir ./tmp/sst2-class/ '\
          '--num_train_epochs 2 ' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_class_train.txt '

os.system(command)