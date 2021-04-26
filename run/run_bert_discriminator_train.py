import os

# disc. eval on drop attacks

 # change the name of .txt if preferred # change the data_dir path if required.

command = 'python bert_discriminator.py  ' \
          '--task_name sst-2  ' \
          '--do_train  ' \
          '--do_lower_case  ' \
          '--data_dir data/sst-2/train.tsv  ' \
          '--bert_model bert-base-uncased  ' \
          '--max_seq_length 128  ' \
          '--train_batch_size 8  ' \
          '--learning_rate 2e-5  ' \
          '--num_train_epochs 25  ' \
          '--output_dir ./tmp/disc/' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_discrim_train.txt '


os.system(command)