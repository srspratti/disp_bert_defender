import os

# disc. eval on drop attacks

 # change the name of .txt if preferred # change the data_dir path if required.
command = 'python bert_discriminator.py  ' \
          '--task_name sst-2  ' \
          '--do_eval  ' \
          '--eval_batch_size 32  ' \
          '--do_lower_case  ' \
          '--data_dir ./data/sst-2/add_1/  ' \
          '--data_file ./data/sst-2/add_1/enum_attacks_disp/disc_enum_swap_base.tsv  ' \
          '--bert_model bert-base-uncased  ' \
          '--max_seq_length 128  ' \
          '--train_batch_size 16  ' \
          '--learning_rate 2e-5  ' \
          '--num_train_epochs 5  ' \
          '--output_dir ./models/  ' \
          '--single' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_discrim_eval_swap_attacks.txt '

os.system(command)