import os

# disc. eval on drop attacks

 # change the name of .txt if preferred # change the data_dir path if required.

command = 'python bert_generator.py  ' \
          '--task_name sst-2  ' \
          '--do_eval  ' \
          '--do_lower_case  ' \
          '--data_dir data/sst-2/add_1/disc_eval_outputs_drop.tsv  ' \
          '--bert_model bert-base-uncased  ' \
          '--max_seq_length 64  ' \
          '--train_batch_size 8  ' \
          '--learning_rate 2e-5  ' \
          '--output_dir ./tmp/sst2-gnrt/  ' \
          '--num_eval_epochs 2  ' \
          '> GAN2vec_RobGAN_data_oup/console_outputs/console_outputs_gnrt_eval_drop_attacks_debug.txt '

os.system(command)