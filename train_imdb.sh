#!/usr/bin/env bash

BERT_BASE_DIR='/data/checkpoints/uncased_L-12_H-768_A-12'
# BERT_BASE_DIR='/data/checkpoints/uncased_L-24_H-1024_A-16'
# DATASETS="$HOME/.data/glue_data"
DATASETS='/data/datasets'

tensorboard --logdir /data/outputs &

python run_classifier.py \
  --task_name=IMDB \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATASETS/IMDB \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3 \
  --output_dir=/data/outputs/imdb \
  $@
