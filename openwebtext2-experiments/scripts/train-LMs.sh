#!/bin/bash

# Baseline T=8192
torchrun --nproc_per_node=2 ./src/main.py --config_format base --model base --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl

# Baseline T=16384
torchrun --nproc_per_node=3 ./src/main.py --config_format base --model base --n_embd 768 --n_head 12 --n_layer 12 --batch_size 6 --sequence_length 16384 --acc_steps 5 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl



# Hash-sparse LM (H-LM) for T=8192
# nb = 32
torchrun --nproc_per_node=2 ./src/main.py --config_format hashformer --model hashformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl --nb_hash 32
# nb = 16
torchrun --nproc_per_node=2 ./src/main.py --config_format hashformer --model hashformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl --nb_hash 16
# nb = 8
torchrun --nproc_per_node=2 ./src/main.py --config_format hashformer --model hashformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl --nb_hash 8

# Hash-sparse LM (H-LM) for T=16384
# nb = 32
torchrun --nproc_per_node=3 ./src/main.py --config_format hashformer --model hashformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 6 --sequence_length 16384 --acc_steps 5 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl  --nb_hash 32
# nb = 16
torchrun --nproc_per_node=3 ./src/main.py --config_format hashformer --model hashformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 6 --sequence_length 16384 --acc_steps 5 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl  --nb_hash 16
# nb = 8
torchrun --nproc_per_node=3 ./src/main.py --config_format hashformer --model hashformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 6 --sequence_length 16384 --acc_steps 5 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl  --nb_hash 8



# QK-sparse LM (D-LM) for T=8192
# s = 70
torchrun --nproc_per_node=2 ./src/main.py --config_format dropformer --model dropformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl  --sparsity 0.7
# s = 50
torchrun --nproc_per_node=2 ./src/main.py --config_format dropformer --model dropformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl  --sparsity 0.5
# s = 30
torchrun --nproc_per_node=2 ./src/main.py --config_format dropformer --model dropformer --n_embd 768 --n_head 12 --n_layer 12 --batch_size 4 --sequence_length 8192 --acc_steps 16 --dropout 0.0 --no_compile --iterations 15000 --dataset openwebtext2 --data_in_ram --lr 1e-3  --weight_decay 0.1 --eval_freq 100 --distributed_backend nccl  --sparsity 0.3
