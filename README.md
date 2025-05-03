# MixGate
## train(without mask)
torchrun --nproc_per_node=1 --master_port=29958 train_mask.py     --exp_id nomcm_0.00     --batch_size 4     --num_epochs 60     --mask_ratio 0.00     --gpus 0    --hier_tf
## train(mask)
torchrun --nproc_per_node=1 --master_port=28858 train_mask.py     --exp_id test_0.01     --batch_size 8     --num_epochs 20     --mask_ratio 0.01     --gpus 1    --hier_tf
## test(reasoning)
torchrun --nproc_per_node=1 --master_port=29958 test.py    --exp_id test_0.01     --batch_size 1     --num_epochs 20     --mask_ratio 0.01     --gpus 4    --hier_tf