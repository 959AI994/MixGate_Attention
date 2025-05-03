from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
from config import get_parse_args
import mixgate.top_model_rexmg as top_model
import mixgate.top_model_hier_tf as top_model_hier_tf
# import mixgate.top_model as top_model
import mixgate.top_trainer 
import torch.distributed as dist

DATA_DIR = '/home/xqgrp/wangjingxin/datasets/mixgate_data/'

if __name__ == '__main__':
    args = get_parse_args()
    # here,we need to build some npz formate including mig,xmg,xag,aig fusion graph
    circuit_path ='/home/xqgrp/wangjingxin/datasets/mixgate_data/merged_all1500.npz'
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    # dataset = mixgate.AigParser(DATA_DIR, circuit_path)

    train_dataset, val_dataset = dataset.get_dataset()

    print('[INFO] Create Model and Trainer')
    model = top_model.TopModel(
        args, 
        # dc_ckpt='./ckpt/dc.pth', 
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )

    trainer = mixgate.top_trainer.TopTrainer(args, model, distributed=True)
    trainer.set_training_args(lr=1e-3, lr_step=50, loss_weight = [0.0, 1.0, 0.0])
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)

    # 保存第一阶段训练结束后的权重
    trainer.save(os.path.join(trainer.log_dir, 'stage1_model.pth'))
    # 加载第一阶段的模型权重
    print('[INFO] Loading Stage 1 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage1_model.pth'))

    trainer.set_training_args(lr=1e-3, lr_step=50, loss_weight = [5.0, 0.1, 0.0])
    print('[INFO] Stage 2 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)

    # 保存第一阶段训练结束后的权重
    trainer.save(os.path.join(trainer.log_dir, 'stage2_model.pth'))
    # 加载第一阶段的模型权重
    print('[INFO] Loading Stage 2 Checkpoint...')
    trainer.load(os.path.join(trainer.log_dir, 'stage2_model.pth'))


    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight = [5.0, 0.1, 2.0])
    print('[INFO] Stage 3 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)

    
    