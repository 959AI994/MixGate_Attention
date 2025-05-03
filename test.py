from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from torch import nn
import time
from progress.bar import Bar
from torch_geometric.loader import DataLoader

from config import get_parse_args
import mixgate
# from mixgate import top_model

import mixgate.top_model_rexmg as top_model

from mixgate.arch.mlp import MLP
from mixgate.utils.utils import zero_normalization, AverageMeter, get_function_acc
from mixgate.utils.logger import Logger
import torch.distributed as dist
# from .utils.attention_utils import analyze_attention,plot_attention_dist
from mixgate.utils.attention_utils import AttentionAnalyzer  # 添加导入

import numpy as np

DATA_DIR = '/home/xqgrp/wangjingxin/datasets/mixgate_data/'
MODEL_PATH = '/home/xqgrp/wangjingxin/pythonproject/MixGate/exp/nomcmxmg_0.01/nomcmxmg_0.01/model_last.pth'  

def save_attention(attention_info, batch_ids, circuit_names, attention_analyzer, model_epoch):
    """保存每50个batch的注意力信息"""
    batch_counter = 0
    batch_counter += 1  # 每次调用时增加计数器

    # 每200个batch保存一次
    if batch_counter % 50 == 0:
        for batch_id in attention_info['attentions']:
            # 获取对应数据
            attn = attention_info['attentions'][batch_id]  # [heads, seq, seq]
            tokens = attention_info['token_names'][batch_id]
            
            circuit_name = circuit_names[batch_id]

            # 生成唯一标识
            uid = f"epoch{model_epoch}_batch{batch_id}_{circuit_name}_{time.strftime('%m%d%H%M')}"
            
            # 保存为压缩npz文件
            np.savez_compressed(
                os.path.join(attention_analyzer.save_dir, f"{uid}.npz"),
                attention=attn.cpu().detach().numpy(),
                tokens=np.array(tokens)
            )

        # 保存后重置计数器
        batch_counter = 0

def test():
    # 初始化配置和模型
    args = get_parse_args()
    circuit_path = '/home/xqgrp/wangjingxin/datasets/mixgate_data/merged_all1500.npz'
    
    # 加载数据集
    print('[INFO] Loading Dataset')
    dataset = mixgate.NpzParser_Pair(DATA_DIR, circuit_path)
    _, test_dataset = dataset.get_dataset()
    
    # 创建模型并加载权重
    print('[INFO] Loading Model')
    model = top_model.TopModel(
        args,
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )
    
    # 加载训练好的权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu')['state_dict'])
        print(f'[INFO] Loaded weights from {MODEL_PATH}')
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, 
                           batch_size=args.batch_size, 
                           shuffle=False,
                           num_workers=args.num_workers)
    
    # 定义损失函数
    reg_loss = torch.nn.L1Loss()
    
    # 指标统计
    metrics = {
        'prob_loss': AverageMeter(),
        'mcm_loss': AverageMeter(),
        'func_loss': AverageMeter(),
        'prob_aig': AverageMeter(),
        'prob_mig': AverageMeter(),
        'prob_xmg': AverageMeter(),
        'prob_xag': AverageMeter()
    }
    
    # 创建AttentionAnalyzer实例用于保存注意力信息
    attention_analyzer = AttentionAnalyzer(save_dir=os.path.join(args.save_dir, 'attentions'))
    
    # 开始测试
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # 前向传播
            mcm_pm_tokens, mask_indices, pm_tokens, pm_prob, aig_prob, mig_prob, xmg_prob, xag_prob, attention_info = model(batch)
            
            # 如果模型返回了attention信息，保存它
            if attention_info is not None:
                save_attention(attention_info, batch.batch.cpu().numpy(), batch.name, attention_analyzer, model_epoch=0)

            # 计算各项损失
            prob_loss = reg_loss(pm_prob, batch['prob'].unsqueeze(1))
            mcm_loss = reg_loss(mcm_pm_tokens[mask_indices], pm_tokens[mask_indices])
            
            # 功能相似性损失
            node_a = mcm_pm_tokens[batch['tt_pair_index'][0], args.dim_hidden:]
            node_b = mcm_pm_tokens[batch['tt_pair_index'][1], args.dim_hidden:]
            emb_dis = 1 - torch.cosine_similarity(node_a, node_b, eps=1e-8)
            emb_dis_z = zero_normalization(emb_dis)
            tt_dis_z = zero_normalization(batch['tt_dis'])
            func_loss = reg_loss(emb_dis_z, tt_dis_z)
            
            # 子模型概率损失
            prob_aig = reg_loss(aig_prob, batch['aig_prob'].unsqueeze(1))
            prob_mig = reg_loss(mig_prob, batch['mig_prob'].unsqueeze(1))
            prob_xmg = reg_loss(xmg_prob, batch['prob'].unsqueeze(1))
            prob_xag = reg_loss(xag_prob, batch['xag_prob'].unsqueeze(1))
            
            # 更新统计
            metrics['prob_loss'].update(prob_loss.item())
            metrics['mcm_loss'].update(mcm_loss.item())
            metrics['func_loss'].update(func_loss.item())
            metrics['prob_aig'].update(prob_aig.item())
            metrics['prob_mig'].update(prob_mig.item())
            metrics['prob_xmg'].update(prob_xmg.item())
            metrics['prob_xag'].update(prob_xag.item())
    
    # 打印结果
    print('\n===== Test Results =====')
    print(f'Prob Loss: {metrics["prob_loss"].avg:.4f}')
    print(f'MCM Loss: {metrics["mcm_loss"].avg:.4f}')
    print(f'Func Loss: {metrics["func_loss"].avg:.4f}')
    print(f'AIG Prob Loss: {metrics["prob_aig"].avg:.4f}')
    print(f'MIG Prob Loss: {metrics["prob_mig"].avg:.4f}')
    print(f'XMG Prob Loss: {metrics["prob_xmg"].avg:.4f}')
    print(f'XAG Prob Loss: {metrics["prob_xag"].avg:.4f}')

if __name__ == '__main__':
    test()

