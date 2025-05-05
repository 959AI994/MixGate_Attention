from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mixgate
import torch
import os
from config import get_parse_args
from mixgate.single_parser import NpzParser
import mixgate.top_model_rexmg as top_model
import mixgate.top_model_hier_tf as top_model_hier_tf
# import mixgate.top_model as top_model
import mixgate.top_trainer 
import torch.distributed as dist
import torch.nn as nn

DATA_DIR = './data/'

def check_relationship(A, B):
    set_A = set(A)
    set_B = set(B)

    if set_A.issubset(set_B):
        return 'in', 0  # A 中的所有元素都在 B 中
    elif set_A & set_B:
        return 'overlap', 1  # A 和 B 有交集，但 A 不是 B 的子集
    else:
        return 'out', 2  # A 和 B 没有任何交集


if __name__ == '__main__':
    args = get_parse_args()
    circuit_path ='./dataset/npz_dataset/alu_DW02_mult_0_206.npz'           # Change this to your circuit path
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = NpzParser(DATA_DIR, circuit_path)
    # dataset = mixgate.AigParser(DATA_DIR, circuit_path)
    train_dataset, val_dataset = dataset.get_dataset()
    g = val_dataset[0]
    
    # Get LUT ID Map
    lut_txt_path = './dataset/txt/{}.txt'.format(g.name)
    aig_lutid = {}
    mig_lutid = {}
    xmg_lutid = {}
    xag_lutid = {}
    f = open(lut_txt_path, 'r')
    lines = f.readlines()
    f.close()
    flag = 'aig'
    for line in lines:
        if 'aig' in line:
            flag = 'aig'
            for i in range(g.aig_x.shape[0]):
                aig_lutid[i] = []
            continue
        if 'mig' in line:
            flag = 'mig'
            for i in range(g.mig_x.shape[0]):
                mig_lutid[i] = []
            continue
        if 'xmg' in line:
            flag = 'xmg'
            for i in range(g.x.shape[0]):
                xmg_lutid[i] = []
            continue
        if 'xag' in line:
            flag = 'xag'
            for i in range(g.xag_x.shape[0]):
                xag_lutid[i] = []
            continue
        if ':' in line:
            lut_id = int(line.split(':')[0])
            arr = line.split(':')[1].replace('\n', '').split(' ')[1:]
            for node_id in arr:
                node_id = int(node_id)
                if flag == 'aig':
                    aig_lutid[node_id].append(lut_id)
                elif flag == 'mig':
                    mig_lutid[node_id].append(lut_id)
                elif flag == 'xmg':
                    xmg_lutid[node_id].append(lut_id)
                elif flag == 'xag':
                    xag_lutid[node_id].append(lut_id)


    print('[INFO] Create Model and Trainer')
    model = top_model.TopModel(
        args, 
        # dc_ckpt='./ckpt/dc.pth', 
        dg_ckpt_aig='./ckpt/model_func_aig.pth',
        dg_ckpt_xag='./ckpt/model_func_xag.pth',
        dg_ckpt_xmg='./ckpt/model_func_xmg.pth',
        dg_ckpt_mig='./ckpt/model_func_mig.pth'
    )
    
    model.load("./model_last.pth")
    print(f'[INFO] Loaded weights from {"./model_last.pth"}')

    mcm_pm_tokens, mask_indices, pm_tokens, pm_prob,aig_prob, mig_prob, xmg_prob, xag_prob,attention_info = model(
        g, node_lutid_map = {
            'aig': aig_lutid,
            'mig': mig_lutid,
            'xmg': xmg_lutid,
            'xag': xag_lutid
        }
    )
    
    # Analysis
    reg_loss = nn.L1Loss()
    # aig_prob_loss = reg_loss(aig_prob, g.aig_prob)
    
    token_name = model.mask_tf.token_name_buffer[0]
    attn_score = model.mask_tf.attention_buffer[0]
    token_lutid = model.mask_tf.token_lutid_buffer[0]
    
    for token_k, token in enumerate(token_name):
        if 'xmg' in token and len(token_lutid[token_k]) > 0:
            node_id = int(token.split('_')[1])
            print('Token Name: {}, LUT ID: {}'.format(token, token_lutid[token_k]))
            for hier_key in ['h', 's', 'g']:
                in_attn_score = 0
                out_attn_score = 0 
                olp_attn_score = 0
                in_attn_cnt = 0
                out_attn_cnt = 0
                olp_attn_cnt = 0
                for p_k, p_name in enumerate(token_name):
                    if '_{}'.format(hier_key) in p_name:
                        _, rel = check_relationship(token_lutid[token_k], token_lutid[p_k])
                        if rel == 0:
                            in_attn_score += attn_score[:, token_k, p_k].sum().item()
                            in_attn_cnt += 1
                        elif rel == 1:
                            olp_attn_score += attn_score[:, token_k, p_k].sum().item()
                            olp_attn_cnt += 1
                        elif rel == 2:
                            out_attn_score += attn_score[:, token_k, p_k].sum().item()
                            out_attn_cnt += 1
                print('Level: {}'.format(hier_key))
                print('In: {:.4f}, Out: {:.4f}, Overlap: {:.4f}'.format(in_attn_score, out_attn_score, olp_attn_score))
                print('In Count: {}, Overlap Count: {}, Out Count: {}'.format(in_attn_cnt, olp_attn_cnt, out_attn_cnt))
                print('In Ratio: {:.2f}%, Overlap Ratio: {:.2f}%, Out Ratio: {:.2f}%'.format(
                    in_attn_score * 100 / (in_attn_score + out_attn_score + olp_attn_score),
                    olp_attn_score * 100 / (in_attn_score + out_attn_score + olp_attn_score), 
                    out_attn_score * 100 / (in_attn_score + out_attn_score + olp_attn_score)
                ))
                in_avg = in_attn_score / in_attn_cnt if in_attn_cnt > 0 else 0
                olp_avg = olp_attn_score / olp_attn_cnt if olp_attn_cnt > 0 else 0
                out_avg = out_attn_score / out_attn_cnt if out_attn_cnt > 0 else 0
                print('In Avg: {:.4f}, Overlap Avg: {:.4f}, Out Avg: {:.4f}'.format(
                    in_avg, olp_avg, out_avg
                ))
                if in_avg > olp_avg and olp_avg > out_avg:
                    print('In > Overlap > Out')
                print()
            print('='*20)
            print()
    
    print()
    
    
    