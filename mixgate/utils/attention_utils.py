import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm
import pandas as pd


class AttentionAnalyzer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._ensure_vis_dir()
        
    def _ensure_vis_dir(self):
        vis_dir = os.path.join(self.save_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        return vis_dir

    @staticmethod
    def load_attention(file_path):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found!")
            return None
        try:
            data = np.load(file_path, allow_pickle=True)
            return {'attention': data['attention'], 'tokens': data['tokens']}
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    # ---------------------------
    # 可视化部分
    # ---------------------------
    def visualize_average_attention(self, file_path):
        """显示所有头的平均注意力热力图"""
        data = self.load_attention(file_path)
        if data is None: 
            return
        avg_attn = data['attention'].mean(axis=0)
        vis_dir = self._ensure_vis_dir()
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(avg_attn,
                    xticklabels=data['tokens'],
                    yticklabels=data['tokens'],
                    cmap="YlGnBu")
        plt.title(f"Average Attention Map - {os.path.basename(file_path)}")
        plt.xticks(fontsize=6, rotation=90)
        plt.yticks(fontsize=6)
        save_path = os.path.join(vis_dir, f"average_{os.path.basename(file_path)}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print("Average attention visualization saved to:", save_path)

        # 新增Excel导出
        df = pd.DataFrame(avg_attn, 
                        index=data['tokens'],
                        columns=data['tokens'])
        excel_path = os.path.join(vis_dir, f"average_{os.path.basename(file_path)}.xlsx")
        df.to_excel(excel_path)
        print("Attention matrix saved to:", excel_path)


    def visualize_single_head(self, file_path, head_idx=0):
        """显示指定头的注意力热力图"""
        data = self.load_attention(file_path)
        if data is None: 
            return
        attention = data['attention']
        tokens = data['tokens']
        vis_dir = self._ensure_vis_dir()
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(attention[head_idx],
                    xticklabels=tokens,
                    yticklabels=tokens,
                    cmap="YlGnBu",
                    square=True)
        plt.title(f"Head {head_idx} Attention Map")
        plt.xticks(fontsize=6, rotation=90)
        plt.yticks(fontsize=6)
        save_path = os.path.join(vis_dir, f"head_{head_idx}_{os.path.basename(file_path)}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print("Single head visualization saved to:", save_path)

        # 新增Excel导出
        df = pd.DataFrame(attention[head_idx],
                        index=tokens,
                        columns=tokens)
        excel_path = os.path.join(vis_dir, f"head_{head_idx}_{os.path.basename(file_path)}.xlsx")
        df.to_excel(excel_path)
        print("Single head matrix saved to:", excel_path)


    def plot_attention_distribution(self, file_path, bins=50):
        """绘制注意力值的直方图"""
        data = self.load_attention(file_path)
        if data is None: 
            return
        avg_attn = data['attention'].mean(axis=0)
        vis_dir = self._ensure_vis_dir()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(avg_attn.flatten(), bins=bins, kde=True)
        plt.xlabel("Attention Value")
        plt.ylabel("Frequency")
        plt.title("Attention Value Distribution")
        save_path = os.path.join(vis_dir, f"attention_distribution_{os.path.basename(file_path)}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print("Attention distribution plot saved to:", save_path)

    # ---------------------------
    # 分析场景函数
    # ---------------------------
    def top_attention_pairs(self, file_path, top_k=5):
        """输出平均注意力矩阵中注意力值最高的 top_k 对 token 配对"""
        data = self.load_attention(file_path)
        if data is None:
            return
        avg_attn = data['attention'].mean(axis=0)
        tokens = data['tokens']
        
        flat_indices = np.argsort(avg_attn.flatten())[::-1][:top_k]
        rows, cols = np.unravel_index(flat_indices, avg_attn.shape)
        print("\n=== Top {} Attention Pairs ===".format(top_k))
        for i, (r, c) in enumerate(zip(rows, cols)):
            print(f"{i+1}. {tokens[r]} -> {tokens[c]} : {avg_attn[r, c]:.4f}")

    def analyze_token_types(self, file_path, output_file="token_analysis.txt"):
        data = self.load_attention(file_path)
        if data is None:
            return
        tokens = data['tokens']
        
        total_tokens = len(tokens)
        unique_tokens = set(tokens)
        unique_count = len(unique_tokens)
        
        # 计算每种 token 出现的次数
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # 按出现次数排序
        sorted_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 将结果写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("总 token 数量: {}\n".format(total_tokens))
            f.write("唯一 token 数量: {}\n".format(unique_count))
            f.write("\n各个 token 及其出现的次数:\n")
            for token, count in sorted_counts:
                f.write(f"{token}: {count}\n")
            
            # 统计以 "aig" 开头的 token 数量
            aig_tokens = [token for token in tokens if token.startswith("aig")]
            f.write(f"\n以 'aig' 开头的 token 数量: {len(aig_tokens)}\n")
        
        print("Token 分析结果已输出到文件: {}".format(output_file))


    def enhanced_cross_modal_analysis(self, file_path, src_modal, tgt_modal):
        data = self.load_attention(file_path)
        if data is None:
            return
        avg_attn = data['attention'].mean(axis=0)
        tokens = data['tokens']

        # 添加维度验证
        seq_len = avg_attn.shape[0]
        assert len(tokens) == seq_len, f"Token数量({len(tokens)})与矩阵维度({seq_len})不匹配！"

        src_indices = [i for i, name in enumerate(tokens) if src_modal in name]
        tgt_indices = [i for i, name in enumerate(tokens) if tgt_modal in name]

        # 添加索引范围检查
        if src_indices:
            print(f"Source indices range: {min(src_indices)}-{max(src_indices)}")
            if max(src_indices) >= seq_len:
                raise ValueError(f"发现非法源索引 {max(src_indices)} (矩阵维度 {seq_len})")
                
        if tgt_indices:
            print(f"Target indices range: {min(tgt_indices)}-{max(tgt_indices)}")
            if max(tgt_indices) >= seq_len:
                raise ValueError(f"发现非法目标索引 {max(tgt_indices)} (矩阵维度 {seq_len})")

        if not src_indices or not tgt_indices:
            print(f"无法找到 {src_modal} 或 {tgt_modal} 的对应token")
            return

        try:
            cross_attn = avg_attn[np.ix_(src_indices, tgt_indices)]
        except IndexError as e:
            print(f"索引错误详情：")
            print(f"- 源索引数量: {len(src_indices)}")
            print(f"- 目标索引数量: {len(tgt_indices)}")
            print(f"- 矩阵形状: {avg_attn.shape}")
            raise e

    def hierarchical_analysis(self, file_path, graph_suffix='_graph', subg_suffix='_subg'):
        """
        检查 token 中以 graph_suffix 结尾的部分是否对以 subg_suffix 结尾的部分有较高关注度，
        并输出关注矩阵，之后可考虑扩展成图形化展示。
        """
        data = self.load_attention(file_path)
        if data is None:
            return
        avg_attn = data['attention'].mean(axis=0)
        tokens = data['tokens']

        graph_indices = [i for i, name in enumerate(tokens) if name.endswith(graph_suffix)]
        subg_indices = [i for i, name in enumerate(tokens) if name.endswith(subg_suffix)]

        if not graph_indices or not subg_indices:
            print("No tokens match the hierarchical naming rules.")
            return

        hier_attn = avg_attn[np.ix_(graph_indices, subg_indices)]
        print("\n=== Graph Node to Subgraph Attention ===")
        print(hier_attn)

    def attention_statistics(self, file_path, threshold=0.1):
        """
        计算注意力稀疏性（小于 threshold 的比例）和注意力矩阵与转置之间的非对称性。
        同时检查每行的注意力是否归一化。
        """
        data = self.load_attention(file_path)
        if data is None:
            return
        avg_attn = data['attention'].mean(axis=0)
        
        sparsity = (avg_attn < threshold).mean()
        symmetry_diff = np.abs(avg_attn - avg_attn.T).mean()
        row_sums = avg_attn.sum(axis=-1)
        max_norm_error = np.abs(row_sums - 1).max()

        print(f"\nAttention Sparsity (values < {threshold}): {sparsity:.2%}")
        print(f"Attention Matrix Asymmetry: {symmetry_diff:.4f}")
        print(f"Maximum row normalization error: {max_norm_error:.4f}")

    def modal_comparison(self, file_path, modal_prefix='aig'):
        """
        计算以 modal_prefix 开头的 token 内部与跨模态注意力的平均值对比。
        """
        data = self.load_attention(file_path)
        if data is None:
            return
        avg_attn = data['attention'].mean(axis=0)
        tokens = data['tokens']
        modal_mask = np.array([name.startswith(modal_prefix) for name in tokens])
        if modal_mask.sum() == 0:
            print(f"No tokens found starting with {modal_prefix}")
            return
        intra_attention = avg_attn[modal_mask][:, modal_mask].mean()
        inter_attention = avg_attn[modal_mask][:, ~modal_mask].mean()
        print(f"\n{modal_prefix.upper()} intra-modal attention: {intra_attention:.4f} vs inter-modal attention: {inter_attention:.4f}")

    def batch_analyze(self, folder_path):
        """
        遍历 folder_path 中所有 .npz 文件，生成基本统计信息和 top attention pairs 报告，
        并输出到 analysis_report.txt 中。
        """
        all_files = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
        report_path = os.path.join(folder_path, "analysis_report.txt")
        with open(report_path, 'w') as report:
            for file in tqdm(all_files, desc="Batch Analyzing"):
                file_full_path = os.path.join(folder_path, file)
                data = self.load_attention(file_full_path)
                if data is None:
                    continue
                avg_attn = data['attention'].mean(axis=0)
                tokens = data['tokens']
                report.write(f"File: {file}\n")
                report.write(f"- Max Attention: {avg_attn.max():.4f}\n")
                report.write(f"- Avg Attention: {avg_attn.mean():.4f}\n")
                report.write("- Top 3 Attention Pairs:\n")
                top_indices = np.argsort(avg_attn.flatten())[-3:][::-1]
                for idx in top_indices:
                    r, c = np.unravel_index(idx, avg_attn.shape)
                    report.write(f"  {tokens[r]} -> {tokens[c]}: {avg_attn[r, c]:.4f}\n")
                report.write("\n")
        print("Batch analysis report generated at:", report_path)

    def check_attention_normalization(self, file_path):
        """验证注意力矩阵的行是否归一化"""
        data = self.load_attention(file_path)
        if data is None:
            return
        avg_attn = data['attention'].mean(axis=0)
        row_sums = avg_attn.sum(axis=-1)
        norm_error = np.abs(row_sums - 1).max()
        print(f"Maximum normalization error across rows: {norm_error:.4f}")



    def analyze_mig_masked_attention(self, file_path, target_token="xmg_0", 
                                save_prefix="xmg_analysis"):
        """分析特定节点与aig节点的注意力关系，新增热力图"""
        data = self.load_attention(file_path)
        if data is None:
            return
        
        avg_attn = data['attention'].mean(axis=0)
        tokens = data['tokens']
        vis_dir = self._ensure_vis_dir()
        
        # 目标节点索引获取
        target_indices = np.where(np.array(tokens) == target_token)[0]
        if not target_indices:
            print(f"Token '{target_token}' not found")
            return
        target_index = target_indices[0]

        # 获取所有aig节点的索引和名称
        aig_indices = [i for i, t in enumerate(tokens) if t.startswith("aig")]
        aig_names = [tokens[i] for i in aig_indices]
        
        if not aig_indices:
            print("No aig tokens found")
            return

        # 提取对应注意力值
        attention_values = avg_attn[target_index, aig_indices]
        
        # 创建DataFrame并排序
        df = pd.DataFrame({
            "source_token": [target_token]*len(aig_names),
            "target_token": aig_names,
            "attention": attention_values
        }).sort_values("attention", ascending=False)

        # 保存到Excel
        excel_path = os.path.join(vis_dir, f"{save_prefix}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Attention data saved to {excel_path}")

        # 绘制热力图（新增）
        plt.figure(figsize=(20, len(aig_names)//2))  # 动态调整高度
        sns.heatmap(
            attention_values.reshape(1, -1),  # 单行热力图
            annot=True, fmt=".2f",
            yticklabels=[target_token],
            xticklabels=aig_names,
            cmap="YlGnBu"
        )
        plt.title(f"Attention from {target_token} to AIG Tokens")
        plt.xticks(rotation=90, fontsize=6)
        
        heatmap_path = os.path.join(vis_dir, f"{save_prefix}_heatmap.png")
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Heatmap saved to {heatmap_path}")

        # 保留原有的分布图
        plt.figure(figsize=(10, 6))
        plt.figure(figsize=(10, 6))
        plt.hist(attention_values, bins=50, edgecolor='black', alpha=0.6, label="Attention values")

        dist_path = os.path.join(vis_dir, f"{save_prefix}_distribution.png")
        plt.savefig(dist_path)
        plt.close()
        print(f"Distribution plot saved to {dist_path}")


# ---------------------------
# 使用示例说明
# ---------------------------
if __name__ == "__main__":
    # 指向实际存在的 .npz 文件
    file_path = "/home/xqgrp/wangjingxin/pythonproject/MixGate/exp/04_xmg_0.01/attentions/epoch59_batch7_sort_unit_722_05030044.npz"
    
    analyzer = AttentionAnalyzer(save_dir="/home/xqgrp/wangjingxin/pythonproject/MixGate/logs_xmg_node_sort_unit_722")
    
    # 1. 基础数据检查与统计
    print("\n>> 基础数据检查与统计:")
    analyzer.check_attention_normalization(file_path)
    analyzer.attention_statistics(file_path, threshold=0.1)

    analyzer.analyze_token_types(file_path)

    # 2. 可视化部分
    print("\n>> 可视化平均注意力:")
    analyzer.visualize_average_attention(file_path)
    print("\n>> 单头注意力可视化 (Head 0):")
    analyzer.visualize_single_head(file_path, head_idx=0)
    
    print("\n>> 注意力分布直方图:")
    analyzer.plot_attention_distribution(file_path)
    
    # # 3. 关键分析场景
    print("\n>> Top Attention Pairs:")
    analyzer.top_attention_pairs(file_path, top_k=100)
    print("\n>> 增强版跨模态分析 (xmg -> aig):")
    analyzer.enhanced_cross_modal_analysis(file_path, "xmg", "aig")
    print("\n>> 层次结构分析:")
    analyzer.hierarchical_analysis(file_path, graph_suffix='_graph', subg_suffix='_subg')
    
    # # 4. 模态内 vs 跨模态比较
    print("\n>> 模态比较:")
    analyzer.modal_comparison(file_path, modal_prefix='aig')
    
    # 5. 批量处理示例（若有多个文件进行分析）
    # analyzer.batch_analyze("/home/xqgrp/wangjingxin/pythonproject/MixGate/exp/mcm_test_0.01/attentions")

    # 6. 单个掩码节点的attention weight情况
    print("\n>> 分析 'xmg_4' 节点的注意力:")
    analyzer.analyze_mig_masked_attention(file_path, target_token="xmg_4")

