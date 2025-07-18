'''
python /home/cunyuliu/blt-main/bytelatent/analyze_sequence_properties.py \
    --input_file /home/cunyuliu/blt-main/tmp/generation_results/generated_sequences.txt \
    --train_file /home/cunyuliu/train_data/splited_val_train_fortrain/mergedByCode_new_train.arrow \
    --output_dir /home/cunyuliu/blt-main/bytelatent/analysis_results_sampledd \
    --train_sample_size 10
'''

import argparse
import logging
import math
import os
import random
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

# --- 生物信息学与统计库 ---
import RNA
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt

# --- 绘图库 ---
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import track

# --- 日志设置 ---
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def read_sequences_from_file(file_path: Path) -> list[str]:
    """从TXT文件中读取序列，假定每行一个序列。"""
    if not file_path.exists():
        logger.error(f"文件未找到: {file_path}")
        return []
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    logger.info(f"从TXT文件 {file_path} 中成功读取 {len(sequences)} 条序列。")
    return sequences


def load_training_set(file_path: Path) -> list[str]:
    """
    从文件加载训练集序列。
    支持 .feather, .arrow, 和 .txt 格式。
    """
    if not file_path.exists():
        logger.error(f"训练集文件未找到: {file_path}")
        return []

    # 根据文件扩展名选择加载方式
    suffix = file_path.suffix.lower()
    if suffix in ['.feather', '.arrow']:
        logger.info(f"检测到 Feather 文件格式。正在从 {file_path} 加载...")
        try:
            train_df = feather.read_table(file_path).to_pandas()
            if 'text' not in train_df.columns:
                logger.error(f"错误: Feather 文件 {file_path} 中未找到 'text' 列。")
                return []
            sequences = train_df['text'].dropna().tolist()
            logger.info(f"从 Feather 文件 {file_path} 的 'text' 列成功读取 {len(sequences)} 条序列。")
            return sequences
        except Exception as e:
            logger.error(f"读取 Feather 文件 {file_path} 时发生错误: {e}")
            return []
    elif suffix == '.txt':
        return read_sequences_from_file(file_path)
    else:
        logger.error(f"不支持的训练集文件格式: '{suffix}'。请使用 .feather, .arrow, 或 .txt 文件。")
        return []


def plot_frequency_distributions(user_freqs: dict, train_freqs: dict, output_path: str):
    """绘制并保存碱基和二联碱基频率的对比条形图。"""
    logger.info("正在绘制单核苷酸和二核苷酸的频率分布图...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Nucleotide Frequency Distributions', fontsize=16)

    # --- 1. 碱基频率 ---
    base_labels = sorted(list(set(user_freqs['base'].keys()) | set(train_freqs['base'].keys())))
    user_base_vals = [user_freqs['base'].get(b, 0) / (user_freqs['total_len'] or 1) for b in base_labels]
    train_base_vals = [train_freqs['base'].get(b, 0) / (train_freqs['total_len'] or 1) for b in base_labels]
    
    base_data = {
        'Base': base_labels * 2,
        'Frequency': user_base_vals + train_base_vals,
        'Set': ['User Sequences'] * len(base_labels) + ['Training Set'] * len(base_labels)
    }
    
    sns.barplot(x='Base', y='Frequency', hue='Set', data=base_data, ax=axes[0])
    axes[0].set_title('Single Nucleotide Frequencies')
    axes[0].set_ylabel('Frequency')

    # --- 2. 二联碱基频率 ---
    dinu_labels = sorted(list(set(user_freqs['dinucleotide'].keys()) | set(train_freqs['dinucleotide'].keys())))
    user_dinu_vals = [user_freqs['dinucleotide'].get(d, 0) / (user_freqs['total_dinucleotides'] or 1) for d in dinu_labels]
    train_dinu_vals = [train_freqs['dinucleotide'].get(d, 0) / (train_freqs['total_dinucleotides'] or 1) for d in dinu_labels]
    
    dinu_data = {
        'Dinucleotide': dinu_labels * 2,
        'Frequency': user_dinu_vals + train_dinu_vals,
        'Set': ['User Sequences'] * len(dinu_labels) + ['Training Set'] * len(dinu_labels)
    }
    
    sns.barplot(x='Dinucleotide', y='Frequency', hue='Set', data=dinu_data, ax=axes[1])
    axes[1].set_title('Dinucleotide Frequencies')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    logger.info(f"频率分布图已保存至: {output_path}")


def plot_property_distributions(user_props: dict, training_props: dict, output_path: str):
    """绘制并保存用户数据与训练数据理化性质的分布对比图。"""
    logger.info("正在绘制理化性质分布对比图...")
    common_keys = sorted(list(set(user_props.keys()) & set(training_props.keys())))
    if not common_keys:
        logger.warning("没有共同的理化性质可供绘图。")
        return

    n_plots = len(common_keys)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()
    fig.suptitle('Physicochemical Property Distributions', fontsize=16)

    for i, key in enumerate(common_keys):
        # 替换key中的下划线和首字母大写，用于标题显示
        title_key = key.replace("_", " ").title()
        if key == 'tm':
            title_key = 'Melting Temperature (Tm)'
        elif key == 'gc_content':
            title_key = 'GC Content'
        
        sns.kdeplot(user_props[key], ax=axes[i], label="User Sequences", fill=True, alpha=0.5, warn_singular=False)
        sns.kdeplot(training_props[key], ax=axes[i], label="Training Set", fill=True, alpha=0.5, warn_singular=False)
        axes[i].set_title(f'Distribution of {title_key}')
        axes[i].set_xlabel(key)
        axes[i].set_ylabel('Density')
        axes[i].legend()

    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
    logger.info(f"理化性质分布图已保存至: {output_path}")


def calculate_single_rna_properties(seq: str) -> dict | None:
    """(用于并行化)计算单个RNA序列的多种理化性质。"""
    if not isinstance(seq, str) or len(seq) < 2:
        return None
        
    seq_upper = seq.upper().replace("T", "U")
    seq_upper = seq_upper.replace("N", "")

    if not all(c in "ACGU" for c in seq_upper):
        offenders = sorted(list(set(c for c in seq_upper if c not in "ACGU")))
        offender_str = ", ".join(offenders)
        logger.warning(
            f"序列 '{seq[:10]}...' 因包含无法处理的字符 '{offender_str}' 而被跳过。"
        )
        return None

    properties = {}
    seq_len = len(seq_upper)
    
    # --- 结构性质 (使用ViennaRNA) ---
    try:
        # 使用 fold_compound 对象进行更高级的计算
        fc = RNA.fold_compound(seq_upper)
        
        # 1. 计算 MFE 和相关属性
        (structure_mfe, mfe) = fc.mfe()
        properties["mfe_normalized"] = mfe / seq_len
        paired_bases = structure_mfe.count('(') + structure_mfe.count(')')
        properties["structuredness"] = paired_bases / seq_len
        
        # 2. 计算质心结构和编辑距离
        fc.pf()  # 计算分区函数，为质心计算做准备
        (structure_centroid, _) = fc.centroid()
        
        # 计算MFE结构与质心结构之间的碱基对距离
        distance = RNA.bp_distance(structure_mfe, structure_centroid)
        properties["structure_edit_distance"] = distance

    except Exception as e:
        logger.debug(f"RNA 属性计算失败，序列: '{seq_upper}', 错误: {e}")
        properties["mfe_normalized"] = None
        properties["structuredness"] = None
        properties["structure_edit_distance"] = None

    # --- 组成性质 ---
    properties["gc_content"] = (seq_upper.count('G') + seq_upper.count('C')) / seq_len
    
    counts = Counter(seq_upper)
    entropy = 0.0
    for base in counts:
        p = counts[base] / seq_len
        entropy -= p * math.log2(p)
    properties["sequence_entropy"] = entropy

    # --- 热力学性质 (使用BioPython) ---
    try:
        properties["tm"] = mt.Tm_NN(Seq(seq_upper), nn_table=mt.R_DNA_NN1)
    except Exception:
        properties["tm"] = None

    # --- 用于后续聚合的频率计数 ---
    properties["base_counts"] = Counter(seq_upper)
    properties["dinucleotide_counts"] = Counter([seq_upper[i:i+2] for i in range(seq_len - 1)])
    
    return properties


def calculate_rna_properties(sequences: list[str]) -> tuple[dict, dict]:
    """为一批RNA序列并行计算多种性质，并聚合结果。"""
    dist_properties = defaultdict(list)
    freq_properties = {
        "base": Counter(),
        "dinucleotide": Counter(),
        "total_len": 0,
        "total_dinucleotides": 0
    }
    
    num_processes = 100
    logger.info(f"使用 {num_processes} 个CPU核心进行并行计算...")

    with Pool(processes=num_processes) as pool:
        results = list(track(
            pool.imap_unordered(calculate_single_rna_properties, sequences, chunksize=100),
            description="正在计算理化性质...",
            total=len(sequences)
        ))

    for res in results:
        if res:
            # --- 修改点: 更新收集的属性列表 ---
            # 移除 'mfe'，新增 'structure_edit_distance'
            prop_keys = [
                "gc_content", "sequence_entropy", "mfe_normalized", 
                "structuredness", "tm", "structure_edit_distance"
            ]
            for key in prop_keys:
                if res.get(key) is not None:
                    dist_properties[key].append(res[key])
            
            freq_properties["base"].update(res["base_counts"])
            freq_properties["dinucleotide"].update(res["dinucleotide_counts"])
            seq_len = sum(res["base_counts"].values())
            freq_properties["total_len"] += seq_len
            freq_properties["total_dinucleotides"] += seq_len - 1

    return dict(dist_properties), freq_properties


def main():
    """主函数：解析参数，执行分析和绘图。"""
    parser = argparse.ArgumentParser(
        description="比较两组序列的理化性质和核苷酸频率。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_file", 
        type=Path, 
        required=True, 
        help="包含待分析序列的TXT文件路径 (每行一个序列)。"
    )
    parser.add_argument(
        "--train_file", 
        type=Path, 
        required=True, 
        help="""包含训练集序列的文件路径。
支持以下格式:
- .txt (每行一个序列)
- .feather/.arrow (包含一个名为 'text' 的列)"""
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        required=True, 
        help="用于保存输出图表的目录。"
    )
    parser.add_argument(
        "--train_sample_size",
        type=int,
        default=None,
        help="从训练集中随机采样的序列数量以提高效率。如果未指定，则使用全部序列。"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- 开始读取序列 ---")
    user_sequences = read_sequences_from_file(args.input_file)
    train_sequences = load_training_set(args.train_file)

    if not user_sequences or not train_sequences:
        logger.error("一个或两个输入文件为空或无法读取，程序终止。")
        return

    if args.train_sample_size:
        total_size = len(train_sequences)
        if args.train_sample_size < total_size:
            logger.info(f"训练集共包含 {total_size} 条序列。将从中随机采样 {args.train_sample_size} 条进行分析...")
            random.seed(42)
            train_sequences = random.sample(train_sequences, args.train_sample_size)
        else:
            logger.info(f"请求的采样数量 ({args.train_sample_size}) 大于或等于训练集总数 ({total_size})。将使用全部序列。")

    logger.info("\n--- 开始计算用户序列的性质 ---")
    user_dist_props, user_freq_props = calculate_rna_properties(user_sequences)
    
    logger.info("\n--- 开始计算训练集序列的性质 ---")
    train_dist_props, train_freq_props = calculate_rna_properties(train_sequences)

    logger.info("\n--- 开始生成对比图表 ---")
    
    props_plot_path = args.output_dir / "physicochemical_properties_distribution.png"
    plot_property_distributions(user_dist_props, train_dist_props, str(props_plot_path))

    freq_plot_path = args.output_dir / "frequency_distribution.png"
    plot_frequency_distributions(user_freq_props, train_freq_props, str(freq_plot_path))

    logger.info("\n✅ 分析完成！所有图表已保存至: %s", args.output_dir)


if __name__ == "__main__":
    main()
