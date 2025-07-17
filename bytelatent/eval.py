# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import math
import os
import random
from collections import defaultdict, Counter
from datetime import datetime
from multiprocessing import Pool, cpu_count

# --- 环境设置 ---
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# --- 标准库和第三方库导入 ---
import itertools
import Levenshtein
import pandas as pd
import numpy as np
import pyarrow.feather as feather
import torch
import torch.distributed as dist
from torch.nn import functional as F

# --- 生物信息学与统计库 ---
import RNA 
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
## NEW: Add biopython for Melting Temperature calculation
from Bio.SeqUtils import MeltingTemp as mt
from Bio.Seq import Seq

# --- 项目内部模块导入 ---
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from rich.progress import track

from bytelatent.args import (
    EvalArgs,
)
from bytelatent.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator
from bytelatent.data.iterators.packing_iterator import (
    PackingArgs,
    PackingIterator,
    PackingMode,
)
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.iterators.sequence_iterator import (
    SequenceIterator,
    SequencePackingArgs,
)
from bytelatent.data.patcher import PatcherArgs, PatchingModeEnum
from bytelatent.distributed import (
    DistributedArgs,
    dist_sum,
    get_device_mesh,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
    to_py_num,
)
from bytelatent.generate import (
    PackedCausalTransformerGenerator,
    load_consolidated_model_and_tokenizer,
)
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs # Unused
from bytelatent.transformer import LMTransformer

# --- 新增: 绘图库导入 ---
import matplotlib.pyplot as plt
import seaborn as sns


# --- 全局变量和日志设置 ---
EVAL_FOLDER_NAME = "{:010d}"
logger = logging.getLogger()
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

# --- 辅助函数 ---

def all_dicts_same(dict_list):
    if not dict_list:
        return True
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)

def calculate_levenshtein_distance_pair(pair):
    """(用于并行化)计算一对序列的 Levenshtein 距离"""
    return Levenshtein.distance(pair[0], pair[1])

## NEW: Function to plot base and dinucleotide frequency distributions
def plot_frequency_distributions(gen_freqs, train_freqs, output_path):
    """绘制并保存碱基和二联碱基频率的对比条形图"""
    logger.info("Plotting base and dinucleotide frequency distributions...")
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # --- 1. Base Frequencies ---
    base_labels = sorted(list(set(gen_freqs['base'].keys()) | set(train_freqs['base'].keys())))
    gen_base_vals = [gen_freqs['base'].get(b, 0) / (gen_freqs['total_len'] or 1) for b in base_labels]
    train_base_vals = [train_freqs['base'].get(b, 0) / (train_freqs['total_len'] or 1) for b in base_labels]
    
    df_base = pd.DataFrame({
        'Base': base_labels * 2,
        'Frequency': train_base_vals + gen_base_vals,
        'Set': ['Training'] * len(base_labels) + ['Generated'] * len(base_labels)
    })
    
    sns.barplot(x='Base', y='Frequency', hue='Set', data=df_base, ax=axes[0])
    axes[0].set_title('Single Nucleotide Frequencies')
    axes[0].set_ylabel('Frequency')

    # --- 2. Dinucleotide Frequencies ---
    dinu_labels = sorted(list(set(gen_freqs['dinucleotide'].keys()) | set(train_freqs['dinucleotide'].keys())))
    gen_dinu_vals = [gen_freqs['dinucleotide'].get(d, 0) / (gen_freqs['total_dinucleotides'] or 1) for d in dinu_labels]
    train_dinu_vals = [train_freqs['dinucleotide'].get(d, 0) / (train_freqs['total_dinucleotides'] or 1) for d in dinu_labels]
    
    df_dinu = pd.DataFrame({
        'Dinucleotide': dinu_labels * 2,
        'Frequency': train_dinu_vals + gen_dinu_vals,
        'Set': ['Training'] * len(dinu_labels) + ['Generated'] * len(dinu_labels)
    })
    
    sns.barplot(x='Dinucleotide', y='Frequency', hue='Set', data=df_dinu, ax=axes[1])
    axes[1].set_title('Dinucleotide Frequencies')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Frequency distribution plot saved to {output_path}")

## NEW: Renamed from the AIDO version to be more generic, as requested
def plot_property_distributions(generated_props, training_props, output_path):
    """绘制并保存生成数据与训练数据理化性质的分布对比图。"""
    common_keys = sorted(list(set(generated_props.keys()) & set(training_props.keys())))
    if not common_keys:
        print.warning("No common properties to plot.")
        return

    n_plots = len(common_keys)
    n_cols = 3 # Increased columns to better fit more plots
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
    axes = axes.flatten()

    for i, key in enumerate(common_keys):
        sns.kdeplot(generated_props[key], ax=axes[i], label="Generated", fill=True, alpha=0.5, warn_singular=False)
        sns.kdeplot(training_props[key], ax=axes[i], label="Training", fill=True, alpha=0.5, warn_singular=False)
        axes[i].set_title(f'Distribution of {key}')
        axes[i].set_xlabel(key)
        axes[i].set_ylabel('Density')
        axes[i].legend()

    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Property distribution plot saved to {output_path}")

## MODIFIED: Added Melting Temperature (tm) calculation
def calculate_single_rna_properties(seq: str) -> dict | None:
    """(用于并行化)计算单个RNA序列的多种性质。"""
    if not seq or len(seq) < 2:
        return None
        
    seq_upper = seq.upper().replace("T", "U")
    if not all(c in "ACGU" for c in seq_upper):
        return None

    properties = {}
    seq_len = len(seq_upper)

    # --- 结构性质 ---
    try:
        structure, mfe = RNA.fold(seq_upper)
        properties["mfe"] = mfe
        properties["mfe_normalized"] = mfe / seq_len
        paired_bases = structure.count('(') + structure.count(')')
        properties["structuredness"] = paired_bases / seq_len
    except Exception:
        # If folding fails, we can still calculate other properties
        properties["mfe"] = None
        properties["mfe_normalized"] = None
        properties["structuredness"] = None

    # --- 组成性质 ---
    properties["gc_content"] = (seq_upper.count('G') + seq_upper.count('C')) / seq_len
    
    counts = Counter(seq_upper)
    entropy = 0.0
    for base in counts:
        p = counts[base] / seq_len
        entropy -= p * math.log2(p)
    properties["sequence_entropy"] = entropy

    ## NEW: Calculate Melting Temperature
    try:
        # Use a standard RNA thermodynamic table. This assumes self-hybridization.
        properties["tm"] = mt.Tm_NN(Seq(seq_upper), nn_table=mt.R_DNA_NN1)
    except Exception:
        properties["tm"] = None # Cannot calculate for very short sequences

    # --- 频率计数 (用于后续聚合) ---
    properties["base_counts"] = Counter(seq_upper)
    properties["dinucleotide_counts"] = Counter([seq_upper[i:i+2] for i in range(seq_len - 1)])
    
    return properties

## MODIFIED: Updated to collect the new 'tm' property
def calculate_rna_properties(sequences: list[str]) -> tuple[dict, dict]:
    """为一批RNA序列并行计算多种性质。"""
    dist_properties = defaultdict(list)
    freq_properties = {
        "base": Counter(),
        "dinucleotide": Counter(),
        "total_len": 0,
        "total_dinucleotides": 0
    }
    num_processes = 10
    logger.info(f"Using {num_processes} CPU cores for enhanced property calculation...")

    with Pool(processes=num_processes) as pool:
        results = list(track(
            pool.imap_unordered(calculate_single_rna_properties, sequences, chunksize=100),
            description="Parallel enhanced RNA property calculation...",
            total=len(sequences)
        ))

    for res in results:
        if res:
            # Add 'tm' to the list of properties to collect
            for key in ["gc_content", "mfe", "sequence_entropy", "mfe_normalized", "structuredness", "tm"]:
                # Ensure the property was successfully calculated before appending
                if res.get(key) is not None:
                    dist_properties[key].append(res[key])
            
            freq_properties["base"].update(res["base_counts"])
            freq_properties["dinucleotide"].update(res["dinucleotide_counts"])
            seq_len = sum(res["base_counts"].values())
            freq_properties["total_len"] += seq_len
            freq_properties["total_dinucleotides"] += seq_len - 1

    return dict(dist_properties), freq_properties

# ... [MockAccelerator and EvalHarnessLM classes remain unchanged] ...
class MockAccelerator:
    def gather(self, tensor):
        if get_world_size() > 1:
            l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
            torch.distributed.all_gather(l, tensor)
            return torch.stack(l)
        return tensor.unsqueeze(0)

    def wait_for_everyone(self):
        if get_world_size() > 1:
            torch.distributed.barrier()

class EvalHarnessLM(LM):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: list[Instance]) -> list[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(list(gen_args)), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        self.generator.temperature = gen_args.get("temperature", 0.0)
        self.generator.top_p = gen_args.get("top_p", None)
        self.generator.top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        generations, _, _ = self.generator.generate(prompts)
        
        if not until:
            return generations
            
        filtered_gen = []
        for g in generations:
            for e in until:
                g = g.split(e)[0]
            filtered_gen.append(g)
        return filtered_gen

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        prompts, continuations = zip(*[req.args for req in requests])
        inputs = [req.args[0] + req.args[1] for req in requests]
        max_gen_len = self.generator.max_gen_len
        self.generator.max_gen_len = 1
        _, lls, greedy = self.generator.generate(inputs)
        results = []
        for p, ll, gr in zip(prompts, lls, greedy):
            p_len = len(self.generator.tokenizer.encode(p, add_bos=False, add_eos=False))
            results.append((ll[p_len:].sum().item(), gr[p_len:].all().item()))
        self.generator.max_gen_len = max_gen_len
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        prompts = [req.args[0] for req in requests]
        max_gen_len = self.generator.max_gen_len
        self.generator.max_gen_len = 1
        _, lls, _ = self.generator.generate(prompts)
        results = [ll.sum().item() for ll in lls]
        self.generator.max_gen_len = max_gen_len
        return results


@torch.no_grad() 
def eval_ppl_on_path( 
    *, 
    world_rank: int, 
    world_size: int, 
    model: LMTransformer | ByteLatentTransformer, 
    tokenizer_args: TokenizerArgs, 
    tokenizer,
    patcher_args: PatcherArgs, 
    packing_args: PackingArgs, 
    add_patches: bool, 
    path: str, 
    arrow_batch_size: int, 
    max_n_docs: int | None, 
    s3_profile: str | None = None, 
    training_set_path: str | None = None, 
    eval_args: EvalArgs, 
): 
    # ... [PPL calculation setup remains unchanged] ...
    model.eval() 
    seq_len = model.get_output_seq_len() 
    arrow_iterator = ArrowFileIterator( 
        file_path=None, 
        dataset_files=[path], 
        entropy_model_name=None, 
        worker_id=world_rank, 
        num_workers=world_size, 
        arrow_batch_size=arrow_batch_size, 
        preprocess_dir=None, 
        s3_profile=s3_profile, 
        file_format="arrow" if path.endswith("arrow") else "json", 
    ) 
    if max_n_docs is not None: 
        arrow_iterator = LimitIterator(arrow_iterator, limit=max_n_docs) 

    preprocess_iterator = PreprocessIterator( 
        arrow_iterator, 
        patcher_args=patcher_args, 
        tokenizer_args=tokenizer_args, 
        add_patches=add_patches, 
    ) 
    sequence_iterator = SequenceIterator( 
        preprocess_iterator, 
        sequence_packing_args=SequencePackingArgs( 
            output_seq_len=seq_len, 
            buffer_size=1, 
        ), 
        rng_state=None, 
    ) 
    packing_iterator = PackingIterator(sequence_iterator, packing_args=packing_args) 
    total_loss = 0.0 
    n_bytes = 0 
    all_sequences = []
    pad_token_id = 0 if tokenizer_args.name == "bytes" else tokenizer.pad_id 
    batch_iterator = packing_iterator.create_iter() 
    
    for batch in track(batch_iterator, description=f"Rank {world_rank} PPL eval..."): 
        x = torch.from_numpy(batch.x).cuda() 
        y = torch.from_numpy(batch.y).cuda() 
        mask = None if batch.mask is None else torch.from_numpy(batch.mask).cuda() 
        patch_lengths = batch.patch_lengths 
        if patch_lengths is not None: 
            patch_lengths = torch.from_numpy(patch_lengths).cuda() 

        for seq_tokens in y: 
            actual_tokens = seq_tokens[seq_tokens != pad_token_id] 
            decoded_seq = tokenizer.decode(actual_tokens.cpu().numpy()) 
            if decoded_seq:
                all_sequences.append(decoded_seq) 

        if tokenizer_args.name in ["bytes", "blt"]: 
            n_bytes += y.numel() if mask is None else mask.sum().item() 
            if isinstance(model, ByteLatentTransformer): 
                pred = model(x, patch_lengths=patch_lengths) 
            else: 
                pred = model(x) 
            loss = F.cross_entropy(pred.flatten(0, 1), y.flatten(0, 1), reduction="sum") 
            total_loss += loss.item() 
        else: 
            raise NotImplementedError() 
            
    if world_size > 1: 
        all_n_bytes = to_py_num(dist_sum(n_bytes)) 
        all_total_loss = to_py_num(dist_sum(total_loss)) 
    else: 
        all_n_bytes = n_bytes 
        all_total_loss = total_loss 

    if all_n_bytes == 0: 
        print.warning("0 bytes evaluated, returning 0 for PPL/BPB.") 
        return { 
            "n_bytes": 0, "n_bytes_gpu": 0, "loss_sum": 0, "loss_sum_gpu": 0, 
            "loss_mean": 0, "loss_mean_gpu": 0, "ppl": 0.0, "bpb": 0.0 
        } 

    metrics_results = {} 
    if world_rank == 0:
        print(all_sequences)
        logger.info(f"Gathered {len(all_sequences)} sequences for metric calculation on rank 0.")
        
        # --- Uniqueness, Novelty, Diversity remain unchanged ---
        unique_sequences = set(all_sequences)
        metrics_results["uniqueness"] = len(unique_sequences) / len(all_sequences) if all_sequences else 0.0

        training_set = set()
        if training_set_path:
            try:
                train_df = feather.read_table(training_set_path).to_pandas()
                training_set = set(train_df['text'].tolist())
                novel_sequences = unique_sequences.difference(training_set)
                metrics_results["novelty"] = len(novel_sequences) / len(unique_sequences) if unique_sequences else 0.0
            except Exception as e:
                print.warning(f"Could not calculate Novelty. Failed to load training set from {training_set_path}: {e}")
                metrics_results["novelty"] = None
        else:
            print.warning("`training_set_path` not provided, skipping Novelty calculation.")
            metrics_results["novelty"] = None

        unique_sequences_list = list(unique_sequences)
        n_unique = len(unique_sequences_list)
        diversity = 0.0
        if n_unique > 1:
            SAMPLING_THRESHOLD, SAMPLE_SIZE = 500, 10000
            if n_unique < SAMPLING_THRESHOLD:
                pairs = list(itertools.combinations(unique_sequences_list, 2))
            else:
                pairs = [random.sample(unique_sequences_list, 2) for _ in range(SAMPLE_SIZE)]
            
            with Pool(processes=cpu_count()) as pool:
                distances = pool.map(calculate_levenshtein_distance_pair, pairs)
            diversity = np.mean(distances) if distances else 0.0
        metrics_results["diversity"] = diversity
        
        # --- MODIFIED: Comprehensive RNA Property Evaluation ---
        if eval_args.run_rna_metrics and training_set:
            logger.info("Starting comprehensive RNA property evaluation...")
            
            PROP_SAMPLE_SIZE = 10000
            sequences_to_calc_gen = random.sample(unique_sequences_list, min(n_unique, PROP_SAMPLE_SIZE))
            sequences_to_calc_train = random.sample(list(training_set), min(len(training_set), PROP_SAMPLE_SIZE))
            
            logger.info(f"Calculating properties for {len(sequences_to_calc_gen)} generated and {len(sequences_to_calc_train)} training sequences...")
            generated_dist_props, generated_freq_props = calculate_rna_properties(sequences_to_calc_gen)
            training_dist_props, training_freq_props = calculate_rna_properties(sequences_to_calc_train)

            rna_metrics = {}
            # a) Distribution comparison (K-S test)
            common_dist_keys = set(generated_dist_props.keys()) & set(training_dist_props.keys())
            for prop_name in common_dist_keys:
                stat, pval = ks_2samp(generated_dist_props[prop_name], training_dist_props[prop_name])
                rna_metrics[f"dist_sim_{prop_name}"] = {"ks_statistic": stat, "p_value": pval}

            # b) Composition comparison (J-S divergence)
            bases = ['A', 'C', 'G', 'U']
            gen_base_freq = np.array([generated_freq_props["base"].get(b, 0) for b in bases])
            train_base_freq = np.array([training_freq_props["base"].get(b, 0) for b in bases])
            if gen_base_freq.sum() > 0 and train_base_freq.sum() > 0:
                rna_metrics["composition_jsd_base"] = jensenshannon(gen_base_freq, train_base_freq, base=2)
            
            dinucleotides = [b1 + b2 for b1 in bases for b2 in bases]
            gen_dinu_freq = np.array([generated_freq_props["dinucleotide"].get(d, 0) for d in dinucleotides])
            train_dinu_freq = np.array([training_freq_props["dinucleotide"].get(d, 0) for d in dinucleotides])
            if gen_dinu_freq.sum() > 0 and train_dinu_freq.sum() > 0:
                rna_metrics["composition_jsd_dinucleotide"] = jensenshannon(gen_dinu_freq, train_dinu_freq, base=2)

            metrics_results["rna_property_evaluation"] = rna_metrics

            ## NEW: Call the new visualization functions
            if eval_args.dump_dir:
                # Plot for base/dinucleotide frequencies
                freq_plot_path = os.path.join(eval_args.dump_dir, "frequency_distribution.png")
                plot_frequency_distributions(generated_freq_props, training_freq_props, freq_plot_path)
                
                # Plot for all other physicochemical properties (including Tm)
                props_plot_path = os.path.join(eval_args.dump_dir, "physicochemical_properties_distribution.png")
                plot_property_distributions(generated_dist_props, training_dist_props, props_plot_path)


    ppl_results = { 
        "n_bytes": all_n_bytes, 
        "n_bytes_gen": n_bytes, 
        "loss_sum": all_total_loss, 
        "loss_sum_gpu": total_loss, 
        "loss_mean": all_total_loss / all_n_bytes, 
        "loss_mean_gpu": total_loss / n_bytes if n_bytes > 0 else 0, 
        "ppl": math.exp(all_total_loss / all_n_bytes), 
        "bpb": all_total_loss / math.log(2) / all_n_bytes, 
    } 

    return ppl_results | metrics_results 

# ... [launch_eval and main functions remain unchanged] ...
def launch_eval(eval_args: EvalArgs): 
    assert eval_args.dump_dir is not None 
    assert eval_args.ckpt_dir is not None 

    distributed_args = DistributedArgs() 
    distributed_args.configure_world() 

    if not torch.distributed.is_initialized(): 
        setup_torch_distributed(distributed_args) 

    world_mesh = get_device_mesh(distributed_args) 
    dp_mesh = world_mesh["dp_replicate"] 
    assert distributed_args.dp_shard == 1 
    world_size = dp_mesh.size() 
    world_rank = dp_mesh.get_local_rank() 

    fs = get_fs(eval_args.ckpt_dir, s3_profile=eval_args.s3_profile) 
    if ( 
        fs.exists(eval_args.ckpt_dir) 
        and fs.exists(os.path.join(eval_args.ckpt_dir, "params.json")) 
        and len(fs.glob(os.path.join(eval_args.ckpt_dir, "*.pth"))) != 0 
    ): 
        consolidate_path = eval_args.ckpt_dir 
    else: 
        consolidate_path = os.path.join(eval_args.ckpt_dir, CONSOLIDATE_FOLDER) 
        if not fs.exists(consolidate_path) and get_global_rank() == 0: 
            consolidate_path = consolidate_checkpoints(fs, eval_args.ckpt_dir) 

    fs.mkdirs(eval_args.dump_dir, exist_ok=True) 
    with fs.open(os.path.join(eval_args.dump_dir, "config.yaml"), "w") as f: 
        f.write(eval_args.model_dump_json()) 

    logger.info("Loading model...") 
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer( 
        consolidate_path, 
    ) 
    model.eval() 
    logger.info("Model loaded successfully.") 

    logger.info("Synchronizing all ranks after model loading...") 
    torch.distributed.barrier() 
    logger.info("All ranks synchronized.") 
    
    pad_id = 0 if train_cfg.data.tokenizer_args.name == "bytes" else tokenizer.boe_id 

    ppl_results = None 
    if eval_args.run_ppl: 
        assert eval_args.validation is not None 
        packing_args = PackingArgs( 
            batch_size=eval_args.validation.batch_size, 
            seq_len=train_cfg.data.seq_len, 
            max_length=train_cfg.data.max_encoder_seq_length, 
            pad_to_max_length=True, 
            enable_byte_ngrams=False, 
            pad_id=pad_id, 
            packing_mode=( 
                PackingMode.BYTES 
                if train_cfg.data.patcher_args.patching_mode == PatchingModeEnum.byte 
                else PackingMode.PATCHING 
            ), 
        ) 
        if len(eval_args.validation.sources) > 0: 
            ppl_results = {} 
            logger.info("Starting PPL evaluation on validation sets") 
            for source in eval_args.validation.sources: 
                ppl_results[source] = eval_ppl_on_path( 
                    world_rank=world_rank, 
                    world_size=world_size, 
                    model=model, 
                    tokenizer=tokenizer,
                    tokenizer_args=train_cfg.data.tokenizer_args, 
                    patcher_args=train_cfg.data.patcher_args, 
                    packing_args=packing_args, 
                    add_patches=train_cfg.data.add_patches, 
                    path=os.path.join(eval_args.validation.root_dir, source), 
                    max_n_docs=eval_args.validation.max_n_docs, 
                    arrow_batch_size=20, 
                    s3_profile=eval_args.s3_profile, 
                    training_set_path=eval_args.training_set_path, 
                    eval_args=eval_args,
                ) 

    task_results = None 
    if eval_args.run_tasks: 
        assert eval_args.generator is not None 
        assert eval_args.harness is not None 
        generator = PackedCausalTransformerGenerator( 
            eval_args.generator, model, tokenizer 
        ) 
        wrap = EvalHarnessLM(generator) 
        task_results = simple_evaluate(wrap, **eval_args.harness.model_dump()) 

    results = {"ppl": ppl_results, "tasks": task_results} 

    if get_global_rank() == 0: 
        with fs.open(os.path.join(eval_args.dump_dir, "results.json"), "w") as f: 
            f.write(json.dumps(results, indent=2)) 
        logger.info(f"All evaluation results: {json.dumps(results, indent=2)}") 
        if ppl_results is not None: 
            with fs.open(os.path.join(eval_args.dump_dir, "validation.json"), "w") as f: 
                f.write(json.dumps(ppl_results, indent=2)) 
            logger.info(f"All validation results: {json.dumps(ppl_results, indent=2)}") 

    if eval_args.metric_log_dir and get_global_rank() == 0: 
        metric_log_path = os.path.join(eval_args.metric_log_dir, "metrics.eval.jsonl") 
        logger.info(f"Writing metric logs to {metric_log_path}") 
        timestamp: dict[str, int | str] = { 
            "created_at": datetime.utcnow().isoformat(), 
        } 
        if eval_args.global_step is not None: 
            timestamp["global_step"] = eval_args.global_step 
        
        with fs.open(metric_log_path, mode="a") as f: 
            f.write(json.dumps(timestamp | results) + "\n") 

        val_log_path = os.path.join( 
            eval_args.metric_log_dir, "metrics.validation.jsonl" 
        ) 
        if ppl_results is not None: 
            with fs.open(val_log_path, mode="a") as f: 
                f.write(json.dumps(timestamp | ppl_results) + "\n") 


def main(): 
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) 
    torch.cuda.set_device(local_rank) 
    
    try: 
        eval_args = parse_args_to_pydantic_model(EvalArgs) 
        launch_eval(eval_args) 
    finally: 
        if dist.is_initialized(): 
            dist.destroy_process_group() 

if __name__ == "__main__": 
    main()