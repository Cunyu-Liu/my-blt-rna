import logging
import os
import math
import torch
from tqdm import tqdm # 引入tqdm来显示进度
from bytelatent.args import EvalArgs
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.patcher import Patcher
from bytelatent.distributed import (
    DistributedArgs,
    dist_max,
    dist_min,
    dist_sum,
    get_device_mesh,
    setup_torch_distributed,
)
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer

logger = logging.getLogger()


def get_max_length(input_tokens: list[list[int]] | None) -> int:
    # reduce max length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        max_length = 0
    else:
        max_length = max([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        max_length = int(dist_max(max_length))
    return max_length


def get_min_length(input_tokens: list[list[int]] | None) -> int:
    # reduce min length prompt over all processes to have an equal number of call on each process with fsdp
    if input_tokens is None:
        # TODO: Double check this change from int(1e9) is correct
        min_length = 0
    else:
        min_length = min([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        min_length = int(dist_min(min_length))
    return min_length


def get_generation_range(
    prompt_tokens: list[list[int]] | None, max_gen_len: int
) -> tuple[int, int]:
    batch_min_prompt_length = get_min_length(prompt_tokens)
    batch_max_prompt_length = get_max_length(prompt_tokens)
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len


def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)  # batch_sz x topk
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate_nocache(
    prompts: list[str] | None,
    *,
    model: ByteLatentTransformer,
    tokenizer: BltTokenizer,
    patcher: Patcher,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    # --- 1. 添加新的 batch_size 参数 ---
    batch_size: int = 1,
    remove_prompts: bool = True,
) -> list[list[int]]:
    assert (
        patcher.realtime_patching
    ), "generate_nocache requires patcher.realtime_patching=True"
    model.eval()
    if prompts is None:
        # --- 2. 修正这里的逻辑, 使用新的 batch_size 参数 ---
        prompt_tokens = [[tokenizer.bos_id] for _ in range(batch_size)]
        n_truncated_prompts = 0
        total_truncated_prompts = 0
    else:
        prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in prompts]
        n_truncated_prompts = sum([max_prompt_len < len(t) for t in prompt_tokens])
        total_truncated_prompts = dist_sum(n_truncated_prompts)

        # Truncation
        prompt_tokens = [
            t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :]
            for t in prompt_tokens
        ]

    if total_truncated_prompts > 0:
        logger.info(
            f"There are {total_truncated_prompts} prompts that are truncated on the left, "
            f"length greater than max_prompt_len = {max_prompt_len}, "
            f"maximum prompt length = {get_max_length(prompt_tokens)} across all gpus."
        )

    if prompt_tokens is None:
        prompt_tokens = [[tokenizer.bos_id] for _ in range(end_pos)]

    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    batch_size = len(prompt_tokens)
    tokens = torch.full((batch_size, end_pos), tokenizer.pad_id).cuda().long()

    # Copy inputs to tensor for generated tokens
    for i, row_tokens in enumerate(prompt_tokens):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long()
    input_text_mask = tokens != tokenizer.pad_id

    for i, curr_pos in enumerate(range(start_pos, end_pos)):
        current_tokens = tokens[:, :curr_pos]
        patch_lengths, _ = patcher.patch(current_tokens, include_next_token=True)
        logits = model(current_tokens, patch_lengths=patch_lengths)[:, -1]

        if use_sampling:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, top_k)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = torch.where(
            input_text_mask[:, curr_pos], tokens[:, curr_pos], next_token
        )
        tokens[:, curr_pos] = next_token

    if remove_prompts:
        generated_tokens = [
            t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    else:
        generated_tokens = [
            t[: len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    return generated_tokens


def launch_generate(eval_args: EvalArgs):
    # --- 分布式设置和模型加载 (保持不变) ---
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
        raise ValueError("Did not find a consolidated checkpoint in the ckpt_dir")

    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
    )
    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    patcher_args.entropy_model_checkpoint_dir = eval_args.entropy_ckpt_dir
    patcher = patcher_args.build()
    
    
    # --- 核心逻辑修改：从单次执行改为批量循环生成 ---
    num_to_generate = eval_args.num_to_generate
    batch_size = eval_args.batch_size
    output_file = eval_args.output_file

    # 只有主进程 (rank 0) 才执行文件写入和打印操作
    if world_rank == 0:
        print(f"Starting generation of {eval_args.num_to_generate} sequences...")
        print(f"Batch size: {eval_args.batch_size}, saving to: {eval_args.output_file}")
        # 清空或创建输出文件
        with open(eval_args.output_file, "w") as f:
            pass 

    # 计算总共需要多少个批次
    num_batches = math.ceil(eval_args.num_to_generate / eval_args.batch_size)
    
    # 使用tqdm来显示进度条，仅在主进程上显示
    progress_bar = tqdm(range(num_batches), disable=(world_rank != 0))

    generated_count = 0
    with open(output_file, "a") as f:
        for i in progress_bar:
            if generated_count >= num_to_generate:
                break
            
            current_batch_size = min(batch_size, num_to_generate - generated_count)
            
            # 直接调用 generate_nocache，并从 eval_args 的正确位置传递参数
            outputs = generate_nocache(
                prompts=None,
                model=model,
                tokenizer=tokenizer,
                patcher=patcher,
                batch_size=current_batch_size,
                # 从 eval_args.generator 获取生成器参数
                max_gen_len=eval_args.generator.max_gen_len,
                use_sampling=eval_args.generator.temperature > 0,
                temp=eval_args.generator.temperature,
                top_k=eval_args.generator.top_k,
                top_p=eval_args.generator.top_p
            )
            
            text_outputs = [tokenizer.decode(t) for t in outputs]
            
            if world_rank == 0:
                for text in text_outputs:
                    clean_text = text.replace("\n", " ").replace("\r", "")
                    f.write(clean_text + "\n")
                f.flush()

            generated_count += len(text_outputs)
            if world_rank == 0:
                progress_bar.set_description(f"Generated {generated_count}/{num_to_generate}")

    if world_rank == 0:
        print(f"Generation complete. {generated_count} sequences saved to {output_file}")

    for p, t in zip(eval_args.prompts, text_outputs):
        print(f'Prompt: "{p}" Completion: "{t}"')
        print()



def main():
    eval_args = parse_args_to_pydantic_model(EvalArgs)
    launch_generate(eval_args)


if __name__ == "__main__":
    main()
