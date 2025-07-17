import os

import torch
import typer

from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.generate_blt import generate_nocache
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer


def main(prompt: str, model_name: str = "blt-1b"):
    assert model_name in ["blt-1b", "blt-7b"]
    model_name = model_name.replace("-", "_")
    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)
    checkpoint_path = os.path.join("hf-weights", model_name)
    print(f"Loading BLT model: {model_name}")
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        checkpoint_path,
    )
    assert isinstance(model, ByteLatentTransformer)
    assert isinstance(tokenizer, BltTokenizer)
    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    print("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = os.path.join(
        "hf-weights", "entropy_model"
    )
    patcher = patcher_args.build()
    prompts = [prompt]
    outputs = generate_nocache(
        prompts, model=model, tokenizer=tokenizer, patcher=patcher
    )
    text_outputs = [tokenizer.decode(t) for t in outputs]
    for p, t in zip(prompts, text_outputs):
        print(f'Prompt: "{p}" Completion: "{t}"')
        print()


if __name__ == "__main__":
    typer.run(main)
