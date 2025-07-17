import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import typer
from huggingface_hub import hf_hub_download
from huggingface_hub.hub_mixin import ModelHubMixin

from bytelatent.args import TrainArgs
from bytelatent.data.patcher import PatcherArgs, to_device
from bytelatent.distributed import DistributedArgs, setup_torch_distributed
from bytelatent.entropy_model import load_entropy_model
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.generate_blt import generate_nocache
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.transformer import LMTransformer

app = typer.Typer()


class BltTokenizerAndPatcher(ModelHubMixin):
    def __init__(
        self,
        *,
        patcher_args: PatcherArgs,
        tokenizer_args: TokenizerArgs,
        distributed_args: DistributedArgs,
    ):
        self.patcher_args = patcher_args
        self.tokenizer_args = tokenizer_args
        self.distributed_args = distributed_args

    def push_to_hub(self, *args, **kwargs):
        raise ValueError(
            "For meta authors: Do not push BLT weights with this, save weights with save_pretrained() then push them manually to HF hub to ensure the repository metadata is correct."
        )

    def save_pretrained(self, *args, **kwargs):
        raise ValueError(
            "Tokenizer and Patcher are saved by BLT, this class is just for loading"
        )

    def _save_pretrained(self, *args, **kwargs):
        raise ValueError(
            "Tokenizer and Patcher are saved by BLT, this class is just for loading"
        )

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ):
        if os.path.isdir(model_id):
            train_args_file = os.path.join(model_id, "train_args.json")
        else:
            train_args_file = hf_hub_download(
                repo_id=model_id,
                filename="train_args.json",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            )

        with open(train_args_file) as f:
            train_args = TrainArgs(**json.load(f))
        return cls(
            patcher_args=train_args.data.patcher_args,
            tokenizer_args=train_args.data.tokenizer_args,
            distributed_args=train_args.distributed,
        )


@app.command()
def convert_to_transformers(blt_weights_dir: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(blt_weights_dir)
    blt_dir = os.path.join(output_dir, "blt")
    entropy_dir = os.path.join(output_dir, "entropy")
    model.save_pretrained(blt_dir, config={"args": train_cfg.model.model_dump()})
    shutil.copyfile(
        os.path.join(blt_weights_dir, "params.json"),
        os.path.join(blt_dir, "train_args.json"),
    )
    blt_readme_file = os.path.join(blt_dir, "README.md")
    if os.path.exists(blt_readme_file):
        os.remove(blt_readme_file)

    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = False
    print("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = os.path.join(
        blt_weights_dir, "entropy_model"
    )
    state_path = os.path.join(
        patcher_args.entropy_model_checkpoint_dir, "consolidated.pth"
    )
    entropy_model, entropy_model_args = load_entropy_model(
        patcher_args.entropy_model_checkpoint_dir, state_path
    )
    entropy_model.save_pretrained(
        entropy_dir, config={"args": entropy_model_args.model_dump()}
    )
    entropy_readme_file = os.path.join(entropy_dir, "README.md")
    if os.path.exists(entropy_readme_file):
        os.remove(entropy_readme_file)


@app.command()
def load_transformers(
    source: str,
    entropy_repo: str = "facebook/blt-entropy",
    blt_repo: str = "facebook/blt-1b",
    entropy_dir: str | None = None,
    blt_dir: str | None = None,
    prompt: str | None = None,
):
    if source == "local":
        assert entropy_dir is not None
        assert blt_dir is not None
        entropy_model = LMTransformer.from_pretrained(
            entropy_dir, local_files_only=True
        )
        blt_model = ByteLatentTransformer.from_pretrained(
            blt_dir, local_files_only=True
        )
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(
            blt_dir, local_files_only=True
        )
        tokenizer = tok_and_patcher.tokenizer_args.build()
        patcher = tok_and_patcher.patcher_args.build()
        print("Loaded all local")
        print(entropy_model)
        print(blt_model)
        print(tok_and_patcher)
    elif source == "hub":
        entropy_model = LMTransformer.from_pretrained(entropy_repo)
        blt_model = ByteLatentTransformer.from_pretrained(blt_repo)
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(blt_repo)
        tokenizer = tok_and_patcher.tokenizer_args.build()
        patcher = tok_and_patcher.patcher_args.build()
        print("Loaded all remote")
        print(entropy_model)
        print(blt_model)
        print(tok_and_patcher)
    else:
        raise ValueError(f"Unknown source: {source}")

    if prompt is not None:
        assert isinstance(tokenizer, BltTokenizer)
        # Move args to correct GPU
        param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
            tok_and_patcher.distributed_args.model_dtype
        ]
        blt_model = blt_model.cuda().eval()
        for param in blt_model.parameters():
            param.data = param.data.to(dtype=param_dtype)

        # Enable realtime patching
        patcher.realtime_patching = True
        patcher.entropy_model, _ = to_device(
            entropy_model, tok_and_patcher.patcher_args.patching_device
        )

        # Setup distributed
        distributed_args = DistributedArgs()
        distributed_args.configure_world()
        if not torch.distributed.is_initialized():
            setup_torch_distributed(distributed_args)
        prompts = [prompt]
        outputs = generate_nocache(
            prompts, model=blt_model, tokenizer=tokenizer, patcher=patcher
        )
        text_outputs = [tokenizer.decode(t) for t in outputs]
        for p, t in zip(prompts, text_outputs):
            print(f'Prompt: "{p}"\nCompletion: "{t}"')
            print()


if __name__ == "__main__":
    app()
