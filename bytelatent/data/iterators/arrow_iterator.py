# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import re
import json
from logging import getLogger
from typing import Any, Generator

import fsspec
import pyarrow as pa

# pyarrow needs the initialization from this import
import pyarrow.dataset  # pyright: ignore
import s3fs
from pydantic import BaseModel, ConfigDict

from bytelatent import ByteLatentError
from bytelatent.data.data_types import BltExample
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.abstract_iterator import (
    PydanticIteratorState,
    StatefulIterator,
)
from bytelatent.preprocess.preprocess_entropies import get_id_key, get_text

logger = getLogger(__name__)


class ArrowFileIteratorState(PydanticIteratorState):
    model_config = ConfigDict(extra="forbid")
    file_path: str | None
    row_num: int
    num_workers: int
    worker_id: int
    preprocess_dir: str | None
    dataset_files: list[str] | None
    entropy_model_name: str | None
    arrow_batch_size: int = 100
    s3_profile: str | None
    filesystem_type: str | None = None
    file_format: str

    def build(self) -> "ArrowFileIterator":
        arrow_file = ArrowFileIterator(
            file_path=self.file_path,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            preprocess_dir=self.preprocess_dir,
            entropy_model_name=self.entropy_model_name,
            arrow_batch_size=self.arrow_batch_size,
            dataset_files=self.dataset_files,
            s3_profile=self.s3_profile,
            filesystem_type=self.filesystem_type,
            file_format=self.file_format,
        )
        # Only set row number for non-jsonl files
        if self.row_num != 0 and self.file_format != "jsonl":
            arrow_file._set_row_num(self.row_num)
        return arrow_file


def shard_sort_key(file: str):
    assert isinstance(file, str)
    match = re.search(r".+\.shard_([0-9]+)\.arrow", file)
    shard_number = int(match.group(1))
    return shard_number


def maybe_truncate_string(text: str, max_length: int):
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length] + "..."


class ArrowFileIterator(StatefulIterator):
    def __init__(
        self,
        *,
        file_path: str | None,
        worker_id: int,
        num_workers: int,
        preprocess_dir: str | None,
        entropy_model_name: str | None,
        arrow_batch_size: int,
        dataset_files: list[str] | None = None,
        s3_profile: str | None = None,
        filesystem_type: str | None = None,
        file_format: str = "arrow",   
    ):
        assert 0 <= worker_id < num_workers, (worker_id, num_workers)
        if file_path is None and dataset_files is None:
            raise ByteLatentError("file_path and dataset_files cannot both be None")
        
        self.row_num = 0
        self.iter_id = 0
        self.batch_iterator = None
        self.batch_to_consume = None
        self.dataset = None
        self.file_path = file_path
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.preprocess_dir = preprocess_dir
        self.entropy_model_name = entropy_model_name
        self.arrow_batch_size = arrow_batch_size
        self.s3_profile = s3_profile
        self.filesystem_type = filesystem_type
        self.file_format = file_format
        self.fs = None
        
        if self.filesystem_type is not None:
            if self.filesystem_type == "file":
                self.fs = fsspec.filesystem("file")
            elif self.filesystem_type == "s3":
                self.fs = fsspec.filesystem("s3", profile=s3_profile)
            else:
                raise ValueError("Unknown filesystem")
            logger.info("Arrow iterator using fs=%s", self.fs)

        if dataset_files is None:
            assert file_path is not None, "Must specify file_Path if dataset_files is None"
            if file_format == "json":
                if self.fs is None:
                    self.fs = get_fs(file_path, s3_profile=s3_profile)
                    if isinstance(self.fs, s3fs.S3FileSystem):
                        self.filesystem_type = "s3"
                    else:
                        self.filesystem_type = "file"
                self.dataset_files = [file_path]
            elif file_format == "jsonl":
                if self.fs is None:
                    self.fs = get_fs(file_path, s3_profile=s3_profile)
                    if isinstance(self.fs, s3fs.S3FileSystem):
                        self.filesystem_type = "s3"
                    else:
                        self.filesystem_type = "file"
                self.dataset_files = [file_path]
            else:
                if preprocess_dir is None:
                    # Directly use the arrow file if no preprocess_dir specified
                    self.dataset_files = [file_path]
                else:
                    # Otherwise look for preprocessed shards
                    jsonl_basename = os.path.basename(file_path)
                    dataset = os.path.splitext(jsonl_basename)[0]
                    data_dir = os.path.join(preprocess_dir, dataset, entropy_model_name)
                    data_dir_with_glob = os.path.join(
                        data_dir, f"{jsonl_basename}.shard_*.arrow"
                    )
                    if self.fs is None:
                        self.fs = get_fs(data_dir_with_glob, s3_profile=s3_profile)
                        if isinstance(self.fs, s3fs.S3FileSystem):
                            self.filesystem_type = "s3"
                        else:
                            self.filesystem_type = "file"
                    shard_files = self.fs.glob(data_dir_with_glob)

                    for s in shard_files:
                        complete_file = os.path.join(
                            data_dir, f"{os.path.basename(s)}.complete"
                        )
                        if not self.fs.exists(complete_file):
                            raise ValueError(f"Missing .complete for input file: {s}")

                    shard_files = sorted(shard_files, key=shard_sort_key)
                    if len(shard_files) == 0:
                        raise ByteLatentError(
                            f"Zero shard_files found corresponding to: {file_path} using preprocess_dir={preprocess_dir} and entropy_model_name={entropy_model_name}, so the search path is data_dir={data_dir} for matches to {jsonl_basename}.shard_*.arrow"
                        )
                    self.dataset_files = [f for f in shard_files]
        else:
            self.preprocess_dir = None
            self.dataset_files = dataset_files
            if dataset_files[0].startswith("s3://"):
                for f in dataset_files:
                    assert f.startswith("s3://")
            if self.fs is None:
                self.fs = get_fs(dataset_files[0], s3_profile=s3_profile)
                if isinstance(self.fs, s3fs.S3FileSystem):
                    self.filesystem_type = "s3"
                else:
                    self.filesystem_type = "file"

    def get_state(self) -> ArrowFileIteratorState:
        return ArrowFileIteratorState(
            file_path=self.file_path,
            row_num=self.row_num,
            worker_id=self.worker_id,
            num_workers=self.num_workers,
            preprocess_dir=self.preprocess_dir,
            entropy_model_name=self.entropy_model_name,
            arrow_batch_size=self.arrow_batch_size,
            dataset_files=self.dataset_files,
            s3_profile=self.s3_profile,
            filesystem_type=self.filesystem_type,
            file_format=self.file_format,
        )

    def create_iter(self) -> Generator[BltExample, Any, None]:
        if self.file_format == "jsonl":
            # Custom handling for jsonl files
            with self.fs.open(self.file_path) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            raise ValueError("Expected JSON object but got other type")
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                        # Try decoding with GBK if UTF-8 fails
                        try:
                            decoded_line = line.decode('gbk').encode('utf-8')
                            data = json.loads(decoded_line)
                            if not isinstance(data, dict):
                                raise ValueError("Expected JSON object but got other type")
                        except Exception:
                            # Check if line contains binary data (high bytes)
                            if any(b > 127 for b in line):
                                logger.warning(f"Skipping binary data line in {self.file_path}")
                                continue
                            logger.warning(f"Failed to parse line {self.row_num} in {self.file_path}: {str(e)}")
                            logger.warning(f"Problematic line start: {str(line[:100])}")
                            continue
                    
                    if not isinstance(data, dict):
                        logger.warning(f"Skipping non-object line {self.row_num} in {self.file_path}")
                        continue
                        
                    out = BltExample(
                        sample_id=data.get("sample_id", ""),
                        entropies=None,
                        text=data.get("text", ""),
                        tokens=None,
                        mask=None,
                        patch_lengths=None,
                    )
                    self.row_num += 1
                    if (self.row_num - 1) % self.num_workers == self.worker_id:
                        yield out
            return

        if self.dataset is None:
            if isinstance(self.fs, s3fs.core.S3FileSystem):
                filesystem = self.fs
            else:
                filesystem = None
            self.dataset = pa.dataset.dataset(
                self.dataset_files, 
                format=self.file_format if self.file_format != "jsonl" else "json",
                filesystem=filesystem
            )

        self.iter_id += 1
        if self.batch_to_consume is not None:
            batch_columns: dict[str, list] = self.batch_to_consume
            self.batch_to_consume = None
            if self.file_format == "arrow":
                sample_ids = batch_columns["sample_id"]
                texts = batch_columns["text"]
                entropies = batch_columns["entropies"]
            elif self.file_format == "json":
                sample_ids = batch_columns[get_id_key(batch_columns)]
                texts = get_text(batch_columns)
                entropies = None
            else:
                raise ValueError(f"Unknown file format: {self.file_format}")
            
            for i in range(len(sample_ids)):
                out = BltExample(
                    sample_id=sample_ids[i],
                    entropies=entropies[i] if entropies is not None else None,
                    text=texts[i],
                    tokens=None,
                    mask=None,
                    patch_lengths=None,
                )
                self.row_num += 1
                if (self.row_num - 1) % self.num_workers == self.worker_id:
                    yield out

        self.batch_iterator = self.dataset.to_batches(
            batch_size=self.arrow_batch_size,
            fragment_readahead=1,
            batch_readahead=1,
        )
        for batch in self.batch_iterator:
            batch_columns = batch.to_pydict()
            if self.file_format == "arrow":
                sample_ids = batch_columns["sample_id"]
                texts = batch_columns["text"]
                entropies = batch_columns["entropies"]
            elif self.file_format == "json":
                sample_ids = batch_columns[get_id_key(batch_columns)]
                texts = get_text(batch_columns)
                entropies = None
            else:
                raise ValueError(f"Unknown file format: {self.file_format}")
            
            for i in range(len(sample_ids)):
                out = BltExample(
                    sample_id=sample_ids[i],
                    entropies=entropies[i] if entropies is not None else None,
                    text=texts[i],
                    tokens=None,
                    mask=None,
                    patch_lengths=None,
                )
                self.row_num += 1
                if (self.row_num - 1) % self.num_workers == self.worker_id:
                    yield out

    def _set_row_num(self, target_row_num: int):
        data_str = maybe_truncate_string(str(self.dataset_files), 200)
        logger.info(f"Setting arrow position to {target_row_num} for {data_str}")
        
        if self.file_format == "jsonl":
            # For jsonl files, we can't seek to specific rows efficiently
            # Just reset to beginning if target is 0, otherwise raise error
            if target_row_num == 0:
                self.row_num = 0
                self.dataset = None
                self.batch_iterator = None
                self.batch_to_consume = None
            else:
                raise ValueError("Cannot seek to specific rows in jsonl files")
            return

        if target_row_num is None or target_row_num == 0:
            self.row_num = 0
            self.dataset = None
            self.batch_iterator = None
            self.batch_to_consume = None
        else:
            if isinstance(self.fs, s3fs.core.S3FileSystem):
                filesystem = self.fs
            else:
                filesystem = None
            self.dataset = pa.dataset.dataset(
                self.dataset_files, format="arrow", filesystem=filesystem
            )
            self.batch_iterator = self.dataset.to_batches(
                batch_size=self.arrow_batch_size
            )
            curr_remaining = target_row_num
            for batch in self.batch_iterator:
                if len(batch) > curr_remaining:
                    batch_columns: dict[str, list] = batch.to_pydict()
                    if self.file_format == "arrow":
                        leftover_sample_ids = batch_columns["sample_id"][curr_remaining:]
                        leftover_entropies = batch_columns["entropies"][curr_remaining:]
                        leftover_texts = batch_columns["text"][curr_remaining:]
                    elif self.file_format == "json":
                        leftover_sample_ids = batch_columns[get_id_key(batch_columns)][curr_remaining:]
                        leftover_entropies = None
                        leftover_texts = get_text(batch_columns)[curr_remaining:]
                    else:
                        raise ValueError(f"Unknown file format: {self.file_format}")

                    batch_columns["sample_id"] = leftover_sample_ids
                    batch_columns["entropies"] = leftover_entropies
                    batch_columns["text"] = leftover_texts
                    self.batch_to_consume = batch_columns
                    break
                elif len(batch) == curr_remaining:
                    break
                else:
                    curr_remaining -= len(batch)
            self.row_num = target_row_num
        data_str = maybe_truncate_string(str(self.dataset_files), 200)
        logger.info(f"Finished setting arrow position to {target_row_num} for {data_str}")
