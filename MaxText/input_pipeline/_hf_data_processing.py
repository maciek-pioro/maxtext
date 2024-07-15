"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Input pipeline using Huggingface datasets."""

import ml_collections
import jax
import datasets
import transformers
import grain.python as grain

from input_pipeline import _input_pipeline_utils
import multihost_dataloading


def preprocessing_pipeline(
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    tokenizer_path,
    global_batch_size,
    max_target_length,
    shuffle,
    data_shuffle_seed,
    add_bos=True,
    add_eos=True,
    packing=True,
    shift=True,
    num_threads=1,
):
  """pipeline for preprocessing HF dataset"""

  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  if shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path,
      add_bos_token=add_bos,
      add_eos_token=add_eos,
      model_max_length=max_target_length,
      legacy=False,
  )

  dataset = dataset.map(
      _input_pipeline_utils.tokenization,
      batched=True,
      fn_kwargs={"hf_tokenizer": tokenizer, "max_length": max_target_length - 1},
  )
  dataset = dataset.select_columns(["input_ids"])

  dataset = _input_pipeline_utils.HFDataSource(dataset, dataloading_host_index, dataloading_host_count, num_threads)
  operations = []
  operations.append(_input_pipeline_utils.HFNormalizeFeatures())

  if packing:
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=global_batch_size // jax.process_count(),
            length_struct={"inputs": max_target_length, "targets": max_target_length},
        )
    )
    operations.append(_input_pipeline_utils.ReformatPacking())
  else:
    operations.append(_input_pipeline_utils.PadToMaxLength(max_target_length))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=True))

  if shift:
    operations.append(_input_pipeline_utils.ShiftData(axis=1))

  # Since HuggingFace IterableDataset does not support access through index
  # Indexes generated by dummy_index_sampler is not used.
  # dummy_index_sampler is used as an input place holder for grain.Dataloader
  dummy_index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=1,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=False,
      seed=0,
  )

  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=dummy_index_sampler,
      worker_count=1,  # only supports one worker for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=128),
  )

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen

def make_hf_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    add_bos,
    add_eos,
    process_indices,
  ):
  """Load, preprocess dataset and return iterators"""
  train_ds = datasets.load_dataset(
      config.hf_path,
      data_dir=config.hf_data_dir,
      data_files=config.hf_train_files,
      split="train",
      streaming=True,
      token=config.hf_access_token,
  )
  train_iter = preprocessing_pipeline(
    dataloading_host_index=process_indices.index(jax.process_index()),
    dataloading_host_count=len(process_indices),
    global_mesh=global_mesh,
    dataset=train_ds,
    tokenizer_path=config.tokenizer_path,
    global_batch_size=config.global_batch_size_to_load,
    max_target_length=config.max_target_length,
    shuffle=config.enable_data_shuffling,
    data_shuffle_seed=config.data_shuffle_seed,
    add_bos=add_bos,
    add_eos=add_eos,
  )

  if config.eval_interval > 0:
    eval_ds = datasets.load_dataset(
      config.get("hf_eval_path") or config.hf_path,
      data_dir=config.hf_data_dir,
      data_files=config.hf_eval_files,
      split=config.hf_eval_split,
      streaming=True,
      token=config.hf_access_token,
    )
    if config.eval_per_device_batch_size > 0:
      eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
    else:
      eval_batch_size = config.global_batch_size_to_load
    eval_iter = preprocessing_pipeline(
      dataloading_host_index=process_indices.index(jax.process_index()),
      dataloading_host_count=len(process_indices),
      global_mesh=global_mesh,
      dataset=eval_ds,
      tokenizer_path=config.tokenizer_path,
      global_batch_size=eval_batch_size,
      max_target_length=config.max_target_length,
      shuffle=False,
      data_shuffle_seed=config.data_shuffle_seed,
      add_bos=add_bos,
      add_eos=add_eos,
    )
  else:
    eval_iter = None

  return train_iter, eval_iter
