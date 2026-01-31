# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import time
import logging
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Type, Tuple

import numpy as np
import ray
import torch
from ray.exceptions import ActorDiedError, RayTaskError
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        import time
        val_start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"🔍 Starting Validation at step {self.global_step}")
        print(f"{'='*80}")
        print(f"  Validation batches: {len(self.val_dataloader)}")
        print(f"  Batch size: {self.config.data.val_batch_size}")
        print(f"  Total samples: ~{len(self.val_dataloader) * self.config.data.val_batch_size}")
        print(f"{'='*80}\n")
        
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        skipped_samples = []
        
        for batch_idx, batch_dict in enumerate(self.val_dataloader):
            if batch_idx % 50 == 0:  # 每50个batch打印一次
                print(f"[Validation] Progress: {batch_idx}/{len(self.val_dataloader)} batches "
                      f"({100*batch_idx/len(self.val_dataloader):.1f}%)")
            
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            batch_size = input_ids.size(0)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info.update({
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "global_step": f"{self.global_step}-val",  # 🔥 关键修改
            })
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            
            # Check GPU memory before batch processing
            exceeds_threshold, usage_ratio = self._check_gpu_memory_usage(threshold=0.99)
            used_per_sample = False
            successful_indices = None
            
            if exceeds_threshold:
                print(f"Warning: GPU memory usage ({usage_ratio*100:.1f}%) is high before batch processing. "
                      f"Falling back to per-sample processing to avoid OOM...")
                # Try to free some memory
                try:
                    torch.cuda.empty_cache()
                    time.sleep(1)
                except:
                    pass
                # Skip directly to per-sample processing
                used_per_sample = True
                test_output_gen_batch, successful_indices = self._validate_per_sample(
                    test_batch, test_gen_batch, pad_size, skipped_samples
                )
                # Filter test_batch to only include successful samples
                if successful_indices:
                    test_batch = DataProto.concat([test_batch[i:i+1] for i in successful_indices])
                    input_texts = [input_texts[i] for i in successful_indices]
                else:
                    print("Warning: No successful samples in this batch. Skipping.")
                    continue
            else:
                # Try to process the batch, fallback to per-sample processing on OOM
                test_output_gen_batch = None
                max_retries = 2
                retry_delay = 3  # seconds
                
                for retry in range(max_retries):
                    try:
                        test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
                        break
                    except (ActorDiedError, RayTaskError) as e:
                        # Check if it's an ActorDiedError (either directly or wrapped in RayTaskError)
                        is_actor_died = isinstance(e, ActorDiedError)
                        if isinstance(e, RayTaskError):
                            try:
                                cause = e.as_instanceof_cause()
                                is_actor_died = isinstance(cause, ActorDiedError)
                            except:
                                pass
                        
                        if is_actor_died:
                            if retry < max_retries - 1:
                                print(f"Warning: Actor died during batch validation (likely OOM). Retrying ({retry + 1}/{max_retries}) after {retry_delay}s...")
                                time.sleep(retry_delay)
                                # Reinitialize workers if needed
                                try:
                                    self.actor_rollout_wg.init_model()
                                except Exception as init_error:
                                    print(f"Warning: Failed to reinitialize workers: {init_error}")
                                continue
                            else:
                                # Fallback to per-sample processing
                                print(f"Warning: Batch processing failed after {max_retries} retries. Falling back to per-sample processing to skip OOM samples...")
                                used_per_sample = True
                                test_output_gen_batch, successful_indices = self._validate_per_sample(
                                    test_batch, test_gen_batch, pad_size, skipped_samples
                                )
                                # Filter test_batch to only include successful samples
                                if successful_indices:
                                    test_batch = DataProto.concat([test_batch[i:i+1] for i in successful_indices])
                                    input_texts = [input_texts[i] for i in successful_indices]
                                break
                        else:
                            raise
            
            if test_output_gen_batch is None or len(test_output_gen_batch) == 0:
                print("Error: Failed to generate sequences even with per-sample fallback. Skipping this batch.")
                continue
            
            # Unpad if needed (per-sample processing already handled padding)
            if not used_per_sample and pad_size > 0:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size)
            
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
        
        if skipped_samples:
            print(f"Warning: Skipped {len(skipped_samples)} samples due to OOM during validation.")
        
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}

        # Print validation summary
        val_duration = time.time() - val_start_time
        print(f"\n{'='*80}")
        print(f"✅ Validation Completed at step {self.global_step}")
        print(f"{'='*80}")
        print(f"  Duration: {val_duration:.2f}s ({val_duration/60:.2f} min)")
        print(f"  Processed batches: {len(self.val_dataloader)}")
        print(f"  Processed samples: ~{len(sample_inputs)}")
        print(f"  Skipped samples: {len(skipped_samples)}")
        print(f"  Mean reward score: {reward_score:.4f}")
        print(f"  Metrics: {convert_dict_to_str(val_reward_metrics)}")
        print(f"{'='*80}\n")

        return {"val/reward_score": reward_score, **val_reward_metrics}
    
    def _check_gpu_memory_usage(self, threshold: float = 0.95) -> Tuple[bool, float]:
        """Check if GPU memory usage exceeds threshold.
        
        Args:
            threshold: Memory usage threshold (0.95 = 95%)
            
        Returns:
            tuple: (exceeds_threshold, current_usage_ratio)
        """
        if not torch.cuda.is_available():
            return False, 0.0
        
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
            exceeds = usage_ratio >= threshold
            return exceeds, usage_ratio
        except Exception as e:
            # If we can't check memory, assume it's safe
            print(f"Warning: Could not check GPU memory: {e}")
            return False, 0.0
    
    def _validate_per_sample(
        self, test_batch: DataProto, test_gen_batch: DataProto, pad_size: int, skipped_samples: List[int]
    ) -> Tuple[DataProto, List[int]]:
        """Process validation samples one by one, skipping those that cause OOM.
        
        Returns:
            tuple: (successful_outputs, successful_indices)
        """
        from ..protocol import DataProto
        
        batch_size = len(test_gen_batch)
        successful_outputs = []
        successful_indices = []
        memory_threshold = 0.99  # Skip if memory usage >= 99%
        
        print(f"Processing {batch_size} samples individually to skip OOM samples...")
        
        for i in range(batch_size):
            # Check GPU memory before processing
            exceeds_threshold, usage_ratio = self._check_gpu_memory_usage(memory_threshold)
            if exceeds_threshold:
                prompt_len = len(test_gen_batch[i].batch['input_ids'][0]) if len(test_gen_batch[i].batch['input_ids']) > 0 else 0
                print(f"Warning: Skipping sample {i} - GPU memory usage ({usage_ratio*100:.1f}%) exceeds threshold ({memory_threshold*100:.1f}%) (prompt length: {prompt_len} tokens)")
                skipped_samples.append(i)
                # Try to free some memory
                try:
                    torch.cuda.empty_cache()
                    time.sleep(0.5)  # Brief pause to allow memory cleanup
                except:
                    pass
                continue
            
            try:
                # Extract single sample
                single_gen_batch = test_gen_batch[i:i+1]
                single_gen_batch.meta_info = test_gen_batch.meta_info
                
                # Pad to divisor if needed
                single_gen_batch, single_pad_size = pad_dataproto_to_divisor(
                    single_gen_batch, self.actor_rollout_wg.world_size
                )
                
                # Try to generate
                single_output = self.actor_rollout_wg.generate_sequences(single_gen_batch)
                single_output = unpad_dataproto(single_output, single_pad_size)
                
                successful_outputs.append(single_output)
                successful_indices.append(i)
                
                # Clear cache after successful generation
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                
            except (ActorDiedError, RayTaskError) as e:
                # Check if it's an ActorDiedError
                is_actor_died = isinstance(e, ActorDiedError)
                if isinstance(e, RayTaskError):
                    try:
                        cause = e.as_instanceof_cause()
                        is_actor_died = isinstance(cause, ActorDiedError)
                    except:
                        pass
                
                if is_actor_died:
                    prompt_len = len(test_gen_batch[i].batch['input_ids'][0]) if len(test_gen_batch[i].batch['input_ids']) > 0 else 0
                    # Check memory usage for logging
                    _, usage_ratio = self._check_gpu_memory_usage()
                    print(f"Warning: Skipping sample {i} due to OOM (GPU memory: {usage_ratio*100:.1f}%, prompt length: {prompt_len} tokens)")
                    skipped_samples.append(i)
                    # Reinitialize workers after OOM
                    try:
                        torch.cuda.empty_cache()
                        time.sleep(2)  # Brief delay before reinitializing
                        self.actor_rollout_wg.init_model()
                    except Exception as init_error:
                        print(f"Warning: Failed to reinitialize workers after skipping sample {i}: {init_error}")
                else:
                    # For non-OOM errors, re-raise
                    print(f"Error processing sample {i}: {e}")
                    raise
        
        if not successful_outputs:
            print("Error: All samples failed. Returning empty DataProto.")
            return test_gen_batch[:0], []  # Return empty DataProto with same structure
        
        # Concatenate successful outputs
        result = DataProto.concat(successful_outputs)
        print(f"Successfully processed {len(successful_outputs)}/{batch_size} samples.")
        
        return result, successful_indices

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        # Setup logging - only log to file, avoid console spam
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'training_debug_{int(time.time())}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger = logging.getLogger('training_debug')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        # Don't propagate to root logger to avoid console spam
        logger.propagate = False
        
        logger.info("=" * 80)
        logger.info("Starting training loop")
        logger.info(f"Total epochs: {self.config.trainer.total_epochs}, Training steps: {self.training_steps}")
        logger.info("=" * 80)
        
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        logger.info("Loading checkpoint...")
        try:
            self._load_checkpoint()
            logger.info(f"Checkpoint loaded. Starting from global_step: {self.global_step}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}\n{traceback.format_exc()}")
            raise

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            logger.info("Running validation before training...")
            try:
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
                logger.info(f"Validation completed. Metrics: {val_metrics}")
            except Exception as e:
                logger.error(f"Error during validation: {e}\n{traceback.format_exc()}")
                raise
            if self.config.trainer.val_only:
                logger.info("Validation only mode. Exiting.")
                return

        logger.info("Starting training epochs...")
        for epoch_idx in range(self.config.trainer.total_epochs):
            if epoch_idx == 0 or (epoch_idx + 1) % 10 == 0:  # Log every 10 epochs
                logger.info(f"Starting epoch {epoch_idx + 1}/{self.config.trainer.total_epochs}")
            try:
                for batch_idx, batch_dict in enumerate(self.train_dataloader):
                    self.global_step += 1
                    
                    if self.global_step > self.training_steps:
                        logger.info(f"Reached training steps limit ({self.training_steps}). Stopping.")
                        break

                    metrics, timing_raw = {}, {}
                    try:
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                    except Exception as e:
                        logger.error(f"[Step {self.global_step}] Error creating batch: {e}\n{traceback.format_exc()}")
                        raise

                    # pop those keys for generation
                    try:
                        if "multi_modal_data" in batch.non_tensor_batch.keys():
                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                            )
                            gen_batch.meta_info.update({
                                "min_pixels": self.config.data.min_pixels,
                                "max_pixels": self.config.data.max_pixels,
                            })
                        else:
                            gen_batch = batch.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids"],
                            )

                        with timer("step", timing_raw):
                            # generate a batch
                            # Only log memory every 50 steps or on errors
                            log_memory = (self.global_step % 50 == 0) or (self.global_step == 1)
                            
                            if log_memory and torch.cuda.is_available():
                                try:
                                    free_mem, total_mem = torch.cuda.mem_get_info()
                                    usage_ratio = (total_mem - free_mem) / total_mem if total_mem > 0 else 0.0
                                    logger.info(f"[Step {self.global_step}] GPU memory: {usage_ratio*100:.1f}% ({free_mem/1024**3:.2f}GB free)")
                                except Exception:
                                    pass
                            
                            # Add step info to gen_batch for logging
                            gen_batch.meta_info["global_step"] = self.global_step
                            
                            with timer("gen", timing_raw):  # wg: worker group
                                try:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                except Exception as e:
                                    logger.error(f"[Step {self.global_step}] Error during sequence generation: {e}\n{traceback.format_exc()}")
                                    raise

                            if self.config.algorithm.adv_estimator == "remax":
                                with timer("gen_max", timing_raw):
                                    gen_baseline_batch = deepcopy(gen_batch)
                                    gen_baseline_batch.meta_info["temperature"] = 0
                                    gen_baseline_batch.meta_info["n"] = 1
                                    gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                    batch = batch.union(gen_baseline_output)
                                    reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                    batch.batch["reward_baselines"] = reward_baseline_tensor
                                    del gen_baseline_batch, gen_baseline_output

                            batch.non_tensor_batch["uid"] = np.array(
                                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                            )
                            # repeat to align with repeated responses in rollout
                            batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                            batch = batch.union(gen_batch_output)

                            # balance the number of valid tokens on each dp rank.
                            # Note that this breaks the order of data inside the batch.
                            # Please take care when you implement group based adv computation such as GRPO and rloo
                            self._balance_batch(batch, metrics=metrics)

                            # compute global_valid tokens
                            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                            # compute reward
                            with timer("reward", timing_raw):
                                reward_ref = self.reward_fn.compute_reward.remote(batch)

                            # recompute old_log_probs
                            with timer("old", timing_raw):
                                try:
                                    old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                                    batch = batch.union(old_log_probs)
                                except Exception as e:
                                    logger.error(f"[Step {self.global_step}] Error computing old log probs: {e}\n{traceback.format_exc()}")
                                    raise

                            # compute ref_log_probs
                            if self.use_reference_policy:
                                with timer("ref", timing_raw):
                                    ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                                    batch = batch.union(ref_log_probs)

                            # compute values
                            if self.use_critic:
                                with timer("values", timing_raw):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)

                            with timer("adv", timing_raw):
                                # get token level scores
                                reward_tensor, reward_metrics = ray.get(reward_ref)
                                batch.batch["token_level_scores"] = reward_tensor
                                reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                                metrics.update(reward_metrics)

                                # apply kl penalty if available
                                if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                                    # apply kl penalty to reward
                                    batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                                    metrics.update(kl_metrics)
                                else:
                                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                                # compute advantages, executed on the driver process
                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                )

                            # update critic
                            if self.use_critic:
                                with timer("update_critic", timing_raw):
                                    critic_output = self.critic_wg.update_critic(batch)

                                critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                                metrics.update(critic_metrics)

                            # update actor
                            if self.config.trainer.critic_warmup <= self.global_step:
                                with timer("update_actor", timing_raw):
                                    try:
                                        actor_output = self.actor_rollout_wg.update_actor(batch)
                                    except Exception as e:
                                        logger.error(f"[Step {self.global_step}] Error during actor update: {e}\n{traceback.format_exc()}")
                                        # Log memory after error
                                        if torch.cuda.is_available():
                                            try:
                                                free_mem, total_mem = torch.cuda.mem_get_info()
                                                usage_ratio = (total_mem - free_mem) / total_mem if total_mem > 0 else 0.0
                                                logger.error(f"[Step {self.global_step}] GPU memory after error: {usage_ratio*100:.1f}%")
                                            except:
                                                pass
                                        raise

                                actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                                metrics.update(actor_metrics)

                            # validate
                            if (
                                self.val_reward_fn is not None
                                and self.config.trainer.val_freq > 0
                                and self.global_step % self.config.trainer.val_freq == 0
                            ):
                                with timer("validation", timing_raw):
                                    val_metrics = self._validate()

                                metrics.update(val_metrics)

                            if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                                with timer("save_checkpoint", timing_raw):
                                    self._save_checkpoint()

                            # collect metrics
                            num_gpus = self.resource_pool_manager.get_num_gpus()
                            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                            
                            # Ensure step timing is recorded (fallback if not already recorded)
                            if "step" not in timing_raw:
                                # Calculate step time from other timings if available
                                step_time = sum(timing_raw.values()) if timing_raw else 0.0
                                if step_time > 0:
                                    timing_raw["step"] = step_time
                            
                            # Only compute throughout metrics if 'step' timing is available
                            if "step" in timing_raw:
                                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                            self.logger.log(data=metrics, step=self.global_step)
                            
                            # Print reward-related information after each step
                            reward_info = []
                            if "reward" in metrics:
                                for key, value in metrics.items():
                                    if key.startswith("reward/"):
                                        reward_info.append(f"{key}={value:.4f}")
                            
                            # Also include sequence-level reward metrics
                            if "data/sequence_score_mean" in metrics:
                                reward_info.append(f"sequence_score={metrics['data/sequence_score_mean']:.4f}")
                            if "data/sequence_reward_mean" in metrics:
                                reward_info.append(f"sequence_reward={metrics['data/sequence_reward_mean']:.4f}")
                            
                            if reward_info:
                                logger.info(f"[Step {self.global_step}] Reward: {', '.join(reward_info)}")
                    
                    except Exception as e:
                        logger.error(f"[Step {self.global_step}] Fatal error in training step: {e}\n{traceback.format_exc()}")
                        raise
            except BaseException as e:
                logger.error(f"Error in epoch {epoch_idx + 1}, batch loop: {e}\n{traceback.format_exc()}")
                print(f"Error in epoch {epoch_idx + 1}, batch loop: {e}")
                logger.info("Saving checkpoint due to error...")
                print("Saving checkpoint due to error...")
                try:
                    self._save_checkpoint()
                    logger.info("Checkpoint saved successfully after error.")
                    print("Checkpoint saved successfully after error.")
                except Exception as save_e:
                    logger.error(f"Failed to save checkpoint during error handling: {save_e}\n{traceback.format_exc()}")
                    print(f"Failed to save checkpoint during error handling: {save_e}")
                raise

        logger.info(f"Training loop completed. Final global_step: {self.global_step}")

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
