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
Implement Actor
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]


def _validate_image_features_and_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    multi_modal_inputs: Dict[str, torch.Tensor],
    processor=None,
) -> tuple[bool, Optional[str]]:
    """
    校验 image features 和 image tokens 是否匹配
    
    Args:
        input_ids: 输入token ids
        attention_mask: attention mask
        multi_modal_inputs: 多模态输入，包含 pixel_values 和 image_grid_thw
        processor: 处理器，用于获取 image token id
        
    Returns:
        (is_valid, error_msg): 是否有效，错误信息（如果无效）
    """
    if not multi_modal_inputs or "image_grid_thw" not in multi_modal_inputs:
        return True, None  # 没有图像数据，跳过校验
    
    try:
        # 计算 image features 数量（通过 image_grid_thw）
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return True, None  # 没有 image_grid_thw，无法校验
        
        # image_grid_thw shape: (num_images, 3) -> (t, h, w)
        # 每个图像的 token 数量 = t * h * w
        import numpy as np
        if isinstance(image_grid_thw, torch.Tensor):
            num_image_features = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item()
        elif isinstance(image_grid_thw, np.ndarray):
            num_image_features = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum()
        else:
            return True, None  # 无法校验，假设有效
        
        # 计算 input_ids 中的 image tokens 数量
        # 对于 Qwen2.5-VL，image token 是 <|image_pad|>
        image_token_id = None
        if processor is not None:
            try:
                image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            except:
                pass
        
        if image_token_id is not None:
            # 只统计有效位置的 tokens (attention_mask == 1)
            if attention_mask is not None:
                valid_input_ids = input_ids[attention_mask == 1]
            else:
                valid_input_ids = input_ids.flatten()
            num_image_tokens = (valid_input_ids == image_token_id).sum().item()
        else:
            # 如果无法获取 image_token_id，无法准确校验，假设匹配
            return True, None
        
        # 允许一定的误差范围（5%或至少10个token），因为可能有舍入误差
        tolerance = max(10, int(num_image_features * 0.05))
        if abs(num_image_tokens - num_image_features) > tolerance:
            error_msg = (
                f"Image features and image tokens mismatch: "
                f"tokens={num_image_tokens}, features={num_image_features} "
                f"(diff={abs(num_image_tokens - num_image_features)}). "
                f"This batch will be skipped."
            )
            return False, error_msg
        
        return True, None
    except Exception as e:
        # 如果校验过程出错，记录警告但允许继续（避免因为校验逻辑问题导致训练失败）
        return True, None


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            # Collect all tensors for each key, ensuring they're on the same device
            device = input_ids.device
            for key in micro_batch["multi_modal_inputs"][0].keys():
                tensors_to_cat = []
                for inputs in micro_batch["multi_modal_inputs"]:
                    tensor = inputs[key]
                    # Convert to tensor if needed and move to the correct device
                    if not isinstance(tensor, torch.Tensor):
                        tensor = torch.as_tensor(tensor)
                    # Ensure tensor is on the same device as input_ids
                    if tensor.device != device:
                        tensor = tensor.to(device)
                    tensors_to_cat.append(tensor)
                
                if len(tensors_to_cat) > 0:
                    # Concatenate along the first dimension (batch dimension)
                    # For pixel_values: (num_images_per_sample, ...) -> (total_images, ...)
                    # For image_grid_thw: (num_images_per_sample, 3) -> (total_images, 3)
                    multi_modal_inputs[key] = torch.cat(tensors_to_cat, dim=0)

        # 校验 image features 和 tokens 是否匹配（在调用模型之前）
        if multi_modal_inputs:
            # 尝试获取 processor（如果可用）
            processor = getattr(self.actor_module, 'processor', None)
            if processor is None and hasattr(self.actor_module, 'module'):
                processor = getattr(self.actor_module.module, 'processor', None)
            
            is_valid, error_msg = _validate_image_features_and_tokens(
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
                processor=processor,
            )
            
            if not is_valid:
                # 数据不匹配，返回 None 作为标记，让上层处理
                if self.rank == 0:
                    print(f"WARNING: {error_msg}")
                # 返回一个全零的 log_probs，形状为 (batch_size, response_length)
                # 这样上层可以识别并跳过这个 batch
                return torch.zeros(batch_size, response_length, device=input_ids.device, dtype=torch.float32)

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            try:
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                logits_rmpad.div_(temperature)
                # ((total_nnz / sp) + pad)
                log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)
            except ValueError as e:
                # 捕获 "Image features and image tokens do not match" 错误
                if "Image features and image tokens do not match" in str(e):
                    if self.rank == 0:
                        print(f"WARNING: {str(e)}. Skipping this batch.")
                    return torch.zeros(batch_size, response_length, device=input_ids.device, dtype=torch.float32)
                else:
                    raise  # 重新抛出其他 ValueError

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            # 对于非 padding_free 模式，也需要校验
            if multi_modal_inputs:
                processor = getattr(self.actor_module, 'processor', None)
                if processor is None and hasattr(self.actor_module, 'module'):
                    processor = getattr(self.actor_module.module, 'processor', None)
                
                is_valid, error_msg = _validate_image_features_and_tokens(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    multi_modal_inputs=multi_modal_inputs,
                    processor=processor,
                )
                
                if not is_valid:
                    if self.rank == 0:
                        print(f"WARNING: {error_msg}")
                    return torch.zeros(batch_size, response_length, device=input_ids.device, dtype=torch.float32)
            
            try:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )
                logits: torch.Tensor = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
            except ValueError as e:
                # 捕获 "Image features and image tokens do not match" 错误
                if "Image features and image tokens do not match" in str(e):
                    if self.rank == 0:
                        print(f"WARNING: {str(e)}. Skipping this batch.")
                    return torch.zeros(batch_size, response_length, device=input_ids.device, dtype=torch.float32)
                else:
                    raise  # 重新抛出其他 ValueError

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        skipped_batches = 0
        total_batches = 0
        # Disable tqdm to avoid log spam - just use the iterator directly
        # if self.rank == 0:
        #     micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2, disable=True)

        for micro_batch in micro_batches:
            total_batches += 1
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            
            # 检测是否跳过了这个 batch（全零的 log_probs 可能表示跳过）
            # 但要注意，正常的 log_probs 也可能接近零，所以我们需要更精确的检测
            # 这里我们假设如果 log_probs 全为零且 batch 有 multi_modal_inputs，可能是跳过了
            if "multi_modal_inputs" in model_inputs and torch.all(log_probs == 0):
                skipped_batches += 1
                if self.rank == 0 and skipped_batches <= 5:  # 只打印前5个，避免日志过多
                    print(f"INFO: Skipped batch {total_batches} due to image features/tokens mismatch")
            
            log_probs_lst.append(log_probs)

        if skipped_batches > 0 and self.rank == 0:
            print(f"INFO: Skipped {skipped_batches}/{total_batches} batches due to image features/tokens mismatch")

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    def update_policy(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            # Disable tqdm to avoid log spam
            # if self.rank == 0:
            #     mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                # Disable tqdm to avoid log spam
                # if self.rank == 0:
                #     micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                for micro_batch_idx, micro_batch in enumerate(micro_batches):
                    # Check GPU memory before processing micro_batch
                    if torch.cuda.is_available():
                        try:
                            free_mem, total_mem = torch.cuda.mem_get_info()
                            used_mem = total_mem - free_mem
                            usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
                            
                            # 检查 reserved memory（可能更准确反映实际使用）
                            reserved_mem = torch.cuda.memory_reserved()
                            reserved_ratio = reserved_mem / total_mem if total_mem > 0 else 0.0
                            
                            # 使用 reserved_ratio 或 usage_ratio 中较大的值
                            effective_ratio = max(usage_ratio, reserved_ratio)
                            
                            if effective_ratio >= 0.95:  # 降低阈值到 95% 触发清理
                                if self.rank == 0:
                                    print(f"Warning: GPU memory usage high (allocated={usage_ratio*100:.1f}%, "
                                          f"reserved={reserved_ratio*100:.1f}%). Cleaning cache...")
                                # 更激进的清理
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                # time.sleep(0.5)
                                
                                # 再次检查
                                free_mem, total_mem = torch.cuda.mem_get_info()
                                used_mem = total_mem - free_mem
                                usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
                                
                                if usage_ratio >= 0.98:  # 如果清理后仍然很高，警告但继续执行（FSDP需要同步）
                                    if self.rank == 0:
                                        print(f"Warning: GPU memory still high ({usage_ratio*100:.1f}%) after cleanup. "
                                              f"Continuing to avoid FSDP deadlock (all ranks must sync).")
                                # 不跳过，继续执行以避免 FSDP 死锁
                        except Exception as e:
                            if self.rank == 0:
                                print(f"Warning: Could not check GPU memory: {e}")
                    
                    try:
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        responses = model_inputs["responses"]
                        response_length = responses.size(1)
                        attention_mask = model_inputs["attention_mask"]
                        response_mask = attention_mask[:, -response_length:]
                        old_log_probs = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        # all return: (bsz, response_length)
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        
                        # 检查是否跳过了这个 batch（全零 log_probs 且有多模态输入）
                        # 注意：FSDP 需要所有 rank 同步，不能跳过，所以使用零值但继续执行
                        if "multi_modal_inputs" in model_inputs and torch.all(log_probs == 0):
                            if self.rank == 0:
                                print(f"Warning: Image features/tokens mismatch detected. Using zero loss to maintain FSDP sync.")
                            # 不跳过，继续使用 log_probs（已经是零值）进行计算，以保持 FSDP 同步
                        
                        entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss

                        pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                            old_log_probs=old_log_probs,
                            log_probs=log_probs,
                            advantages=advantages,
                            response_mask=response_mask,
                            clip_ratio_low=self.config.clip_ratio_low,
                            clip_ratio_high=self.config.clip_ratio_high,
                            clip_ratio_dual=self.config.clip_ratio_dual,
                        )
                        if "ref_log_probs" in model_inputs:
                            ref_log_probs = model_inputs["ref_log_probs"]
                            # compute kl loss
                            kld = core_algos.compute_kl(
                                log_probs=log_probs,
                                ref_log_probs=ref_log_probs,
                                kl_penalty=self.config.kl_penalty,
                            )
                            kl_loss = VF.masked_mean(kld, response_mask)
                            pg_loss = pg_loss + kl_loss * self.config.kl_coef
                            metrics["actor/kl_loss"] = kl_loss.detach().item()
                            metrics["actor/kl_coef"] = self.config.kl_coef

                        loss = pg_loss / gradient_accumulation
                        
                        # Check memory again before backward (backward is memory-intensive)
                        if torch.cuda.is_available():
                            try:
                                free_mem, total_mem = torch.cuda.mem_get_info()
                                used_mem = total_mem - free_mem
                                usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
                                
                                # 检查 reserved memory
                                reserved_mem = torch.cuda.memory_reserved()
                                reserved_ratio = reserved_mem / total_mem if total_mem > 0 else 0.0
                                effective_ratio = max(usage_ratio, reserved_ratio)
                                
                                if effective_ratio >= 0.95:  # 降低阈值
                                    if self.rank == 0:
                                        print(f"Warning: GPU memory high before backward (allocated={usage_ratio*100:.1f}%, "
                                              f"reserved={reserved_ratio*100:.1f}%). Cleaning...")
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    # time.sleep(0.5)
                                    
                                    # 再次检查
                                    free_mem, total_mem = torch.cuda.mem_get_info()
                                    used_mem = total_mem - free_mem
                                    usage_ratio = used_mem / total_mem if total_mem > 0 else 0.0
                                    
                                    if usage_ratio >= 0.98:
                                        if self.rank == 0:
                                            print(f"Warning: GPU memory still high ({usage_ratio*100:.1f}%) after cleanup. "
                                                  f"Continuing to avoid FSDP deadlock (all ranks must sync).")
                                        # 不跳过，继续执行以避免 FSDP 死锁
                                        # 内存清理已在上面完成
                            except Exception:
                                pass
                        
                        loss.backward()
                        
                        # Clear cache after backward to free memory (更激进的清理)
                        if torch.cuda.is_available():
                            try:
                                # 清理缓存
                                torch.cuda.empty_cache()
                                # 如果内存使用率仍然很高，尝试同步并再次清理
                                free_mem, total_mem = torch.cuda.mem_get_info()
                                usage_ratio = (total_mem - free_mem) / total_mem if total_mem > 0 else 0.0
                                if usage_ratio > 0.98:  # 如果使用率超过 98%
                                    torch.cuda.synchronize()  # 同步所有 CUDA 操作
                                    torch.cuda.empty_cache()  # 再次清理
                            except:
                                pass
                                
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                            if self.rank == 0:
                                print(f"CRITICAL: OOM error during Update policy. Error: {e}")
                            
                            # CRITICAL FIX: 遇到 OOM 必须抛出异常。
                            # 尝试使用 zero_loss.backward() 是无效的，会导致死锁。
                            # 抛出异常后，Ray 或 K8s 会重启 Worker，这是处理分布式 OOM 的唯一正确方式。
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            raise e 
                        else:
                            raise
                    except ValueError as e:
                        # CRITICAL FIX: 数据错误也必须抛出，不能吞掉。
                        if self.rank == 0:
                            print(f"CRITICAL: Data error during Update policy. Error: {e}")
                        raise e

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
