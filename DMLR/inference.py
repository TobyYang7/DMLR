import torch
import torch.nn as nn
from .reward import RewardModel
from .logger import log
from typing import Dict, Union, Optional, Tuple, List
from pathlib import Path
import numpy as np
import os
import csv
from PIL import Image
from .prompts import SYSTEM_PROMPT
try:
    from colorama import Fore, Style
    init_colorama = True
except ImportError:
    class Fore:
        BLUE = '\033[94m'
        GREEN = '\033[92m'

    class Style:
        RESET_ALL = '\033[0m'
    init_colorama = False


def get_stop_reason_vl(outputs, input_length, max_new_tokens, tokenizer):
    '''
    Determine the stop reason for generation
    
    Args:
        outputs: generated token ids (tensor or list)
        input_length: length of input tokens
        max_new_tokens: maximum number of new tokens
        tokenizer: tokenizer object
    
    Returns:
        stop_reason: string indicating why generation stopped
    '''
    if isinstance(outputs, torch.Tensor):
        generated_tokens = outputs[0] if outputs.dim() > 1 else outputs
    else:
        generated_tokens = outputs[0] if isinstance(outputs, list) and len(outputs) > 0 else outputs

    if isinstance(generated_tokens, torch.Tensor):
        generated_tokens = generated_tokens.tolist()

    generated_length = len(generated_tokens)
    new_tokens_length = generated_length - input_length

    # Check if reached max_new_tokens
    if new_tokens_length >= max_new_tokens:
        return "length"

    # Check if ended with eos_token
    if tokenizer.eos_token_id is not None:
        if generated_tokens and generated_tokens[-1] == tokenizer.eos_token_id:
            return "eos_token"

    # Check if ended with pad_token
    if tokenizer.pad_token_id is not None:
        if generated_tokens and generated_tokens[-1] == tokenizer.pad_token_id:
            return "pad_token"

    # Default: other reason
    return "other"


def _extract_visual_latents(model, inputs, thought_idx, target_hidden_size):
    """
    Derive an initialization tensor for latent thought tokens using visual features.

    Args:
        model: vision-language model providing a vision tower and (optionally) projector.
        inputs: model inputs dict containing pixel values.
        thought_idx: [start, end] indices specifying where latent tokens live.

    Returns:
        A tensor shaped like the latent token span or None when unavailable.
    """
    pixel_values = inputs.get('pixel_values')
    if pixel_values is None:
        return None

    latent_length = thought_idx[1] - thought_idx[0]
    if latent_length <= 0:
        return None

    vision_tower = getattr(model, 'vision_tower', None)
    core_model = getattr(model, 'model', None)
    if vision_tower is None and core_model is not None:
        vision_tower = getattr(core_model, 'vision_tower', None)

    projector = getattr(model, 'mm_projector', None)
    if projector is None and core_model is not None:
        projector = getattr(core_model, 'mm_projector', None)

    try:
        with torch.no_grad():
            vision_hidden = None
            if vision_tower is not None:
                vision_outputs = vision_tower(pixel_values)
                if isinstance(vision_outputs, (tuple, list)):
                    vision_hidden = vision_outputs[0]
                elif hasattr(vision_outputs, 'last_hidden_state'):
                    vision_hidden = vision_outputs.last_hidden_state
                else:
                    vision_hidden = vision_outputs

            if vision_hidden is None:
                return None

            if vision_hidden.dim() == 4:
                vision_hidden = vision_hidden.flatten(2).transpose(1, 2)

            if projector is not None and vision_hidden.dim() == 3:
                vision_hidden = projector(vision_hidden)

            if vision_hidden.dim() == 3:
                pooled = vision_hidden.mean(dim=1, keepdim=True)
            elif vision_hidden.dim() == 2:
                pooled = vision_hidden.unsqueeze(1)
            else:
                return None

            if pooled.size(-1) != target_hidden_size:
                return None

            pooled = pooled.expand(-1, latent_length, -1).contiguous()
            return pooled
    except Exception as exc:
        try:
            log.warning(f"Failed to initialize latent tokens from vision features: {exc}")
        except Exception:
            pass

    return None


def build_vl_inputs(
    processor,
    num_thought_tokens: int,
    question: str,
    image=None,
    messages=None,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    """
    Build inputs for VL model with thought tokens
    
    Args:
        processor: Qwen2VL processor
        num_thought_tokens: number of thought tokens
        question: question text
        image: PIL image or None
        messages: chat messages
        device: device to use
        data_name: dataset name
        model_name: model name
    
    Returns:
        inputs: processed inputs dict
        thought_idx: [start_idx, end_idx] of thought tokens
    """
    if num_thought_tokens <= 0:
        raise ValueError('You must specify a positive int for num_thought_tokens')

    # Use endoftext tokens for Qwen VL models
    latent_thought_tokens = '<|endoftext|>' * num_thought_tokens
    problem_type = 'code reasoning' if 'cruxeval' in data_name else 'math'

    # Detect if it's a multiple choice question
    is_multiple_choice = 'Choice' in question or 'choice' in question or any(x in question for x in ['\nA:', '\nB:', '\nC:', '\nD:'])

    # Adjust answer format instruction based on question type
    if is_multiple_choice:
        answer_instruction = (
            'IMPORTANT: This is a multiple choice question.\n'
            '- First, solve the problem step by step.\n'
            '- Then, provide your final answer as the option letter (A, B, C, or D) within \\boxed{{}}.\n'
            '- Example: \\boxed{{A}} or \\boxed{{B}}\n'
        )
    else:
        answer_instruction = (
            'IMPORTANT: You MUST always put your final numerical answer within \\boxed{{}}.\n'
        )

    # Build the input content with thought tokens
    input_content = (
        f'PROBLEM: {question}\n\n'
        f'The following special tokens represent YOUR INTERNAL THINKING SPACE where your reasoning happens implicitly.\n'
        f'You do NOT need to output explicit reasoning steps'
        f'After these tokens, directly provide your final answer.'
        f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
    )

    if image is not None and isinstance(image, str):
        lower = image.lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            if os.path.exists(image):
                try:
                    image = Image.open(image).convert("RGB")
                except Exception as exc:
                    raise ValueError(f"Failed to open image file: {image}: {exc}")
            else:
                raise ValueError(f"Image path does not exist: {image}")

    # Build messages
    if messages is None:
        # Check if SYSTEM_PROMPT exists and is not empty
        if SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
            input_messages = [{'role': 'system', 'content': SYSTEM_PROMPT.strip()}]
        else:
            input_messages = []

        if image is not None:
            input_messages.append({
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': input_content}
                ]
            })
        else:
            input_messages.append({'role': 'user', 'content': input_content})
    else:
        # Use provided messages and append thought tokens to the last user message
        # Ensure messages is a list
        if isinstance(messages, dict):
            input_messages = [messages]
        else:
            input_messages = messages.copy() if isinstance(messages, list) else [messages]

        # Check if system prompt already exists
        has_system_prompt = any(
            isinstance(msg, dict) and msg.get('role') == 'system' for msg in input_messages
            if isinstance(msg, dict)
        )

        # Add SYSTEM_PROMPT if not present
        if not has_system_prompt and SYSTEM_PROMPT and SYSTEM_PROMPT.strip():
            input_messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT.strip()})

        # IMPORTANT: Add actual image to messages if provided
        if image is not None and isinstance(input_messages[-1]['content'], list):
            # Replace placeholder {'type': 'image'} with actual image
            for i, content_item in enumerate(input_messages[-1]['content']):
                if isinstance(content_item, dict) and content_item.get('type') == 'image':
                    input_messages[-1]['content'][i] = {'type': 'image', 'image': image}
                    break

        if isinstance(input_messages[-1]['content'], list):
            input_messages[-1]['content'].append({'type': 'text', 'text': f"\n\n{input_content}"})
        else:
            input_messages[-1]['content'] = f"{input_messages[-1]['content']}\n\n{input_content}"

    # Apply chat template
    text = processor.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Process inputs with or without image
    if image is not None:
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        ).to(device)

    # Calculate the start and end index of the latent thought tokens
    input_ids = inputs['input_ids'][0]
    pure_input_length = len(input_ids)
    # For Qwen models, adjust the index
    input_thought_start_idx = pure_input_length - 1 - num_thought_tokens - 1
    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    thought_idx = [input_thought_start_idx, input_thought_end_idx]

    # Add generation prompt tokens
    gen_prompt = '<|im_start|>assistant\n'
    gen_prompt_ids = torch.tensor(
        processor.tokenizer.encode(gen_prompt, add_special_tokens=False),
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    # Append the generation_prompt tokens to inputs['input_ids']
    inputs['input_ids'] = torch.cat([inputs['input_ids'], gen_prompt_ids], dim=1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'], device=device)

    # Handle image-related inputs
    if 'pixel_values' in inputs:
        # Keep pixel_values as is
        pass
    if 'image_grid_thw' in inputs:
        # Keep image_grid_thw as is
        pass

    return inputs, thought_idx


def get_confidence(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k=10,
    thought_positions=None,
):
    """
    Calculate confidence score based on top-k probabilities
    
    Args:
        model: VL model
        inputs: input dict with inputs_embeds
        thought_idx: [start_idx, end_idx] of thought tokens (used if thought_positions is None)
        thought_hidden_states: hidden states of thought tokens [num_thought_tokens, hidden_size]
        k: top-k for confidence calculation
        thought_positions: list of actual positions of thought tokens in the sequence (if non-contiguous)
    
    Returns:
        confidence score (higher is better)
    """
    # If thought_positions is provided, use it (for non-contiguous thought tokens after visual insertion)
    # Otherwise, assume thought tokens are contiguous at thought_idx
    if thought_positions is not None:
        # Replace thought tokens at their actual positions
        for i, pos in enumerate(thought_positions):
            inputs['inputs_embeds'][0, pos] = thought_hidden_states[i]
        # Calculate reward on these positions
        logits = model(**inputs, return_dict=True)['logits'][0]
        probs = torch.softmax(logits, dim=-1)
        confidence = 0.0
        for pos in thought_positions:
            # logits[pos] predicts the next token after position pos
            topk = torch.topk(probs[pos], k=k, largest=True)[0]
            confidence -= torch.sum(torch.log(topk + 1e-10)) / k
        num_tokens = len(thought_positions)
    else:
        # Original behavior: contiguous thought tokens
        inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
        logits = model(**inputs, return_dict=True)['logits'][0]
        probs = torch.softmax(logits, dim=-1)
        confidence = 0.0
        # Note: logits[i] predicts token at position i+1
        # So we calculate confidence for predictions at each thought token position
        for idx in range(thought_idx[0], thought_idx[1]):
            topk = torch.topk(probs[idx], k=k, largest=True)[0]
            confidence -= torch.sum(torch.log(topk + 1e-10)) / k
        num_tokens = thought_idx[1] - thought_idx[0]
    
    return confidence / num_tokens if num_tokens > 0 else 0.0


def _resolve_image_grid_tuple(image_grid_thw) -> Optional[Tuple[int, int, int]]:
    """
    Convert image grid metadata into a (T, H, W) tuple when available.
    """
    if image_grid_thw is None:
        return None

    values = None
    if isinstance(image_grid_thw, torch.Tensor):
        grid = image_grid_thw.detach().cpu()
        if grid.ndim >= 2:
            grid = grid[0]
        values = grid.tolist()
    elif isinstance(image_grid_thw, (list, tuple)):
        values = list(image_grid_thw)
    else:
        try:
            values = list(image_grid_thw)
        except TypeError:
            values = None

    if not values or len(values) < 3:
        return None

    return int(values[0]), int(values[1]), int(values[2])


def compute_image_token_meta(
    input_ids: torch.Tensor,
    processor,
    model=None,
) -> Dict[str, Union[int, torch.Tensor]]:
    """
    Pre-compute image token indices for faster lookup during optimization.
    """
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    vs_mask = input_ids == vision_start_id
    ve_mask = input_ids == vision_end_id

    if vs_mask.any() and ve_mask.any():
        vs_idx = torch.where(vs_mask)[0][0].item()
        ve_idx = torch.where(ve_mask)[0][0].item()
        image_positions = torch.arange(vs_idx + 1, ve_idx, device=input_ids.device)
    else:
        image_token_id = None
        if hasattr(processor, "image_token"):
            image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        elif hasattr(processor.tokenizer, "image_token_id"):
            image_token_id = processor.tokenizer.image_token_id
        elif model is not None and hasattr(model.config, "image_token_id"):
            image_token_id = model.config.image_token_id
        # else:
            # <--- FIX 4: REMOVED bad fallback logic that used "<|image_pad|>"
            # image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        if image_token_id is None:
            raise ValueError(
                "Could not determine a valid image_token_id and <|vision_start|>/<|vision_end|> boundaries not found."
            )

        image_positions = torch.where(input_ids == image_token_id)[0]
        if image_positions.numel() == 0:
            raise ValueError(f"No image tokens (id={image_token_id}) or vision boundaries found in input_ids!")

    return {
        "positions": image_positions,
        "start": image_positions[0].item(),
        "end": image_positions[-1].item() + 1,
        "count": image_positions.numel(),
    }


def generate_vl(
    processor,
    model,
    reward_model: RewardModel,
    question: str,
    image=None,
    messages=None,
    num_thought_tokens: int = 8,
    lr: float = 0.005,
    sigma: float = 20.0,
    sigma_decay: float = 0.95,
    max_rl_steps: int = 20,
    reward_threshold: float = -1,
    max_new_tokens: int = 2048,
    use_auto_grad: bool = True,
    disable_conf_reward: bool = False,
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    device=None,
    visual_only: bool = False,
    num_selected_patches: Optional[int] = 32,
    visual_token_viz: bool = False,
    visual_token_viz_dir: Optional[str] = None,
    data_idx: Optional[int] = None,
    visual_insert_stride: int = 1,
    visual_injection_start_step: int = 0,
    visual_injection_interval: int = 1,
    initial_patch_count: Optional[int] = None,
    patch_increment: int = 0,
    reward_csv_path: Optional[str] = None,
    **kwargs,
):
    """
    generate_vl() - modified version using attention-selected top-k image token insertion
    instead of residual visual injection.
    """

    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model.eval()

    # 1) build input with latent tokens
    inputs, thought_idx = build_vl_inputs(
        processor=processor,
        num_thought_tokens=num_thought_tokens,
        question=question,
        image=image,
        messages=messages,
        data_name=data_name,
        model_name=model_name,
        device=resolved_device,
    )

    # 2) embed text (and image placeholders)
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    input_ids_saved = inputs['input_ids'].clone()
    inputs.pop('input_ids')

    # 2.1) If visual_only is True, replace thought token embeddings with visual latents
    if visual_only:
        vision_latents = _extract_visual_latents(
            model,
            inputs,
            thought_idx,
            inputs_embeds.size(-1),
        )
        if vision_latents is not None:
            if vision_latents.size(-1) == inputs_embeds.size(-1):
                inputs_embeds[0, thought_idx[0]:thought_idx[1]] = vision_latents[0].to(
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
            else:
                if verbose >= 1:
                    log.warning(
                        "Vision latent hidden size mismatch. Skip visual initialization for latent tokens."
                    )

    # 3) initialize latent token embeddings (no visual residual now)
    base_init = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()
    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            base_init.detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = base_init.clone()

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()

    embed_device = inputs_embeds.device
    reward_log_entries = []
    prev_patch_count = None
    locked_patch_ids: Dict[int, List[int]] = {}
    current_step_patch_ids: Dict[int, List[int]] = {}
    max_patch_limit = None
    if num_selected_patches is not None and num_selected_patches > 0:
        max_patch_limit = num_selected_patches

    if initial_patch_count is None or initial_patch_count <= 0:
        if max_patch_limit is not None:
            current_patch_budget = max_patch_limit
        else:
            current_patch_budget = None
    else:
        current_patch_budget = initial_patch_count if max_patch_limit is None else min(initial_patch_count, max_patch_limit)
        current_patch_budget = max(1, current_patch_budget)

    patch_increment = max(0, patch_increment)
    
    # Initialize variables for optimized patches
    optimized_patch_embeds = None  # Will store patch embeddings as Parameter when patches are selected
    total_optimized_tokens = num_thought_tokens  # Start with just thought tokens
    optimized_start_idx = thought_idx[0]  # Start position of optimized block
    patch_structure = {}  # Maps think_offset -> number of patches inserted after it

    # =========================
    # RL LOOP
    # =========================
    for step in range(max_rl_steps):
        current_step_patch_ids = {}
        patches_selected_this_step = 0
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()

        # 3.1 exploration noise
        # If patches are being optimized, use combined optimization block
        if optimized_patch_embeds is not None:
            # Combined optimization: thought tokens + patches
            combined_embeds = torch.cat([
                thought_hidden_states,
                optimized_patch_embeds
            ], dim=0)
            epsilon = torch.normal(mean=0.0, std=sigma, size=combined_embeds.shape).to(embed_device)
            candidate_combined = combined_embeds.detach() + epsilon
            candidate_latent = candidate_combined[:num_thought_tokens]
            candidate_patches = candidate_combined[num_thought_tokens:]
        else:
            # Only thought tokens
            epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(embed_device)
            candidate_latent = thought_hidden_states.detach() + epsilon
            candidate_patches = None

        # 3.3 write candidate latents into copy
        inputs_embeds_step = inputs_embeds.clone()
        inputs_embeds_step[0, thought_idx[0]:thought_idx[1]] = candidate_latent

        # 3.4 forward pass w/ attention
        outputs = model(
            inputs_embeds=inputs_embeds_step,
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
        )

        attentions = outputs.attentions
        hidden_states = outputs.hidden_states[-1]

        # Free memory from outputs
        del outputs
        # torch.cuda.empty_cache()

        # =========================
        # Attention-based Top-K image token selection + embedding insertion
        # Multi-token version: each thought token matches its own visual tokens
        #
        # Control visual injection timing:
        # - visual_injection_start_step: delay injection until this step
        # - visual_injection_interval: inject every N steps (1 = every step, 2 = every other step, etc.)
        # =========================
        should_inject_visual = (
            step >= visual_injection_start_step and
            (visual_injection_interval <= 1 or (step - visual_injection_start_step) % visual_injection_interval == 0)
        )

        if should_inject_visual:
            image_token_meta = compute_image_token_meta(input_ids_saved[0], processor, model)
            image_start, image_end = image_token_meta["start"], image_token_meta["end"]

            valid_attn = [attn for attn in attentions if attn is not None]
            avg_attention = torch.cat(valid_attn, dim=1).mean(dim=1)  # (1, seq_len, seq_len)

            # Process each thought token separately with stride
            num_thought = thought_idx[1] - thought_idx[0]
            all_selected_tokens = {}  # Map from think_offset to selected visual tokens

            for think_offset in range(num_thought):
                if think_offset % visual_insert_stride != 0:
                    continue

                current_thought_idx = thought_idx[0] + think_offset
                att_to_images = avg_attention[0, current_thought_idx, image_start:image_end]
                total_image_tokens = att_to_images.size(0)

                if current_patch_budget is not None:
                    k_limit = min(int(current_patch_budget), total_image_tokens)
                elif max_patch_limit is not None:
                    k_limit = min(max_patch_limit, total_image_tokens)
                else:
                    k_limit = total_image_tokens

                if k_limit <= 0:
                    continue

                sorted_rel_indices = torch.argsort(att_to_images, descending=True)
                chosen_abs_ids: List[int] = []

                locked_ids = locked_patch_ids.get(think_offset, [])
                if locked_ids:
                    for pid in locked_ids:
                        if pid < image_start or pid >= image_end:
                            continue
                        if pid not in chosen_abs_ids:
                            chosen_abs_ids.append(pid)
                        if len(chosen_abs_ids) >= k_limit:
                            break

                if len(chosen_abs_ids) < k_limit:
                    for rel_idx in sorted_rel_indices.tolist():
                        abs_idx = image_start + rel_idx
                        if abs_idx in chosen_abs_ids:
                            continue
                        chosen_abs_ids.append(abs_idx)
                        if len(chosen_abs_ids) >= k_limit:
                            break

                if not chosen_abs_ids:
                    continue

                abs_topk = torch.tensor(chosen_abs_ids, device=att_to_images.device, dtype=torch.long)
                current_step_patch_ids[think_offset] = chosen_abs_ids
                patches_selected_this_step += len(chosen_abs_ids)

                if visual_only:
                    picked_image_embeds = hidden_states[0, abs_topk, :]
                else:
                    picked_image_embeds = inputs_embeds_step[0, abs_topk, :]

                all_selected_tokens[think_offset] = picked_image_embeds
                
                # Track patch structure for optimization
                patch_structure[think_offset] = len(chosen_abs_ids)

                if verbose >= 1:
                    rel_positions = (abs_topk - image_start).cpu().tolist()
                    log.info(
                        f"Step {step}, Think Token {think_offset} (stride={visual_insert_stride}): "
                        f"Selected image token IDs: {abs_topk.cpu().tolist()} (relative: {rel_positions})"
                    )

            # Build new embeddings: interleave thought tokens with their matched visual tokens (only at stride positions)
            # Format: [prefix] [think_0] [visual_0] [think_1] [think_2] [visual_2] ... [suffix]
            embed_parts = [inputs_embeds_step[:, :thought_idx[0], :]]  # prefix before first thought token
            
            # Initialize optimized patches if this is the first time selecting patches
            if optimized_patch_embeds is None and not disable_conf_reward and use_auto_grad:
                # Collect all patches to be inserted
                all_patches_list = []
                for think_offset in range(num_thought):
                    if think_offset in all_selected_tokens:
                        all_patches_list.append(all_selected_tokens[think_offset])
                if all_patches_list:
                    # Initialize patch embeddings as optimizable parameters
                    total_patches = sum(p.size(0) for p in all_patches_list)
                    patch_init = torch.cat(all_patches_list, dim=0).detach()
                    optimized_patch_embeds = torch.nn.Parameter(
                        patch_init.requires_grad_(True)
                    )
                    # Update optimizer to include patches
                    optimizer = torch.optim.Adam(
                        [thought_hidden_states, optimized_patch_embeds], 
                        lr=lr, 
                        maximize=True
                    )
                    total_optimized_tokens = num_thought_tokens + total_patches
                    if verbose >= 1:
                        log.info(f"Initialized {total_patches} patch embeddings as optimizable parameters")
            
            # Build sequence using optimized patches if available, otherwise use selected patches
            patch_idx = 0  # Index into optimized_patch_embeds
            for think_offset in range(num_thought):
                current_thought_pos = thought_idx[0] + think_offset
                # Add current thought token
                embed_parts.append(inputs_embeds_step[:, current_thought_pos:current_thought_pos+1, :])
                # Add matched visual tokens only if this position has them (at stride intervals)
                if think_offset in all_selected_tokens:
                    num_visual = all_selected_tokens[think_offset].size(0)
                    if optimized_patch_embeds is not None and candidate_patches is not None:
                        # Use optimized patch embeddings
                        patch_embeds = candidate_patches[patch_idx:patch_idx+num_visual].unsqueeze(0)
                        embed_parts.append(patch_embeds)
                        patch_idx += num_visual
                    else:
                        # Use original selected patches (first step or not optimizing)
                        embed_parts.append(all_selected_tokens[think_offset].unsqueeze(0))

            # Add suffix after last thought token
            embed_parts.append(inputs_embeds_step[:, thought_idx[1]:, :])

            new_inputs_embeds = torch.cat(embed_parts, dim=1)
            
            # Calculate the positions of all optimized tokens (thought + patches) in the new sequence
            # They form a continuous block in the new sequence: [thought_start, thought_start + total_optimized_tokens]
            optimized_token_positions = []
            current_opt_pos = optimized_start_idx  # Start position in new sequence
            
            for think_offset in range(num_thought):
                # Add thought token position
                optimized_token_positions.append(current_opt_pos)
                current_opt_pos += 1
                # Add patch positions if any
                if think_offset in all_selected_tokens:
                    num_visual = all_selected_tokens[think_offset].size(0)
                    for _ in range(num_visual):
                        optimized_token_positions.append(current_opt_pos)
                        current_opt_pos += 1
            
            # The optimized block is continuous in the new sequence
            optimized_thought_idx = [optimized_start_idx, optimized_start_idx + total_optimized_tokens]

            # update attention mask to match new length
            new_attn_mask = torch.ones((1, new_inputs_embeds.size(1)), device=new_inputs_embeds.device)

            if verbose >= 1:
                log.info(f"Step {step}: Visual injection ENABLED (start_step={visual_injection_start_step}, interval={visual_injection_interval})")
        else:
            # No visual injection for this step - use original embeddings
            new_inputs_embeds = inputs_embeds_step
            new_attn_mask = inputs['attention_mask']

            if verbose >= 1:
                log.debug(f"Step {step}: Visual injection SKIPPED (will start at step {visual_injection_start_step}, interval={visual_injection_interval})")

        # 3.5 compute reward
        # Note: When using inputs_embeds, we should NOT pass pixel_values and image_grid_thw
        inputs_step = dict(
            inputs_embeds=new_inputs_embeds,
            attention_mask=new_attn_mask,
            # pixel_values=inputs.get('pixel_values'),
            # image_grid_thw=inputs.get('image_grid_thw'),
        )
        
        # Use optimized range if patches are being optimized (continuous block)
        if optimized_patch_embeds is not None and should_inject_visual:
            # Build combined candidate embeddings for reward calculation
            combined_candidate = torch.cat([candidate_latent, candidate_patches], dim=0)
            # Use the continuous optimization range in the new sequence
            reward_thought_idx = optimized_thought_idx
            reward_thought_positions = None  # Use continuous range
        else:
            # Use original thought tokens only
            combined_candidate = candidate_latent
            reward_thought_idx = thought_idx
            reward_thought_positions = None

        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(question, candidate_latent)
        else:
            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs_step,
                    thought_idx=reward_thought_idx,
                    thought_hidden_states=combined_candidate,
                    k=top_k,
                    thought_positions=reward_thought_positions,
                )
                reward.backward(retain_graph=True)
            else:
                with torch.no_grad():
                    reward = get_confidence(
                        model=model,
                        inputs=inputs_step,
                        thought_idx=reward_thought_idx,
                        thought_hidden_states=combined_candidate,
                        k=top_k,
                        thought_positions=reward_thought_positions,
                    )

        # 3.6 update latent
        if use_auto_grad:
            optimizer.step()
        else:
            if optimized_patch_embeds is not None:
                # Update both thought tokens and patches
                combined_epsilon = torch.cat([
                    epsilon[:num_thought_tokens],
                    epsilon[num_thought_tokens:]
                ], dim=0)
                grad_ascent = lr * reward * combined_epsilon / sigma**2
                thought_hidden_states += grad_ascent[:num_thought_tokens]
                optimized_patch_embeds.data += grad_ascent[num_thought_tokens:]
            else:
                grad_ascent = lr * reward * epsilon / sigma**2
                thought_hidden_states += grad_ascent

        sigma *= sigma_decay

        reward_value = float(reward.detach().cpu().item())

        # check and update best reward
        is_new_best = reward_value > best_reward
        if is_new_best:
            best_reward, best_reward_step = reward_value, step
            best_thought_hidden_states = thought_hidden_states.clone()
            if optimized_patch_embeds is not None:
                best_patch_embeds = optimized_patch_embeds.clone()
            locked_patch_ids = {k: v.copy() for k, v in current_step_patch_ids.items()}
            if verbose >= 1:
                log.info(
                    f"Step {step}: New best reward, locking current patch selections "
                    f"for {len(locked_patch_ids)} thought tokens."
                )
            if patch_increment > 0 and max_patch_limit is not None:
                prev_budget = current_patch_budget if current_patch_budget is not None else max_patch_limit
                new_budget = min(max_patch_limit, max(1, int(prev_budget)) + patch_increment)
                if new_budget > prev_budget:
                    current_patch_budget = new_budget
                    if verbose >= 1:
                        log.info(
                            f"Step {step}: Increased patch budget from {prev_budget} to "
                            f"{current_patch_budget} due to new best reward."
                        )

        if patches_selected_this_step > 0:
            if prev_patch_count is None or patches_selected_this_step != prev_patch_count:
                if verbose >= 1:
                    log.info(f"Step {step}: visual patches inserted = {patches_selected_this_step}")
            prev_patch_count = patches_selected_this_step

        reward_log_entries.append({
            "step": int(step),
            "reward": reward_value,
            "sigma": float(sigma),
            "is_new_best": int(is_new_best),
            "patch_count": int(patches_selected_this_step),
            "patch_budget": int(current_patch_budget) if current_patch_budget is not None else -1,
        })

        # log reward for each step (blue if no update, green if updated)
        if verbose >= 1:
            if is_new_best:
                color = Fore.GREEN
                update_msg = " [NEW BEST!]"
            else:
                color = Fore.BLUE
                update_msg = ""

            log.debug(f"{color}Step {step}: Reward = {reward_value:.6f}, sigma = {sigma:.6f}, best_reward = {best_reward:.6f}{update_msg}{Style.RESET_ALL}")

        if reward_threshold > 0 and reward_value >= reward_threshold:
            break

        # Clean up memory at the end of each step
        del attentions, hidden_states, new_inputs_embeds, inputs_step
        if 'new_attn_mask' in locals():
            del new_attn_mask
        torch.cuda.empty_cache()

    if reward_csv_path is not None:
        try:
            reward_dir = os.path.dirname(reward_csv_path)
            if reward_dir:
                os.makedirs(reward_dir, exist_ok=True)
            with open(reward_csv_path, "w", newline="") as csv_file:
                fieldnames = ["step", "reward", "sigma", "is_new_best", "patch_count", "patch_budget"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(reward_log_entries)
        except Exception as exc:
            log.error(f"Failed to write reward log to {reward_csv_path}: {exc}")

    # =========================
    # Apply best latent
    # =========================
    inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
    inputs['inputs_embeds'] = inputs_embeds

    # =========================
    # Final generation
    # =========================
    if "qwen3" in model.config.model_type.lower():
        bad_words = []
        bad_words_ids = None
    else:
        bad_words = ["addCriterion"]
        bad_words_ids = processor.tokenizer(bad_words, add_special_tokens=False).input_ids

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        bad_words_ids=bad_words_ids,
        do_sample=False,
        num_beams=1,
    )

    response = processor.decode(outputs[0], skip_special_tokens=True)
    input_length = inputs['inputs_embeds'].shape[1]
    stop_reason = get_stop_reason_vl(outputs, input_length, max_new_tokens, processor.tokenizer)

    return response, best_reward, best_reward_step, stop_reason
