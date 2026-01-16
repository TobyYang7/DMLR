import argparse
import numpy as np
import os
import random
import torch
import json
from DMLR import (
    get_vl_dataset,
    extract_answer, 
    args_to_dict, 
    extract_true_answer, 
    judge_answer,
    verify_solution_equivalence,
    generate_vl,
    SYSTEM_PROMPT,
    RewardModel,
    log
)
from transformers import AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm

import copy
import multiprocessing as mp
import time
import tempfile
from typing import Dict, Any

import openai
openai.logging = "info"

huggingface_token = os.environ.get('HUGGING_FACE_TOKEN', None)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"true", "1", "yes", "y", "t"}:
        return True
    if value_str in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the VL model with LTPO")
    parser.add_argument("--dataset", type=str, default="data/mathverse.json", help="Dataset JSON file path or dataset name to evaluate")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the VL model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=100, help="End index of the data to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default="cuda")

    # prompt
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # optimization args
    parser.add_argument('--num_thought_tokens', type=int, default=8)
    parser.add_argument('--sigma', type=float, default=20.0)
    parser.add_argument('--sigma_decay', type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--max_num_steps", type=int, default=20, help="Number of optimization iterations")

    # reward model
    parser.add_argument("--reward_threshold", type=float, default=-1, help="Threshold for reward to stop optimization")
    parser.add_argument("--top_k", type=int, default=10, help="Use top-k most probable tokens to calculate token-level confidence")
    parser.add_argument("--disable_conf_reward", action="store_true", help="If set, disable using confidence reward")
    parser.add_argument("--disable_best_reward", action="store_true", help="If set, disable using best reward step as output")

    # misc
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--use_auto_grad", action="store_true", help="Use PyTorch's auto-grad")
    parser.add_argument("--verbose", type=int, default=1, help="Print detailed information")
    parser.add_argument("--disable_save_logistics", action="store_true", help="Disable saving the logistics.pt")
    parser.add_argument("--use_llm_verify", action="store_true", help="Use LLM to verify solution equivalence")
    parser.add_argument("--verify_only", action="store_true", help="Only re-verify existing results.json and overwrite it")

    # VL specific
    parser.add_argument("--min_pixels", type=int, default=256*28*28, help="Min pixels for VL model")
    parser.add_argument("--max_pixels", type=int, default=1280*28*28, help="Max pixels for VL model")

    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for inference")
    parser.add_argument("--worker_device_round_robin", action="store_true",
                        help="Assign CUDA devices to workers in a round-robin manner")
    
    # Visual residual and visualization
    parser.add_argument("--num_selected_patches", type=int, default=None,
                        help="Max number of image patches (tokens) to keep per thought token")
    parser.add_argument("--visual_token_viz", action="store_true",
                        help="Enable saving visual token heatmaps for each optimization step")
    parser.add_argument("--visual_token_viz_dir", type=str, default="visual_token_viz",
                        help="Directory to save visual token heatmaps")
    parser.add_argument("--visual_only", action="store_true", default=False,
                        help="Use visual features to initialize thought tokens (default: False, use original token embeddings)")
    parser.add_argument("--visual_insert_stride", type=int, default=1,
                        help="Insert visual tokens every N think tokens (default: 1, insert after every think token)")
    parser.add_argument("--visual_injection_start_step", type=int, default=0,
                        help="Start visual injection from this RL step (default: 0, start from beginning)")
    parser.add_argument("--visual_injection_interval", type=int, default=1,
                        help="Perform visual injection every N RL steps (default: 1, every step; 0 or 1 = every step)")
    parser.add_argument("--initial_patch_count", type=int, default=None,
                        help="Initial number of image patches to insert per thought token")
    parser.add_argument("--patch_increment", type=int, default=0,
                        help="How many additional patches to allow when a new best reward appears")
    parser.add_argument("--save_reward_csv", type=str2bool, default=True,
                        help="Whether to save per-step reward traces to CSV (true/false, default: true)")

    return parser.parse_args()


def set_seed(seed):
    '''
    Set random seed for reproducibility

    Args:
        seed: random seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_stop_reason(outputs, input_length, max_new_tokens, tokenizer):
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


def _split_indices(n_total: int, n_workers: int):
    """Split [0, n_total) into n_workers nearly equal spans."""
    n_workers = max(1, int(n_workers))
    base = n_total // n_workers
    rem = n_total % n_workers
    spans = []
    start = 0
    for w in range(n_workers):
        size = base + (1 if w < rem else 0)
        end = start + size
        spans.append((start, end))
        start = end
    return spans


def _maybe_pick_device_for_worker(args, worker_id: int):
    """Choose device for worker: respect --device unless round-robin is requested."""
    dev = (args.device or "").lower()
    if dev.startswith("cuda"):
        n = torch.cuda.device_count()
        if n >= 1 and args.worker_device_round_robin:
            # When CUDA_VISIBLE_DEVICES is set, torch.cuda.device_count() returns the count
            # of visible devices, and we should use cuda:0, cuda:1, etc. which map to
            # the actual visible devices
            device_id = worker_id % n
            return f"cuda:{device_id}"
        # If device is "cuda" without number, use cuda:0 (which maps to first visible device)
        return dev if ":" in dev else ("cuda:0" if n >= 1 else "cpu")
    return dev if dev else ("cuda:0" if torch.cuda.is_available() else "cpu")


def _atomic_write_json(obj: Dict[str, Any], path: str):
    """Atomically write JSON by writing to a temp file then renaming."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), suffix=".tmp", encoding="utf-8") as tf:
        json.dump(obj, tf, indent=2, ensure_ascii=False)
        tmp_name = tf.name
    os.replace(tmp_name, path)  # atomic on POSIX


def _load_model_with_retry(model_name_or_path, model_kwargs, device, max_retries=3, retry_delay=2.0):
    """
    Load model with retry mechanism to handle concurrent access issues.
    
    Args:
        model_name_or_path: Path to the model
        model_kwargs: Keyword arguments for model loading
        device: Device to load model to
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
    
    Returns:
        Loaded model
    """
    for attempt in range(max_retries):
        try:
            # Disable torch.compile to use eager backend
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True  # Disable dynamo compilation
            torch._dynamo.reset()
            
            # Set environment variable to ensure eager mode
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            
            # Explicitly set device before loading to ensure model loads on correct GPU
            if device.type == "cuda":
                torch.cuda.set_device(device)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                **model_kwargs,
                trust_remote_code=True,
                attn_implementation="eager",
            )
            # Ensure model is moved to the specified device
            model.to(device)
            model.eval()
            # Ensure model uses eager backend (disable any compilation)
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            return model
        except (OSError, ValueError, RuntimeError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # exponential backoff
                log.warning(f"Model loading attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                log.error(f"Model loading failed after {max_retries} attempts: {e}")
                raise


def _load_processor_with_retry(model_name_or_path, processor_kwargs, max_retries=3, retry_delay=2.0):
    """
    Load processor with retry mechanism to handle concurrent access issues.
    
    Args:
        model_name_or_path: Path to the model
        processor_kwargs: Keyword arguments for processor loading
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
    
    Returns:
        Loaded processor
    """
    for attempt in range(max_retries):
        try:
            processor = AutoProcessor.from_pretrained(model_name_or_path,
                                                      padding_side='left',
                                                      **processor_kwargs)
            return processor
        except (OSError, ValueError, RuntimeError) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # exponential backoff
                log.warning(f"Processor loading attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                log.error(f"Processor loading failed after {max_retries} attempts: {e}")
                raise


def _worker_run(wargs, worker_id: int, result_queue: mp.Queue):
    """
    A single worker:
      - loads model/processor/reward_model
      - loads dataset subset [wargs.start_data_idx, wargs.end_data_idx)
      - iterates examples
      - for each example sends a message dict via result_queue:
          {"type":"result", "data": {entry fields ...}, "count_as_total": True/False}
      - at the end, sends {"type":"done", "worker_id": worker_id}
    """
    try:
        # logger level in worker
        import logging
        log_level = logging.DEBUG if wargs.verbose > 0 else logging.INFO
        if log.handlers:
            for handler in log.handlers:
                handler.setLevel(log_level)

        # if wargs.seed:
        #     set_seed(int(wargs.seed) + worker_id)  # decorrelate a bit

        # device - ensure we use the device assigned by parent process
        device_str = wargs.device if wargs.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        # Explicitly set the current device to ensure all operations use the correct GPU
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # verify-only: we keep single-process in main. Workers shouldn't be spawned for verify-only.
        if wargs.verify_only:
            result_queue.put({"type": "done", "worker_id": worker_id})
            return

        # model & processor
        model_kwargs = {"torch_dtype": torch.float32}
        processor_kwargs = {
            "min_pixels": wargs.min_pixels * 28 * 28,
            "max_pixels": wargs.max_pixels * 28 * 28,
        }
        if huggingface_token:
            model_kwargs["token"] = huggingface_token
            processor_kwargs["token"] = huggingface_token

        # Load model with retry mechanism
        model = _load_model_with_retry(
            wargs.model_name_or_path,
            model_kwargs,
            device,
            max_retries=5,  # More retries for worker processes
            retry_delay=3.0  # Longer delay for workers
        )

        # Load processor with retry mechanism
        processor = _load_processor_with_retry(
            wargs.model_name_or_path,
            processor_kwargs,
            max_retries=5,
            retry_delay=3.0
        )

        reward_model = RewardModel(
            model=model,
            tokenizer=processor.tokenizer,
            num_thought_tokens=wargs.num_thought_tokens,
            device=device_str,
        )

        # dataset subset for this worker
        start_idx = max(0, wargs.start_data_idx)
        end_idx = wargs.end_data_idx
        dataset = get_vl_dataset(
            wargs.dataset,
            processor=processor,
            prompt_idx=wargs.solver_prompt_idx,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        reward_log_dir = None
        if wargs.save_reward_csv:
            if wargs.output_dir:
                reward_log_dir = os.path.join(wargs.output_dir, "reward_logs")
                os.makedirs(reward_log_dir, exist_ok=True)
            else:
                log.warning("save_reward_csv=True but no output_dir provided; skipping reward CSV logging.")

        # Iterate local indices (0..len(subset)-1) but compute global index = start_idx + i
        for local_i in range(len(dataset)):
            global_i = start_idx + local_i

            example = dataset[local_i]
            question = example['question']
            image = example.get('image', None)
            image_path = example.get('image_path', '')

            true_answer = extract_true_answer(example["answer"], name=wargs.dataset)

            # Always report progress to parent (even if skipped)
            if true_answer is None:
                result_queue.put({
                    "type": "result",
                    "count_as_total": True,  # still count in tqdm total
                    "data": {
                        "data_idx": int(global_i),
                        "question": question,
                        "image_path": image_path,
                        "sys_prompt": "",
                        "prompt": "",
                        "model_output": "",
                        "answer": None,
                        "ground_truth": None,
                        "ground_truth_text": None,
                        "is_correct": False,
                        "best_reward": None,
                        "best_reward_step": None,
                        "stop_reason": "skipped_no_gt",
                    }
                })
                continue

            # Resolve ground-truth text if dataset provided only a label
            ground_truth_text = None
            gt_letter = None
            if isinstance(true_answer, str):
                for ch in true_answer:
                    if ch.isalpha():
                        gt_letter = ch.upper()
                        break
            if gt_letter is not None:
                ground_truth_text = example.get('answer_text', None)
                if ground_truth_text is None:
                    choices_map = example.get('choices', None)
                    if isinstance(choices_map, dict) and gt_letter in choices_map:
                        ground_truth_text = choices_map[gt_letter]

            # ===== generate =====
            try:
                # Ensure messages is a list (not a dict) for apply_chat_template
                messages_for_template = example["messages"]
                if isinstance(messages_for_template, dict):
                    messages_for_template = [messages_for_template]
                elif not isinstance(messages_for_template, list):
                    messages_for_template = [messages_for_template] if messages_for_template else []
                
                final_prompt = processor.apply_chat_template(
                    messages_for_template,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                sys_prompt = ""
                reward_csv_path = None
                if reward_log_dir is not None:
                    reward_csv_path = os.path.join(
                        reward_log_dir,
                        f"reward_steps_{int(global_i):06d}.csv"
                    )
                output, best_reward, best_reward_step, stop_reason = generate_vl(
                    processor=processor,
                    model=model,
                    reward_model=reward_model,
                    question=question,
                    image=image,
                    messages=example["messages"],
                    num_thought_tokens=wargs.num_thought_tokens,
                    max_rl_steps=wargs.max_num_steps,
                    reward_threshold=wargs.reward_threshold,
                    lr=wargs.lr,
                    sigma=wargs.sigma,
                    sigma_decay=wargs.sigma_decay,
                    use_auto_grad=wargs.use_auto_grad,
                    disable_conf_reward=wargs.disable_conf_reward,
                    disable_best_reward=wargs.disable_best_reward,
                    data_name=wargs.dataset,
                    model_name=wargs.model_name_or_path,
                    verbose=wargs.verbose,
                    top_k=wargs.top_k,
                    max_new_tokens=wargs.max_new_tokens,
                    device=device,
                    num_selected_patches=wargs.num_selected_patches,
                    visual_token_viz=wargs.visual_token_viz,
                    visual_only=wargs.visual_only,
                    visual_token_viz_dir=wargs.visual_token_viz_dir,
                    visual_insert_stride=wargs.visual_insert_stride,
                    visual_injection_start_step=wargs.visual_injection_start_step,
                    visual_injection_interval=wargs.visual_injection_interval,
                    data_idx=int(global_i),
                    initial_patch_count=wargs.initial_patch_count,
                    patch_increment=wargs.patch_increment,
                    reward_csv_path=reward_csv_path,
                )
            except Exception as gen_e:
                # On generation error, still report this sample as progressed
                result_queue.put({
                    "type": "result",
                    "count_as_total": True,
                    "data": {
                        "data_idx": int(global_i),
                        "question": question,
                        "image_path": image_path,
                        "sys_prompt": sys_prompt if 'sys_prompt' in locals() else "",
                        "prompt": final_prompt if 'final_prompt' in locals() else "",
                        "model_output": f"[GENERATION_ERROR] {repr(gen_e)}",
                        "answer": None,
                        "ground_truth": true_answer,
                        "ground_truth_text": ground_truth_text,
                        "is_correct": False,
                        "best_reward": None,
                        "best_reward_step": None,
                        "stop_reason": "exception",
                    }
                })
                continue

            # extract answer
            answer = extract_answer(
                output,
                # data_name=wargs.dataset,
                # prompt_idx=wargs.solver_prompt_idx,
                # model_name=wargs.model_name_or_path,
            )

            # judge
            is_correct = False
            if answer is not None:
                if wargs.use_llm_verify:
                    try:
                        if gt_letter is not None and ground_truth_text:
                            gt_for_verify = f"{gt_letter}:{ground_truth_text}"
                        elif ground_truth_text:
                            gt_for_verify = f"{true_answer}:{ground_truth_text}"
                        else:
                            gt_for_verify = true_answer
                        is_correct = verify_solution_equivalence(answer, gt_for_verify)
                    except Exception as e:
                        is_correct = judge_answer(output, true_answer, data_name=wargs.dataset, prompt_idx=wargs.solver_prompt_idx)
                else:
                    is_correct = judge_answer(output, true_answer, data_name=wargs.dataset, prompt_idx=wargs.solver_prompt_idx)

            # send result
            result_queue.put({
                "type": "result",
                "count_as_total": True,
                "data": {
                    "data_idx": int(global_i),
                    "question": question,
                    "image_path": image_path,
                    "sys_prompt": sys_prompt,
                    "prompt": final_prompt,
                    "model_output": output,
                    "answer": answer,
                    "ground_truth": true_answer,
                    "ground_truth_text": ground_truth_text,
                    "is_correct": bool(is_correct),
                    "best_reward": float(best_reward) if isinstance(best_reward, (int, float)) else None,
                    "best_reward_step": int(best_reward_step) if isinstance(best_reward_step, (int, float)) else None,
                    "stop_reason": stop_reason,
                }
            })

    except Exception as e:
        # Report worker-level fatal error as a special message
        result_queue.put({"type": "worker_error", "worker_id": worker_id, "error": repr(e)})
    finally:
        # signal worker done
        result_queue.put({"type": "done", "worker_id": worker_id})


def main(args):
    '''
    Evaluate VL model with LTPO (single-process path)

    Args:
        args: command line arguments
    '''
    import logging
    log_level = logging.DEBUG if args.verbose > 0 else logging.INFO
    if log.handlers:
        for handler in log.handlers:
            handler.setLevel(log_level)

    if args.seed:
        set_seed(args.seed)

    # set device
    if args.device is None:
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device
    device = torch.device(device_str)
    # Explicitly set the current device to ensure all operations use the correct GPU
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # verify-only: re-open existing results.json, re-judge correctness, overwrite, and exit
    if args.verify_only:
        output_dir = args.output_dir

        results_path = f"{output_dir}/results.json"
        if not os.path.exists(results_path):
            log.error(f"results.json not found for verify-only: {results_path}")
            return

        processor_kwargs = {
            "min_pixels": args.min_pixels * 28 * 28,
            "max_pixels": args.max_pixels * 28 * 28,
        }
        if huggingface_token:
            processor_kwargs["token"] = huggingface_token
        processor = _load_processor_with_retry(
            args.model_name_or_path,
            processor_kwargs,
            max_retries=3,
            retry_delay=2.0
        )

        dataset = get_vl_dataset(
            args.dataset,
            processor=processor,
            prompt_idx=args.solver_prompt_idx,
            start_idx=max(0, args.start_data_idx),
            end_idx=args.end_data_idx,
        )

        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        entries = results.get("entries", [])
        correct = 0
        total = 0

        for entry in entries:
            i = entry.get("data_idx")
            if i is None or i < 0 or i >= len(dataset):
                continue
            example = dataset[i]
            true_answer = extract_true_answer(example["answer"], name=args.dataset)
            ground_truth_text = None
            gt_letter = None
            if isinstance(true_answer, str):
                for ch in true_answer:
                    if ch.isalpha():
                        gt_letter = ch.upper()
                        break
            if gt_letter is not None:
                ground_truth_text = example.get('answer_text', None)
                if ground_truth_text is None:
                    choices_map = example.get('choices', None)
                    if isinstance(choices_map, dict) and gt_letter in choices_map:
                        ground_truth_text = choices_map[gt_letter]
            entry["ground_truth"] = true_answer
            if ground_truth_text is not None:
                entry["ground_truth_text"] = ground_truth_text
            answer = entry.get("answer")
            if answer is None:
                answer = extract_answer(
                    entry.get("model_output", entry.get("response", "")),
                    # data_name=args.dataset,
                    # prompt_idx=args.solver_prompt_idx,
                    # model_name=args.model_name_or_path,
                )

            is_correct = False
            if answer is not None and true_answer is not None:
                if args.use_llm_verify:
                    try:
                        gt_for_verify = true_answer
                        if entry.get("ground_truth_text"):
                            gt_letter = None
                            if isinstance(true_answer, str):
                                for ch in true_answer:
                                    if ch.isalpha():
                                        gt_letter = ch.upper()
                                        break
                            if gt_letter:
                                gt_for_verify = f"{gt_letter}:{entry.get('ground_truth_text')}"
                            else:
                                gt_for_verify = f"{true_answer}:{entry.get('ground_truth_text')}"
                        is_correct = verify_solution_equivalence(answer, gt_for_verify)
                    except Exception as e:
                        log.error(f"LLM verify failed for idx {i}, fallback to rule-based: {e}")
                        is_correct = judge_answer(entry.get("model_output", entry.get("response", "")), true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx)
                else:
                    is_correct = judge_answer(entry.get("model_output", entry.get("response", "")), true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx)

            entry["is_correct"] = bool(is_correct)
            total += 1
            correct += int(is_correct)

        results["correct"] = correct
        results["total"] = total
        results["accuracy"] = (correct / total) if total > 0 else 0.0

        os.makedirs(output_dir, exist_ok=True)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info(f"Re-verified {total} entries. Accuracy now = {results['accuracy']:.4f}. Overwrote: {results_path}")
        return

    # load VL model and processor
    model_kwargs = {"torch_dtype": torch.float32}
    processor_kwargs = {
        "min_pixels": args.min_pixels * 28 * 28,
        "max_pixels": args.max_pixels * 28 * 28,
    }
    if huggingface_token:
        model_kwargs["token"] = huggingface_token
        processor_kwargs["token"] = huggingface_token

    # Load model with retry mechanism
    model = _load_model_with_retry(
        args.model_name_or_path,
        model_kwargs,
        device,
        max_retries=3,
        retry_delay=2.0
    )

    # Load processor with retry mechanism
    processor = _load_processor_with_retry(
        args.model_name_or_path,
        processor_kwargs,
        max_retries=3,
        retry_delay=2.0
    )

    reward_model = RewardModel(
        model=model,
        tokenizer=processor.tokenizer,
        num_thought_tokens=args.num_thought_tokens,
        device=device_str,
    )

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = args.end_data_idx
    dataset = get_vl_dataset(
        args.dataset,
        processor=processor,
        prompt_idx=args.solver_prompt_idx,
        start_idx=start_data_idx,
        end_idx=end_data_idx,
    )

    total = 0
    correct = 0
    entries = []

    output_dir = args.output_dir

    end_data_idx = len(dataset)
    start_data_idx = 0

    if args.resume and not args.disable_save_logistics:
        log.info(f"Resume from {output_dir}")
        logistics = torch.load(f"{output_dir}/logistics.pt")
        start_data_idx = logistics["start_idx"]
        correct = logistics["correct"]
        total = logistics["total"]
        entries = logistics["entries"]

    log.info(f"Start to evaluate {args.dataset} from {start_data_idx} to {end_data_idx}...")
    json_path = f"{output_dir}/results.json"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set visual_token_viz_dir if enabled but not specified
    if args.visual_token_viz and not args.visual_token_viz_dir:
        args.visual_token_viz_dir = os.path.join(output_dir, "visual_token_viz")
        os.makedirs(args.visual_token_viz_dir, exist_ok=True)

    def save_results_json():
        results = {
            "model": args.model_name_or_path,
            "dataset": args.dataset,
            "accuracy": correct / total if total > 0 else 0,
            "correct": correct,
            "total": total,
            "config": {
                "max_new_tokens": args.max_new_tokens,
                "solver_prompt_idx": args.solver_prompt_idx,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            },
            "args": args_to_dict(args),
            "entries": [
                {
                    "data_idx": entry["data_idx"],
                    "question": entry["question"],
                    "image_path": entry.get("image_path", ""),
                    "sys_prompt": entry.get("sys_prompt", ""),
                    "prompt": entry.get("prompt", ""),
                    "model_output": entry.get("model_output", ""),
                    "answer": entry.get("answer", ""),
                    "ground_truth": entry.get("ground_truth"),
                    "ground_truth_text": entry.get("ground_truth_text"),
                    "is_correct": entry.get("is_correct", False),
                    "best_reward": float(entry["best_reward"]) if isinstance(entry.get("best_reward"), (int, float)) else None,
                    "best_reward_step": int(entry["best_reward_step"]) if isinstance(entry.get("best_reward_step"), (int, float)) else None,
                    "stop_reason": entry.get("stop_reason", "unknown"),
                }
                for entry in entries
            ]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    reward_log_dir = None
    if args.save_reward_csv:
        if args.output_dir:
            reward_log_dir = os.path.join(args.output_dir, "reward_logs")
            os.makedirs(reward_log_dir, exist_ok=True)
        else:
            log.warning("save_reward_csv=True but no output_dir provided; skipping reward CSV logging.")

    data_idx_list = range(start_data_idx, end_data_idx)
    for i in tqdm(data_idx_list):
        example = dataset[i]
        question = example['question']
        image = example.get('image', None)
        image_path = example.get('image_path', '')

        true_answer = extract_true_answer(example["answer"], name=args.dataset)
        ground_truth_text = None
        gt_letter = None
        if isinstance(true_answer, str):
            for ch in true_answer:
                if ch.isalpha():
                    gt_letter = ch.upper()
                    break
        if gt_letter is not None:
            ground_truth_text = example.get('answer_text', None)
            if ground_truth_text is None:
                choices_map = example.get('choices', None)
                if isinstance(choices_map, dict) and gt_letter in choices_map:
                    ground_truth_text = choices_map[gt_letter]

        if args.verbose:
            log.debug(f"Index {i}, Question: {question}")
            log.debug(f"Index {i}, Has image: {image is not None}")
            log.debug(f"Index {i}, True answer: {true_answer}")
        if true_answer is None:
            total += 1  # keep progress parity in single-process mode
            save_results_json()
            continue

        # Ensure messages is a list (not a dict) for apply_chat_template
        messages_for_template = example["messages"]
        if isinstance(messages_for_template, dict):
            messages_for_template = [messages_for_template]
        elif not isinstance(messages_for_template, list):
            messages_for_template = [messages_for_template] if messages_for_template else []
        
        final_prompt = processor.apply_chat_template(
            messages_for_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        sys_prompt = ""
        reward_csv_path = None
        if reward_log_dir is not None:
            reward_csv_path = os.path.join(
                reward_log_dir,
                f"reward_steps_{int(i):06d}.csv"
            )
        output, best_reward, best_reward_step, stop_reason = generate_vl(
            processor=processor,
            model=model,
            reward_model=reward_model,
            question=question,
            image=image,
            messages=example["messages"],
            num_thought_tokens=args.num_thought_tokens,
            max_rl_steps=args.max_num_steps,
            reward_threshold=args.reward_threshold,
            lr=args.lr,
            sigma=args.sigma,
            sigma_decay=args.sigma_decay,
            use_auto_grad=args.use_auto_grad,
            disable_conf_reward=args.disable_conf_reward,
            disable_best_reward=args.disable_best_reward,
            data_name=args.dataset,
            model_name=args.model_name_or_path,
            verbose=args.verbose,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            device=device,
            num_selected_patches=args.num_selected_patches,
            visual_token_viz=args.visual_token_viz,
            visual_only=args.visual_only,
            visual_token_viz_dir=args.visual_token_viz_dir,
            visual_insert_stride=args.visual_insert_stride,
            visual_injection_start_step=args.visual_injection_start_step,
            visual_injection_interval=args.visual_injection_interval,
            data_idx=i,
            initial_patch_count=args.initial_patch_count,
            patch_increment=args.patch_increment,
            reward_csv_path=reward_csv_path,
        )

        answer = extract_answer(
            output,
            # data_name=args.dataset,
            # prompt_idx=args.solver_prompt_idx,
            # model_name=args.model_name_or_path,
        )
        if args.verbose:
            if args.verbose > 1:
                log.debug(f"Index {i}, LLM response:\n{output}")
            log.debug(f"Index {i}, LLM answer: {answer}")
            log.debug(f"Index {i}, True answer: {true_answer}")
            log.debug(f"Index {i}, Best reward: {best_reward}, Best reward step: {best_reward_step}")

        is_correct = False
        if answer is not None:
            if args.use_llm_verify:
                try:
                    if gt_letter is not None and ground_truth_text:
                        gt_for_verify = f"{gt_letter}:{ground_truth_text}"
                    elif ground_truth_text:
                        gt_for_verify = f"{true_answer}:{ground_truth_text}"
                    else:
                        gt_for_verify = true_answer
                    is_correct = verify_solution_equivalence(answer, gt_for_verify)
                except Exception as e:
                    log.error(f"LLM verify failed, falling back to rule-based judge: {e}")
                    is_correct = judge_answer(output, true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx)
            else:
                is_correct = judge_answer(output, true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx)
            correct += is_correct

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                image_path=image_path,
                sys_prompt=sys_prompt,
                prompt=final_prompt,
                model_output=output,
                answer=answer,
                ground_truth=true_answer,
                ground_truth_text=ground_truth_text,
                is_correct=is_correct,
                best_reward=best_reward,
                best_reward_step=best_reward_step,
                stop_reason=stop_reason,
            ))

        total += 1
        if not args.disable_save_logistics:
            save_results_json()

        if not args.disable_save_logistics:
            torch.save({
                "start_idx": i+1,
                "total": total,
                "correct": correct,
                "entries": entries,
            }, f"{output_dir}/logistics.pt")
        log.info(f"Current state: correct={correct}, total={total}, accuracy={correct / total:.4f}")

    if args.verbose:
        for i, entry in enumerate(entries):
            if not entry['is_correct']:
                continue
            log.info(f"====================== Entry {i} ======================")
            log.info(f">>> Question:\n{entry['question']}")
            log.info(f">>> Response:\n{entry.get('model_output', entry.get('response', ''))}")
            log.info(f">>> Answer:\n{entry['answer']}")
            log.info(f">>> Data Idx: {entry['data_idx']}")
            log.info(f">>> Best Reward: {entry['best_reward']}, Best Reward Step: {entry['best_reward_step']}")

    log.info(f">>> Final State: correct={correct}, total={total}, accuracy={correct / total:.4f}")
    log.info(f">>> Data Idx with Correct Answer: {[entry['data_idx'] for entry in entries if entry['is_correct']]}")

    if not args.disable_save_logistics:
        save_results_json()
        log.info(f">>> Final results saved to: {json_path}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass  # already set

    args = parse_args()
    for arg in vars(args):
        log.info(f"-- {arg}: {getattr(args, arg)}")

    if args.verify_only or args.num_workers <= 1:
        # single-process path (or verify-only)
        main(args)
    else:
        # Parent prepares dataset length and pre-warms cache
        log.info("[MP] Pre-warming HuggingFace cache to reduce worker contention...")
        processor = None
        try:
            processor_kwargs = {
                "min_pixels": args.min_pixels * 28 * 28,
                "max_pixels": args.max_pixels * 28 * 28,
            }
            if huggingface_token:
                processor_kwargs["token"] = huggingface_token
            # Pre-load processor to warm cache
            processor = _load_processor_with_retry(
                args.model_name_or_path,
                processor_kwargs,
                max_retries=3,
                retry_delay=2.0
            )
            log.info("[MP] Processor cache pre-warmed successfully")
        except Exception as e:
            log.warning(f"[MP] Processor pre-warm failed (workers will retry): {e}")
            processor = None  # tolerate init failure; dataset may still load

        dataset_for_len = get_vl_dataset(
            args.dataset,
            processor=processor,
            prompt_idx=args.solver_prompt_idx,
            start_idx=max(0, args.start_data_idx),
            end_idx=args.end_data_idx,
        )
        n_total = len(dataset_for_len)  # tqdm total counts every example (even if skipped)

        # Use output dir directly from args
        parent_output_dir = args.output_dir
        os.makedirs(parent_output_dir, exist_ok=True)
        json_path = os.path.join(parent_output_dir, "results.json")

        # Set visual_token_viz_dir if enabled but not specified
        if args.visual_token_viz and not args.visual_token_viz_dir:
            args.visual_token_viz_dir = os.path.join(parent_output_dir, "visual_token_viz")
            os.makedirs(args.visual_token_viz_dir, exist_ok=True)
        
        # Initial results skeleton
        results: Dict[str, Any] = {
            "model": args.model_name_or_path,
            "dataset": args.dataset,
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "config": {
                "max_new_tokens": args.max_new_tokens,
                "solver_prompt_idx": args.solver_prompt_idx,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            },
            "args": args_to_dict(args),
            "entries": [],
        }

        # If resume and file exists, load it (best-effort)
        if args.resume and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    old = json.load(f)
                # merge minimal fields
                results["entries"] = old.get("entries", [])
                results["correct"] = int(old.get("correct", 0))
                results["total"] = int(old.get("total", 0))
                results["accuracy"] = float(old.get("accuracy", 0.0))
                # Update args with current args (new args take precedence)
                results["args"] = args_to_dict(args)
                # build a set of processed indices to skip in tqdm total if desired
                processed_idx = {e["data_idx"] for e in results["entries"] if "data_idx" in e}
                # keep total as n_total; we still show full progress but already-processed will be fast to skip
            except Exception:
                processed_idx = set()
        else:
            processed_idx = set()

        # Build spans
        spans = _split_indices(n_total, args.num_workers)
        base_global = max(0, args.start_data_idx)

        result_queue: mp.Queue = mp.Queue(maxsize=max(1000, args.num_workers * 4))
        processes = []
        for wid, (s, e) in enumerate(spans):
            wargs = copy.deepcopy(args)
            wargs.start_data_idx = base_global + s
            wargs.end_data_idx = base_global + e
            wargs.device = _maybe_pick_device_for_worker(args, wid)
            log.info(f"[MP] spawn worker {wid} | idx [{wargs.start_data_idx}, {wargs.end_data_idx}) | device={wargs.device}")
            p = mp.Process(target=_worker_run, args=(wargs, wid, result_queue))
            p.daemon = False
            p.start()
            processes.append(p)

        # Parent loop: single tqdm + single results.json
        done_workers = 0
        with tqdm(total=n_total, desc="Progress", dynamic_ncols=True) as pbar:
            while done_workers < args.num_workers:
                msg = result_queue.get()
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "result":
                    data = msg.get("data", {})
                    count_as_total = bool(msg.get("count_as_total", True))
                    # Update aggregate stats
                    if count_as_total:
                        results["total"] += 1
                        if data.get("is_correct"):
                            results["correct"] += 1
                        # tqdm advance
                        pbar.update(1)
                    results["accuracy"] = (results["correct"] / results["total"]) if results["total"] > 0 else 0.0
                    # Append entry (avoid duplicates by data_idx)
                    didx = data.get("data_idx")
                    if didx is not None and didx in processed_idx:
                        # already exists; skip to avoid duplication
                        pass
                    else:
                        results["entries"].append(data)
                        if didx is not None:
                            processed_idx.add(didx)
                    # Write JSON atomically
                    _atomic_write_json(results, json_path)
                elif mtype == "worker_error":
                    wid = msg.get("worker_id")
                    err = msg.get("error")
                    log.error(f"[MP] Worker {wid} error: {err}")
                elif mtype == "done":
                    done_workers += 1
                else:
                    # unknown message; ignore
                    pass

        # Ensure final write
        _atomic_write_json(results, json_path)
        log.info(f"[MP] merged accuracy={results['accuracy']:.4f}, correct={results['correct']}, total={results['total']}")
        log.info(f"[MP] final results saved to: {json_path}")
