import argparse
import os

import torch
from datasets import load_dataset
from transformers import LlamaTokenizer

import torch
from transformers import LlamaTokenizer

import colossalai
from colossalai.inference.tensor_parallel.engine import TPInferEngine

# from colossalai.inference.tensor_parallel.modeling._utils import init_to_get_rotary
from colossalai.logging import disable_existing_loggers
from colossalai.shardformer import ShardConfig

from colossalai.shardformer import ShardConfig
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

from colossalai.inference.quant.smoothquant.models.llama import SmoothLlamaForCausalLM
from transformers import AutoModel, AutoModelForCausalLM
import time


def print_perf_stats(latency_set, config, max_mem_allocated, batch_size, input_len, out_len, warmup=3):
    # trim warmup queries
    latency_set = list(latency_set)
    latency_set = latency_set[warmup:]
    count = len(latency_set)

    if count > 0:
        latency_set.sort()
        avg = sum(latency_set) / count
        num_layers = getattr(config, "num_layers", config.num_hidden_layers)
        num_parameters = num_layers * config.hidden_size * config.hidden_size * 12
        num_bytes = 2

        print("Batch Size: {}".format(batch_size))
        print("Input Len: {}".format(input_len))
        print("Output Len: {}".format(out_len))
        print("Max CUDA memory allocated: {0:8.4f} GB".format(max_mem_allocated / 1024 / 1024 / 1024))
        print("Avg Per Token Latency: {0:8.2f} ms".format(avg * 1000))
        print("Avg BW: {0:8.2f} GB/s".format(1 / avg * num_parameters * num_bytes / 1e9))
        print("Avg flops: {0:8.2f} TFlops/s".format(1 / avg * num_parameters * num_bytes * batch_size / 1e12))
        print("Avg Throughput: tokens/s: {}".format((1000 / (avg * 1000)) * batch_size))


def run_benchmark(model, iters=10, batch_size=1, input_len=16, **generate_kwargs):
    input_tokens = {
        "input_ids": torch.randint(1, 1000, (batch_size, input_len), device="cuda"),
        "attention_mask": torch.ones((batch_size, input_len), device="cuda"),
    }
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    times = []

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = model.generate(**input_tokens, **generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        out_len = outputs.shape[1]
        print(f" iter {i}: out len {outputs.shape}, generation time {str(end - start)} s")
        times.append((end - start) / (out_len - input_len))
    max_mem_allocated = torch.cuda.max_memory_allocated()

    print_perf_stats(times, model.config, max_mem_allocated, batch_size, input_len, out_len)


def run_tp_benchmark(engine, iters=10, batch_size=1, input_len=16, **generate_kwargs):
    input_tokens = {
        "input_ids": torch.randint(1, 1000, (batch_size, input_len), device="cuda"),
        "attention_mask": torch.ones((batch_size, input_len), device="cuda"),
    }
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    times = []

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = engine.generate(input_tokens, **generate_kwargs)
        torch.cuda.synchronize()
        end = time.time()
        out_len = outputs.shape[1]
        print(f" iter {i}: out len {outputs.shape}, generation time {str(end - start)} s")
        times.append((end - start) / (out_len - input_len))
    max_mem_allocated = torch.cuda.max_memory_allocated()

    print_perf_stats(times, engine.model.config, max_mem_allocated, batch_size, input_len, out_len)


def run_vllm_benchmark(model, iters=10, batch_size=1, input_len=16, max_out_len=16):
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        use_beam_search=False,
        ignore_eos=True,
        max_tokens=max_out_len,
    )

    dummy_prompt_token_ids = []
    dummy_prompt_token_ids_s = torch.randint(1, 10240, (batch_size, input_len))
    for t in range(batch_size):
        a = []
        for i in range(input_len):
            a.append(i)
        dummy_prompt_token_ids.append(a)

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    times = []

    for i in range(iters):
        torch.cuda.synchronize()
        start = time.time()
        outputs = model.generate(
            prompt_token_ids=dummy_prompt_token_ids, sampling_params=sampling_params, use_tqdm=False
        )
        torch.cuda.synchronize()
        end = time.time()
        out_len = outputs.shape[1]
        print(f" iter {i}: out len {outputs.shape}, generation time {str(end - start)} s")
        times.append((end - start) / (out_len - input_len))
    max_mem_allocated = torch.cuda.max_memory_allocated()

    print_perf_stats(times, model.config, max_mem_allocated, batch_size, input_len, out_len)


def build_auto_gptq_model_and_tokenizer(model_name, gptq_quantized_model_dir):
    from auto_gptq import AutoGPTQForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(
        gptq_quantized_model_dir, device=torch.cuda.current_device(), inject_fused_attention=False
    )

    return model, tokenizer


def build_hg_model_and_tokenizer(model_name):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load quantized model to the first GPU
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def build_cai_gptq_model_and_tokenizer(model_name, quantized_model_dir, max_batch_size, max_input_len, max_output_len):
    from auto_gptq import AutoGPTQForCausalLM

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(
        quantized_model_dir, device=torch.cuda.current_device(), inject_fused_attention=False
    )

    model_config = model.config
    shard_config = ShardConfig(enable_tensor_parallelism=False, inference_only=True, inference_gptq=True)
    infer_engine = TPInferEngine(model.model, shard_config, max_batch_size, max_input_len, max_output_len)
    infer_engine.model = infer_engine.model.cuda()

    return infer_engine, tokenizer


def build_cai_model_and_tokenizer(model_name, max_batch_size, max_input_len, max_output_len):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # load quantized model to the first GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
    )

    shard_config = ShardConfig(enable_tensor_parallelism=False, inference_only=True, inference_gptq=True)
    infer_engine = TPInferEngine(model, shard_config, max_batch_size, max_input_len, max_output_len)
    infer_engine.model = infer_engine.model.cuda()

    return infer_engine, tokenizer


def build_vllm_model_and_tokenizer(model_name):
    from vllm import LLM, SamplingParams

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # # Create an LLM.
    llm = LLM(
        # model="/data/scratch/llama-13b-hf",
        model=model_name,
        #   model="facebook/opt-125m",
        tensor_parallel_size=1,
        # max_num_seqs=1,
        # max_num_batched_tokens=2048,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
    )

    return llm, tokenizer


def build_cai_smooth_model_and_tokenizer(model_name, quantized_model_dir):
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = SmoothLlamaForCausalLM.from_quantized(quantized_model_dir, model_basename="llama-7b")

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-type", type=str, choices=["vllm", "auto-gptq", "cai-gptq", "smoothquant", "huggingface", "cai"]
    )
    parser.add_argument("--model-path", type=str, help="model name")
    parser.add_argument(
        "--quantized-path",
        type=str,
        help="location of the gptq model",
    )

    args = parser.parse_args()
    return args


@torch.no_grad()
def bench_llama(args):
    model_path = args.model_path
    quantized_path = args.quantized_path
    if args.bench_type == "auto-gptq":
        model, tokenizer = build_auto_gptq_model_and_tokenizer(model_path, quantized_path)
    elif args.bench_type == "smoothquant":
        model = SmoothLlamaForCausalLM.from_quantized(quantized_path, model_basename="llama-7b")
    elif args.bench_type == "huggingface":
        model, tokenizer = build_hg_model_and_tokenizer(model_path)
    elif args.bench_type == "vllm":
        model, tokenizer = build_vllm_model_and_tokenizer(model_path)
    elif args.bench_type == "cai-gptq":
        model, tokenizer = build_cai_gptq_model_and_tokenizer(model_path, quantized_path, 64, 1024, 128)
    elif args.bench_type == "cai":
        model, tokenizer = build_cai_model_and_tokenizer(model_path, 64, 1024, 128)

    if args.bench_type in ["auto-gptq", "smoothquant", "huggingface"]:
        model = model.cuda()
        generate_kwargs = dict(max_new_tokens=16, do_sample=False, use_cache=True)
        input_tokens = tokenizer(["today is "], return_tensors="pt").to("cuda")
        out = model.generate(input_tokens, **generate_kwargs)
        text = tokenizer.batch_decode(out)
        print("out is:", text)
        generate_kwargs = dict(max_new_tokens=128)
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            run_benchmark(model, batch_size=batch_size, input_len=1024, **generate_kwargs)

    if args.bench_type in ["vllm"]:
        run_vllm_benchmark(model, iters=10, batch_size=1, input_len=1024, max_out_len=128)

        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            run_vllm_benchmark(model, iters=10, batch_size=batch_size, input_len=1024, max_out_len=128)

    if args.bench_type in ["cai-gptq", "cai"]:
        generate_kwargs = dict(max_new_tokens=16, do_sample=False, use_cache=True)
        input_tokens = tokenizer(["auto-gptq is "], return_tensors="pt").to("cuda")
        out = model.generate(input_tokens, **generate_kwargs)
        text = tokenizer.batch_decode(out)
        print("out is:", text)
        generate_kwargs = dict(max_new_tokens=128)
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            run_tp_benchmark(model, batch_size=batch_size, input_len=1024, **generate_kwargs)


def check_llama(rank, world_size, port, args):
    disable_existing_loggers()
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="127.0.0.1", port=port, backend="nccl")
    bench_llama(args)


@rerun_if_address_is_in_use()
@clear_cache_before_run()
def test_llama(args):
    spawn(check_llama, 1, args=args)


if __name__ == "__main__":
    args = parse_args()
    if args.bench_type in ["cai", "cai-gptq"]:
        test_llama(args)
    else:
        bench_llama(args)
