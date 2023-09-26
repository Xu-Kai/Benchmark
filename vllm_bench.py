from vllm import LLM, SamplingParams
import torch 
from torch import distributed as dist 
import time
from tqdm import tqdm
import numpy as np

# # Create an LLM.
llm = LLM(
        #model="/data/scratch/llama-13b-hf", 
        model="/data/scratch/llama-7b-hf", 
        #   model="facebook/opt-125m", 
          tensor_parallel_size=1,
            # max_num_seqs=1,
            # max_num_batched_tokens=2048,
            gpu_memory_utilization=0.95,
            trust_remote_code=True)





def run_to_completion(sampling_params, dummy_prompt_token_ids, profile: bool = False):
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    start_time = time.time()

    llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                    sampling_params=sampling_params,
                    use_tqdm=False)

    end_time = time.time()
    latency = end_time - start_time
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    return latency

config = [[1024, 128]]
for n_config in config:
    run_batchs = [16, 32, 64, 128, 256, 512]
    if n_config[0] == 16:
        run_batchs = [1, 2, 4, 16, 32, 64, 128]
    if n_config[0] == 512:
        run_batchs = [1, 2, 4, 16, 32, 64]


    for batch in run_batchs:
        print("batch", batch, "config is :", n_config)
        input_len = n_config[0]
        out_len =  1
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=out_len,
        )
        dummy_prompt_token_ids = []
        dummy_prompt_token_ids_s = torch.randint(1, 10240, (batch, input_len))
        for t in range(batch):
            a = []
            for i in range(input_len):
                a.append(i)
            dummy_prompt_token_ids.append(a)        # print(dummy_prompt_token_ids)
        # print("Warming up...")
        for i in range(2):
            run_to_completion(sampling_params, dummy_prompt_token_ids, profile=False)

        # Benchmark.
        latencies = []
        for _ in range(5): #tqdm(range(5), desc="Profiling iterations"):
            latencies.append(run_to_completion(sampling_params, dummy_prompt_token_ids, profile=False))

        prefill_avg_latency = np.mean(latencies)
        print(f'prefill latency: {prefill_avg_latency*1000 / out_len} ms')
        # print(f'Avg throughput: {out_len/avg_latency} tokens/seconds')

        input_len = n_config[0]
        out_len =  n_config[1]
        sampling_params = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=out_len,
        )
        dummy_prompt_token_ids = []
        dummy_prompt_token_ids_s = torch.randint(1, 10240, (batch, input_len))
        for t in range(batch):
            a = []
            for i in range(input_len):
                a.append(i)
            dummy_prompt_token_ids.append(a)
        # [[0] * input_len] * batch
        # print(len(dummy_prompt_token_ids[0]))
        # print("Warming up...")
        # for i in range(2):
        #     run_to_completion(sampling_params, dummy_prompt_token_ids, profile=False)

        # Benchmark.
        latencies = []
        for _ in range(5): #tqdm(range(5), desc="Profiling iterations"):
            latencies.append(run_to_completion(sampling_params, dummy_prompt_token_ids, profile=False))

        avg_latency = np.mean(latencies)
        # print(f'Avg latency: {avg_latency*1000 / out_len} ms')
        print(f'Decode throughput: {batch*out_len/(avg_latency - prefill_avg_latency)} tokens/s')
        print(f'total throughput: {batch*out_len/avg_latency} tokens/s')


