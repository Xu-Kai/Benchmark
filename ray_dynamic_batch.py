import requests

# CUDA_VISIBLE_DEVICES=2,3 RAY_DEDUP_LOGS=0 serve run -p 8299 app:dynamic_batch_app path=config.yaml
#   -h, --host TEXT      Host for HTTP server to listen on. Defaults to 127.0.0.1.
#   -p, --port INTEGER   Port for HTTP proxies to listen on. Defaults to 8000.

# PORT_TO_QUERY should be consistent with the port assigned via serve run -p/--port
PORT_TO_QUERY = 36000  # Defaults to 8000

headers = {"Content-Type": "application/json"}

output = requests.get(f"http://180.184.76.150:{PORT_TO_QUERY}/health_check")
print(output)
print(output.text)

output = requests.get(f"http://180.184.76.150:{PORT_TO_QUERY}/engine_check")
print(output)
print(output.text)

prompt = "介绍一下北京"
input_data = {"prompt": prompt, "stream": False}
output = requests.post(f"http://180.184.76.150:{PORT_TO_QUERY}/completions", headers=headers, json=input_data)
print(output)
print(output.text)

# This is the case that we add sampling parameters into the request body
prompt = "Introduce some landmarks in London."
input_data = {"prompt": prompt, "stream": False, "parameters": {"do_sample": False, "max_new_tokens": 32}}
output = requests.post(f"http://180.184.76.150:{PORT_TO_QUERY}/completions", headers=headers, json=input_data)
print(output)
print(output.text)

# {
#   "prompt": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCount up from 1 to 500.\n\n### Response:\n",
#   "stream": false
# }

# Streaming
prompt = "Introduce some landmarks in London"
input_data = {"prompt": prompt, "stream": True, "parameters": {"do_sample": False, "max_new_tokens": 32}}
response = requests.post(f"http://180.184.76.150:{PORT_TO_QUERY}/completions", json=input_data, stream=True)
response.raise_for_status()
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk.decode("utf-8"), end="\n")
