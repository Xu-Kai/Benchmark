import requests
import time

test_sentences = [
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCount up from 1 to 500.\n\n### Response:\n"
    "介绍一下北京",
    "介绍一下南京",
    "介绍一下东京",
]

headers = {"Content-Type": "application/json"}

outputs = []
for text in test_sentences:
    input_data = {"prompt": text, "stream": False, "parameters": {"do_sample": False, "max_new_tokens": 256}}
    output = requests.post("http://180.184.76.150:36000/completions", headers=headers, json=input_data)
    outputs.append(output)

print("Result returned:")
for output in outputs:
    print("output.text: ", output.text)

for text in test_sentences:
    input_data = {"prompt": text, "stream": True, "parameters": {"do_sample": False, "max_new_tokens": 256}}
    response = requests.post(f"http://180.184.76.150:36000/completions", headers=headers, json=input_data, stream=True)
    response.raise_for_status()
    prev_timestamp = time.time()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        current_timestamp = time.time()
        time_interval_ms = int((current_timestamp - prev_timestamp) * 1000)
        print(f"Time Interval: {time_interval_ms} ms {chunk}")
        prev_timestamp = current_timestamp
