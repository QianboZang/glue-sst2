import torch
from torch.utils.data import DataLoader

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline, set_seed

from tqdm import tqdm
import re
import numpy as np


def convert_into_prompt_template(system_prompt, user_message):
    # Convert the dataset into the prompt template format as follows:
    # [INST] <>
    # {{ system_prompt }}
    # <>

    # {{ user_message }} [/INST]
    # Sentiment: {{ label }}       ## label is neccessary for training, but not for inference

    prompt = "<> \n" + system_prompt + "\n<>\n\nSentence:" + user_message
    text = f"[INST] {prompt} [/INST]\nSentiment:"

    return text


def map_dataset(system_prompt, dataset):
    # Convert the dataset into the format required by the model
    def convert(sentence):
        converted_inputs = convert_into_prompt_template(system_prompt, sentence)
        return {'text': converted_inputs}

    return dataset.map(convert, input_columns=['sentence'], batched=False)


data_name = ["glue", "sst2"]
system_prompt = "You are a helpful, respectful and honest sentiment analysis assistant. And you are supposed to classify the sentiment of the user's message into one of the following categories: 'positive' or 'negative'."
model_name = 'llama3/results'
batch_size = 9
set_seed(3407)
device_map = 'auto'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_val_sets = load_dataset(data_name[0], data_name[1], split='validation')
# Convert digital labels to text labels
processed_val_sets = map_dataset(system_prompt, raw_val_sets)
# print(processed_val_sets[0]['text'])

# #############################
# # evaluate the model
# #############################


tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
# use left padding for the model to increase the accuracy
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
)

label_map = {
    'negative': 0,
    'positive': 1,
}

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=150, batch_size=batch_size,)
# use dataloader to batchlize the dataset
dataloader = DataLoader(processed_val_sets, batch_size=batch_size, shuffle=False)

eval_set = []
for i, batch in tqdm(enumerate(dataloader)):
    # sentences = np.reshape(batch['text'], (-1, )).tolist()
    sentences = batch['text']
    labels = batch['label']
    results = pipe(sentences)
    results = np.reshape(results, (-1, ))
    results = [re.findall(r"Sentiment:(.+)", result['generated_text'])[0] for result in results]
    # results = [label_map[result.strip().split()[0]] for result in results]
    preds = [list(filter(None, re.split(r'[^a-zA-Z]+', result.strip())))[0] for result in results]
    
    compared = [int(label_map[pred]==label) for pred, label in zip(preds, labels)]
    # print(results)
    # print(compared)
    eval_set = eval_set + compared

# eval_set = []
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=150)
# for _, val in enumerate(tqdm(raw_val_sets)):
#     sentence = val['sentence']
#     prompt = "<> \n" + system_prompt + "\n<>\n\nSentence: " + sentence
#     result = pipe(f"[INST] {prompt} [/INST]\nSentiment:")[0]['generated_text']
#     match = re.findall(r"Sentiment: (.+)", result)
#     pred = label_map[match[0].strip().split()[0]]
#     if pred == val["label"]:
#         eval_set.append(1)
#     else:
#         eval_set.append(0)
#         print(f"Predicted: {pred}, Actual: {val['label']}, Sentence: {sentence}, match: {match}")

print(f"Accuracy: {sum(eval_set) / len(eval_set)}")
print(eval_set)
    