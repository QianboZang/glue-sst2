import torch

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, set_seed
from peft import LoraConfig
from trl import SFTTrainer


def create_label_text(label):
    label_map = {
        0 : 'negative',
        1 : 'positive',
    }

    return {'label_text': label_map[label]}


def convert_into_prompt_template(system_prompt, user_message, train=True, label_text=""):
    # Convert the dataset into the prompt template format as follows:
    # [INST] <>
    # {{ system_prompt }}
    # <>

    # {{ user_message }} [/INST]
    # Sentiment: {{ label }}       ## label is neccessary for training, but not for inference
    if train:
        text = f"[INST] <>\n{system_prompt}\n<>\n\nSentence: {user_message} [/INST]\nSentiment: {label_text} "
    else:
        text = "[INST] " + system_prompt + "\n<>\n\n" + "Sentence: " + user_message + "[/INST]\n" + "Sentiment: "

    return text


def map_dataset(system_prompt, dataset, train=True):
    # Convert the dataset into the format required by the model
    def convert(sentence, label_text):
        converted_inputs = convert_into_prompt_template(system_prompt, sentence, train, label_text)
        return {'text': converted_inputs, 'label_text': label_text}

    return dataset.map(convert, input_columns=['sentence', 'label_text'], batched=False, remove_columns=['sentence', 'label', 'idx', 'label_text'])



############################
# hyperparameters
############################
access_token = "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" 
data_name = ["glue", "sst2"]
base_model_name = "meta-llama/Meta-Llama-3-8B"
output_dir = "llama3/results"
# for prompt tuning
system_prompt = "You are a helpful, respectful and honest sentiment analysis assistant. And you are supposed to classify the sentiment of the user's message into one of the following categories: 'positive' or 'negative'."
user_prompt = "Classify the sentiment of the following sentence into one of the following categories: 'positive' or 'negative'."


# from training
max_seq_length = 256
learning_rate = 1e-3
batch_size = 4
accumulation_steps = 1
logging_steps = 10
max_steps = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on device:", device)
device_map = 'auto'
set_seed(3407)

############################
# Load Dataset
############################

raw_train_sets = load_dataset(data_name[0], data_name[1], split='train')
# Convert digital labels to text labels
train_sets = raw_train_sets.map(create_label_text, input_columns=['label'])
processed_train_sets = map_dataset(system_prompt, train_sets, train=True)

############################
# Load the tokenizer and llama-3 model
############################
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    trust_remote_code=True,
    token=access_token,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

#############################
# Train the model
#############################
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    max_steps=max_steps,
    gradient_accumulation_steps=accumulation_steps,
)

# use accelerator to train the model on multiple GPUs
trainer = SFTTrainer(
    model=base_model,
    train_dataset=processed_train_sets,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
# save the model
trainer.save_model(output_dir)
