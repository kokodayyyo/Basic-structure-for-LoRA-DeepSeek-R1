import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import BitsAndBytesConfig
from transformers import get_scheduler
from torch.optim import AdamW

# **1. 8-bit 量化**
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 启用 8-bit 量化
    llm_int8_threshold=6.0
)

model_name_or_path = "M"
model = AutoModelForCausalLM.from_pretrained(  #AutoModelForCausalLM 专门用于加载用于因果语言建模（Causal Language Modeling，CLM）的预训练模型。
    model_name_or_path,
    device_map="auto",
    quantization_config=bnb_config,  # 传入 8-bit 量化配置
    trust_remote_code=True
)

# **2. LoRA 适配**
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    bias="none",
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# **3. 加载数据集和分词器**
file_path = "福尔摩斯探案集.txt"
dataset = load_dataset('text', data_files=file_path)
tokenizer = AutoTokenizer.from_pretrained("M", trust_remote_code=True)

# **4. 预处理** 定义一个预处理函数，用于对数据集中的每个样本进行处理
def preprocess_function(example):
    inputs = tokenizer(
        example["text"],
        truncation=True,  #超过指定的最大长度，将对文本进行截断
        max_length=256,  #最大长度
        padding="max_length"  #填充
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# **5. 训练参数**
training_args = TrainingArguments(
    output_dir="./result",
    num_train_epochs=12,
    per_device_train_batch_size=4,  # 8-bit 量化后可以增大 batch size
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    prediction_loss_only=True,
    learning_rate=5e-5,  # 初始学习率
    warmup_steps=600,  # 热身步数
)

# **6. 自定义优化器**
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

# **7. 动态学习率调度器**
num_training_steps = training_args.num_train_epochs * len(tokenized_dataset["train"]) // training_args.per_device_train_batch_size
lr_scheduler = get_scheduler(
    name="linear",  # 线性衰减策略
    optimizer=optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=num_training_steps
)

# **8. 训练**
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    optimizers=(optimizer, lr_scheduler)  # 传入自定义优化器和学习率调度器
)

trainer.train()
trainer.save_model("./finalmodel")