import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("./finalmodel", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer_path = "M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

# 设置填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# 生成超长文本
input_text = "写一篇短篇小说，关于《巷子里的亡灵》"  # 可以根据需要修改起始文本
# 对输入文本进行编码并生成注意力掩码
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True).to(model.device)

# 设置生成参数
max_length = 5000  # 生成文本的最大长度
temperature = 0.7  # 控制生成文本的随机性
num_beams = 5  # 束搜索的束数
do_sample = True  # 设置为 True 使 temperature 生效

output = model.generate(
    **inputs,
    max_length=max_length,
    min_length=2000,
    temperature=temperature,
    num_beams=num_beams,
    no_repeat_ngram_size=2,  # 避免重复的 n-gram
    early_stopping=True,
    do_sample=do_sample
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)