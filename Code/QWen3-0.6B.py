from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import json
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"当前设备: {device}")

#4-bit量化配置（节省显存）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,  # 改为 8-bit
#     llm_int8_threshold=6.0,  # 8-bit 量化阈值
#     llm_int8_has_fp16_weight=False,  # 不使用 fp16 权重
# )

# 加载模型和tokenizer
model_path = "/home/lizixian/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"  # 可替换为其他QWen3模型
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    #device_map="auto",
    device_map={"": 0},
    #device_map={"": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"},  # 关键修改
    trust_remote_code=True
)
model.enable_input_require_grads()  # 启用梯度计算

model.config.use_cache = False
model.config.pretraining_tp = 1

# data = load_dataset(
#     path='json',  # 数据格式为JSON
#     data_files='train.json'  # 本地JSON文件路径（若不在当前目录，需写绝对路径，如'./data/apple_data.json'）
# )

with open('train_new.json', 'r', encoding='utf-8') as f: 
    data = json.load(f)

#print(data[0])

# 转换为Dataset格式
dataset = Dataset.from_list(data)

# 数据预处理函数（格式化输入）
def process_function(examples):
    prompts = []
    for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        # 构造QWen3的对话格式（参考官方文档）
        prompt = f"<s>[INST] {instr} {inp} [/INST] {out}</s>"
        prompts.append(prompt)
    # 分词（截断/填充到最大长度）
    return tokenizer(prompts, truncation=True, max_length=2048, padding="max_length")

# 应用预处理
tokenized_dataset = dataset.map(
    process_function,
    batched=True,
    remove_columns=["instruction", "input", "output"]  # 移除不需要的列
)

lora_config = LoraConfig(
    r=8,  # LoRA秩（越大表达能力越强，显存消耗越高）
    lora_alpha=16,  # 缩放因子
    target_modules=[  # QWen3的注意力层模块名（需根据模型结构调整）
        "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力Q/K/V/O投影层
        "gate_proj", "up_proj", "down_proj"  # FFN层（可选，增强微调效果）
    ],
    lora_dropout=0.05,
    bias="none",  # 不微调偏置参数
    task_type="CAUSAL_LM"  # 因果语言模型任务
)

# 将LoRA应用到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 查看可训练参数比例（通常<1%）

training_args = TrainingArguments(
    output_dir="./qwen3-lora-finetune",  # 模型保存路径
    per_device_train_batch_size=1,  # 单卡batch size（根据显存调整）
    gradient_accumulation_steps=16,  # 梯度累积（等效增大batch size）
    learning_rate=2e-4,  # LoRA学习率（通常比全量微调大10-100倍）
    num_train_epochs=3,  # 训练轮数
    logging_steps=10,
    save_steps=100,
    fp16=True,  # 混合精度训练（节省显存）
    optim="paged_adamw_8bit",  # 8bit优化器
    report_to="none"  # 不使用wandb等日志工具
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    peft_config=lora_config,
    #dataset_text_field="text",  # 新版参数名（小写 t）
    #max_seq_length=4096,
    #tokenizer=tokenizer,
    args=training_args,
    #packing=False,
)

# 开始训练
print("开始训练")
trainer.train()

model.save_pretrained("qwen3_06-lora-adapter_new")  # 仅保存LoRA参数


