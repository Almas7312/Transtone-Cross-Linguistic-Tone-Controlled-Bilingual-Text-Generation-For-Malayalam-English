# =========================
# IMPORTS
# =========================
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# =========================
# CONFIG
# =========================
MODEL_NAME = "facebook/nllb-200-distilled-600M"

KAGGLE_TRAIN = "data/processed/train.csv"
KAGGLE_TEST = "data/processed/valid.csv"

TONE_FILE = "data/processed/custom_tone_dataset.csv"

OUTPUT_DIR = "./nllb_final_model"

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128

BATCH_SIZE = 4
EPOCHS = 1
LR = 2e-4

# =========================
# DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD MODEL
# =========================
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# =========================
# ADD SPECIAL TOKENS
# =========================
special_tokens = [
    "<formal>", "<informal>", "<poetic>",
    "<task:en2ml>", "<task:ml2en>",
    "<task:en2en>", "<task:ml2ml>"
]

tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
model.resize_token_embeddings(len(tokenizer))

model.to(device)

# =========================
# LANGUAGE DETECTION
# =========================
def is_malayalam(text):
    return any('\u0D00' <= c <= '\u0D7F' for c in str(text))

# =========================
# PROCESS KAGGLE DATA
# =========================
def process_kaggle(df):
    rows = []

    for _, row in df.iterrows():
        if "source_text" in df.columns:
            src = str(row["source_text"])
            tgt = str(row["target_text"])
        elif "source_malayalam" in df.columns:
            src = str(row["source_malayalam"])
            tgt = str(row["target_english"])
        else:
            continue

        src_ml = is_malayalam(src)
        tgt_ml = is_malayalam(tgt)

        if src_ml and not tgt_ml:
            rows.append({"input_text": f"<task:ml2en> {src}", "target_text": tgt})
            rows.append({"input_text": f"<task:en2ml> {tgt}", "target_text": src})

        elif not src_ml and tgt_ml:
            rows.append({"input_text": f"<task:en2ml> {src}", "target_text": tgt})
            rows.append({"input_text": f"<task:ml2en> {tgt}", "target_text": src})

        elif src_ml and tgt_ml:
            rows.append({"input_text": f"<task:ml2ml> {src}", "target_text": tgt})
            rows.append({"input_text": f"<task:ml2ml> {tgt}", "target_text": src})

        else:
            rows.append({"input_text": f"<task:en2en> {src}", "target_text": tgt})
            rows.append({"input_text": f"<task:en2en> {tgt}", "target_text": src})

    return pd.DataFrame(rows)

# =========================
# PROCESS TONE DATA
# =========================
def process_tone(df):
    rows = []

    for _, row in df.iterrows():
        src = str(row["source_text"])
        tgt = str(row["target_text"])
        tone = str(row["tone"]).strip().lower()

        src_ml = is_malayalam(src)
        tgt_ml = is_malayalam(tgt)

        if src_ml and not tgt_ml:
            task = "ml2en"
        elif not src_ml and tgt_ml:
            task = "en2ml"
        elif src_ml and tgt_ml:
            task = "ml2ml"
        else:
            task = "en2en"

        rows.append({
            "input_text": f"<task:{task}> <{tone}> {src}",
            "target_text": tgt
        })

    return pd.DataFrame(rows)

# =========================
# 🔥 BALANCE FUNCTION (KEY FIX)
# =========================
def balance_tasks(df, target=1500):
    balanced = []
    tasks = ["en2ml", "ml2en", "ml2ml", "en2en"]

    for task in tasks:
        subset = df[df["input_text"].str.contains(f"<task:{task}>")]

        if len(subset) == 0:
            continue

        if len(subset) >= target:
            subset = subset.sample(target, random_state=42)
        else:
            subset = subset.sample(min(len(subset)*2, target), replace=True, random_state=42)

        balanced.append(subset)

    return pd.concat(balanced)

# =========================
# LOAD DATA
# =========================
print("Loading datasets...")

kaggle_train_df = pd.read_csv(KAGGLE_TRAIN)
kaggle_test_df = pd.read_csv(KAGGLE_TEST)
tone_df = pd.read_csv(TONE_FILE)

# =========================
# PROCESS
# =========================
kaggle_train_df = process_kaggle(kaggle_train_df)
kaggle_test_df = process_kaggle(kaggle_test_df)

tone_df = process_tone(tone_df)

tone_train_df = tone_df.sample(frac=0.9, random_state=42)
tone_test_df = tone_df.drop(tone_train_df.index)

print("Kaggle train:", len(kaggle_train_df))
print("Tone train:", len(tone_train_df))

# =========================
# 🔥 APPLY BALANCING
# =========================
kaggle_train_balanced = balance_tasks(kaggle_train_df, target=1500)
kaggle_test_balanced = balance_tasks(kaggle_test_df, target=500)

train_df = pd.concat([kaggle_train_balanced, tone_train_df]).sample(frac=1, random_state=42)
test_df = pd.concat([kaggle_test_balanced, tone_test_df]).sample(frac=1, random_state=42)

print("Final train:", len(train_df))
print("Final test:", len(test_df))

# =========================
# DATASET
# =========================
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# =========================
# TOKENIZE
# =========================
def preprocess(example):
    inputs = tokenizer(
        example["input_text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        example["target_text"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

# =========================
# LORA
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# =========================
# TRAINING
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

print("Starting training...")
trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ TRAINING COMPLETE")
