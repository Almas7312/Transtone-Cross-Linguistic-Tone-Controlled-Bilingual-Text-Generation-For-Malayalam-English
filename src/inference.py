# =========================
# IMPORTS
# =========================
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "facebook/nllb-200-distilled-600M"
MODEL_PATH = "./nllb_final_model"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# LOAD TOKENIZER
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# =========================
# LOAD BASE MODEL
# =========================
print("Loading base model...")
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# IMPORTANT
model.resize_token_embeddings(len(tokenizer))

# =========================
# LOAD LORA
# =========================
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, MODEL_PATH)

model = model.to(device)
model.eval()

# =========================
# TASK + TONE MAP
# =========================
task_map = {
    "1": "en2ml",
    "2": "ml2en",
    "3": "en2en",
    "4": "ml2ml"
}

tone_map = {
    "1": "formal",
    "2": "informal",
    "3": "poetic"
}

# =========================
# GENERATION FUNCTION
# =========================
def generate_output(text, task, tone):
    input_text = f"<task:{task}> <{tone}> {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,              # stable translation
            early_stopping=True,
            length_penalty=1.0,
            no_repeat_ngram_size=3   # avoids repetition
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# =========================
# INTERACTIVE LOOP
# =========================
print("\n✅ Model ready!\n")

while True:

    print("\n==== SELECT TASK ====")
    print("1: English → Malayalam")
    print("2: Malayalam → English")
    print("3: English → English")
    print("4: Malayalam → Malayalam")

    task_choice = input("Enter choice (or 'q' to quit): ").strip()
    if task_choice.lower() == "q":
        break

    if task_choice not in task_map:
        print("❌ Invalid task")
        continue

    print("\n==== SELECT TONE ====")
    print("1: Formal")
    print("2: Informal")
    print("3: Poetic")

    tone_choice = input("Enter choice: ").strip()

    if tone_choice not in tone_map:
        print("❌ Invalid tone")
        continue

    text = input("\nEnter your sentence: ").strip()

    if not text:
        print("❌ Empty input")
        continue

    print("\n🔄 Generating...\n")

    result = generate_output(
        text,
        task_map[task_choice],
        tone_map[tone_choice]
    )

    print("✅ OUTPUT:")
    print(result)
