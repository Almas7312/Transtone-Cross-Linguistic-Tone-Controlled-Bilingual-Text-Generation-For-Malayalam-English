import pandas as pd
import re
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
INPUT_FILE = "data/processed/custom_tone_dataset.csv"
TRAIN_OUT = "data/processed/tone_train.csv"
TEST_OUT = "data/processed/tone_test.csv"

# =========================
# CLEAN FUNCTIONS
# =========================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def is_malayalam(text):
    return any('\u0D00' <= c <= '\u0D7F' for c in str(text))

# =========================
# LOAD
# =========================
df = pd.read_csv(INPUT_FILE)
print("Initial samples:", len(df))

# =========================
# CLEAN
# =========================
df["source_text"] = df["source_text"].apply(clean_text)
df["target_text"] = df["target_text"].apply(clean_text)
df["tone"] = df["tone"].str.lower().str.strip()

valid_tones = ["formal", "informal", "poetic"]
df = df[df["tone"].isin(valid_tones)]

df = df[df["source_text"] != df["target_text"]]
df = df.drop_duplicates(subset=["source_text", "target_text", "tone"])

print("After cleaning:", len(df))

# =========================
# TASK DETECTION
# =========================
df["src_ml"] = df["source_text"].apply(is_malayalam)
df["tgt_ml"] = df["target_text"].apply(is_malayalam)

def get_task(row):
    if not row["src_ml"] and row["tgt_ml"]:
        return "en2ml"
    elif row["src_ml"] and not row["tgt_ml"]:
        return "ml2en"
    elif not row["src_ml"] and not row["tgt_ml"]:
        return "en2en"
    elif row["src_ml"] and row["tgt_ml"]:
        return "ml2ml"
    return "unknown"

df["task"] = df.apply(get_task, axis=1)
df = df[df["task"] != "unknown"]

print("\nRAW TASK DISTRIBUTION:")
print(df["task"].value_counts())

# =========================
# 🔥 SMART BALANCING (NO DATA DESTRUCTION)
# =========================
TARGET = 1000

balanced = []

for task in ["en2ml", "ml2en", "ml2ml", "en2en"]:
    subset = df[df["task"] == task]

    if len(subset) == 0:
        print(f"❌ WARNING: No data for {task}")
        continue

    if len(subset) >= TARGET:
        subset = subset.sample(TARGET, random_state=42)
    else:
        # limited upsampling (avoid huge duplication)
        subset = subset.sample(min(len(subset)*2, TARGET), replace=True, random_state=42)

    balanced.append(subset)

df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter smart balancing:")
print(df["task"].value_counts())

# =========================
# 🔥 REMOVE DUPLICATES AFTER BALANCING
# =========================
before = len(df)
df = df.drop_duplicates(subset=["source_text", "target_text", "tone"])
after = len(df)

print(f"\nRemoved duplicates after balancing: {before - after}")
print(f"Remaining samples: {after}")

# =========================
# BUILD INPUT
# =========================
def build_input(row):
    return f"<task:{row['task']}> <{row['tone']}> {row['source_text']}"

df["input_text"] = df.apply(build_input, axis=1)

final_df = df[["input_text", "target_text"]]

# =========================
# SPLIT
# =========================
train_df, test_df = train_test_split(
    final_df,
    test_size=0.1,
    random_state=42,
    stratify=df["task"]
)

# =========================
# SAVE
# =========================
train_df.to_csv(TRAIN_OUT, index=False)
test_df.to_csv(TEST_OUT, index=False)

print("\n✅ FINAL OUTPUT")
print("Train:", len(train_df))
print("Test:", len(test_df))
