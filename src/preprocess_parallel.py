import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ============================================
# PATH
# ============================================

INPUT_FILE = r"C:\Users\User\OneDrive\Desktop\TRANSTONE\data\processed\parallel_corpus.csv"
OUTPUT_DIR = r"C:\Users\User\OneDrive\Desktop\TRANSTONE\data\processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# LOAD
# ============================================

df = pd.read_csv(INPUT_FILE, encoding="utf-8")

print("Columns detected:", df.columns.tolist())

# ============================================
# 🔴 FIX COLUMN NAMES HERE
# ============================================

df = df.rename(columns={
    "malayalam": "source_malayalam",
    "english": "target_english"
})

# Now safe to select
df = df[["source_malayalam", "target_english"]]

# ============================================
# CLEAN
# ============================================

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["source_malayalam"] = df["source_malayalam"].astype(str).str.strip()
df["target_english"] = df["target_english"].astype(str).str.strip()

df = df[(df["source_malayalam"] != "") & (df["target_english"] != "")]

# Better split stability
df["length_bin"] = df["source_malayalam"].apply(lambda x: min(len(x.split()) // 5, 5))

print("Total cleaned sentence pairs:", len(df))

# ============================================
# SPLIT
# ============================================

train_df, valid_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    stratify=df["length_bin"]
)

# Remove helper column
train_df = train_df.drop(columns=["length_bin"])
valid_df = valid_df.drop(columns=["length_bin"])

# ============================================
# SAVE
# ============================================

train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
valid_df.to_csv(os.path.join(OUTPUT_DIR, "valid.csv"), index=False)

print("✅ Kaggle preprocessing complete.")
print("Train:", len(train_df))
print("Valid:", len(valid_df))
