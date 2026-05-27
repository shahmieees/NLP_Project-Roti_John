import pandas as pd
import re
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv("GKSL3k_original.csv")

# 2. Select relevant columns
df = df[[
    "Word level Korean Language (WKL) sentence",
    "Gloss level Korean Sign Language (GKSL) sentence"
]]

# Rename columns
df.columns = ["korean", "gloss"]

# 3. Cleaning function
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

df["korean"] = df["korean"].apply(clean_text)
df["gloss"] = df["gloss"].apply(clean_text)

# Remove empty rows
df = df.dropna()
df = df[(df["korean"] != "") & (df["gloss"] != "")]

print("After cleaning:", len(df))


# 4. Tagging function (QUESTION / STATEMENT)
def get_sentence_type(korean):
    korean = korean.strip()
    if korean.endswith("?"):
        return "[QUESTION]"
    else:
        return "[STATEMENT]"

df["type"] = df["korean"].apply(get_sentence_type)

# 5. Add tag to gloss (INPUT side)
df["gloss_tagged"] = df["type"] + " " + df["gloss"]

# Optional: check distribution
print(df["type"].value_counts())


# 6. Grouped split (IMPORTANT → avoid leakage)
unique_gloss = df["gloss"].unique()

train_g, temp_g = train_test_split(
    unique_gloss,
    test_size=0.2,
    random_state=42
)

val_g, test_g = train_test_split(
    temp_g,
    test_size=0.5,
    random_state=42
)

train_df = df[df["gloss"].isin(train_g)]
val_df   = df[df["gloss"].isin(val_g)]
test_df  = df[df["gloss"].isin(test_g)]

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))


# 7. Save to KoBART format (.source / .target)
def save_txt(df, src_path, tgt_path):
    df["gloss_tagged"].to_csv(src_path, index=False, header=False)
    df["korean"].to_csv(tgt_path, index=False, header=False)

save_txt(train_df, "train.source", "train.target")
save_txt(val_df, "val.source", "val.target")
save_txt(test_df, "test.source", "test.target")


# 8. Sanity check (VERY IMPORTANT)
with open("train.source", encoding="utf-8") as f1, \
     open("train.target", encoding="utf-8") as f2:
    
    print("\nSample pairs:\n")
    for _ in range(5):
        print("SRC:", f1.readline().strip())
        print("TGT:", f2.readline().strip())
        print()
