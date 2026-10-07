import pandas as pd
import json
import ast
from datasets import load_dataset
from sklearn.model_selection import train_test_split

GOAL_INSTANCES = 500

df = pd.read_csv("2WikiTestOriginal.csv")
strat_key = df["type"] + "_" + df["mixed"].astype(str) + "_" + df["label"].astype(str)
test_size = GOAL_INSTANCES / df.shape[0]
_, df  = train_test_split(df, stratify=strat_key, test_size=test_size, random_state=42)

refs = []
for i, row in df.iterrows():
    docs = []
    for doc in ast.literal_eval(row["evidences"]):
        docs.append(doc[0])
    refs.append(docs)

df["references"] = refs
df.to_csv("2WikiTest.csv", index=False)

#==============================================================

ds = load_dataset("hotpotqa/hotpot_qa", "distractor")
df_val = pd.DataFrame(ds["validation"])
df = pd.read_csv("HotpotTestOriginal.csv")

strat_key = df["type"] + "_" + df["level"] + "_" + df["mixed"].astype(str) + "_" + df["label"].astype(str)
test_size = GOAL_INSTANCES / df.shape[0]
_, df  = train_test_split(df, stratify=strat_key, test_size=test_size, random_state=42)

questions = df["question"]
df_filtered = df_val[df_val["question"].isin(questions)]  # Kind of useless since merge does it but ok
df = df.merge(
    df_filtered[["question", "supporting_facts"]], 
    on="question", 
    how="left"
)

for i, row in df.iterrows():
    refs = set(row["supporting_facts"]["title"])
    df.loc[i, "references"] = str(list(refs))

df = df.drop(columns=["supporting_facts"])
df.to_csv("HotpotTest.csv", index=False)

#==============================================================

ds = load_dataset("bdsaglam/musique")
df_val = pd.DataFrame(ds["validation"])
df = pd.read_csv("MusiqueTestOriginal.csv")
df = df[df["number_of_supporting_facts"] > 0]
# df = df[df["label"] == True] # Filter so that only when there's enough information, maybe. But the point is that I am using the complete wikipedia, so I think it doesnt really make much difference
strat_key = df["mixed"].astype(str) + "_" + df["label"].astype(str)
test_size = GOAL_INSTANCES / df.shape[0]
_, df  = train_test_split(df, stratify=strat_key, test_size=test_size, random_state=42)

questions = df["question"]
df_filtered = df_val[df_val["question"].isin(questions)] # Kind of useless since merge does it but ok
df_filtered.drop_duplicates(subset=["question"],inplace=True)
df = df.merge(
    df_filtered[["question", "paragraphs"]], 
    on="question", 
    how="left"
)

for i, row in df.iterrows():
    refs = set()
    for doc in row["paragraphs"]:
        if doc["is_supporting"] == True:
            refs.add(doc["title"])
    df.loc[i, "references"] = str(list(refs))

df = df.drop(columns=["paragraphs"])
df.to_csv("MusiqueTest.csv", index=False)