import pandas as pd

df = pd.read_csv("data/data.csv")

with open("train.txt", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        f.write(f"Classify the following text:\n{row['text']}\nAnswer: {row['label']}\n\n")