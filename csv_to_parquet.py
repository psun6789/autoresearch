import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

df = pd.read_csv("data/data.csv")

formatted = []
for _, row in df.iterrows():
    text = f"Classify the following text:\n{row['text']}\nAnswer: {row['label']}"
    formatted.append(text)

new_df = pd.DataFrame({"text": formatted})

# Create cache dir
data_dir = os.path.expanduser("~/.cache/autoresearch/data")
os.makedirs(data_dir, exist_ok=True)

# Save TRAIN shard
pq.write_table(pa.Table.from_pandas(new_df), os.path.join(data_dir, "shard_00000.parquet"))

# Save VALIDATION shard (REQUIRED NAME)
pq.write_table(pa.Table.from_pandas(new_df.sample(frac=0.2)), os.path.join(data_dir, "shard_06542.parquet"))

print("✅ Data ready")