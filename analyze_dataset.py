import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load data (it's 540MB, loading to memory takes ~1-2 seconds)
print("Loading dataset...")
with open('hotpot_train_v1.1.json', 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} examples.")

levels = Counter()
types = Counter()
context_lengths = []

for item in data:
    levels[item.get('level', 'unknown')] += 1
    types[item.get('type', 'unknown')] += 1
    context_lengths.append(len(item.get('context', [])))

print("Levels:", levels)
print("Types:", types)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Question Level
axs[0].bar(levels.keys(), levels.values(), color=['skyblue', 'salmon', 'lightgreen', 'gray'])
axs[0].set_title('Distribution of Question Difficulty Levels')
axs[0].set_ylabel('Number of Queries')
axs[0].set_xlabel('Difficulty Level')

# Plot 2: Question Type
axs[1].bar(types.keys(), types.values(), color=['violet', 'gold', 'gray'])
axs[1].set_title('Distribution of Reasoning Types')
axs[1].set_ylabel('Number of Queries')
axs[1].set_xlabel('Reasoning Type')

plt.tight_layout()
plt.savefig('dataset_insights.pdf')
print("Saved dataset_insights.pdf")
