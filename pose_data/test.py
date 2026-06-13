import numpy as np, pickle

# Check joints shape
joints = np.load("/Users/kritikabhat/Documents/MastersResearch/Code/pose_data/LectureMath_testing_joints.npy")
print("joints shape:", joints.shape)
print("joints dtype:", joints.dtype)

# Check labels pickle
with open("/Users/kritikabhat/Documents/MastersResearch/Code/pose_data/LectureMath_testing_ids_labels.pickle", "rb") as f:
    data = pickle.load(f)
print("pickle type:", type(data))
if isinstance(data, list):
    print("list length:", len(data))
    print("first item:", data[0] if len(data) > 0 else "empty")
    print("second item type:", type(data[1]))
    if hasattr(data[1], '__len__'):
        print("second item length:", len(data[1]))
        print("second item sample:", data[1][:5])

import pickle

with open("/Users/kritikabhat/Documents/MastersResearch/Code/pose_data/LectureMath_testing_ids_labels.pickle", "rb") as f:
    data = pickle.load(f)

print("tuple length:", len(data))
print("item 0 type:", type(data[0]))
print("item 0 sample:", data[0][:5])
print("item 1 type:", type(data[1]))
print("item 1 sample:", data[1][:5])

import pickle
import numpy as np
from collections import Counter

with open("/Users/kritikabhat/Documents/MastersResearch/Code/pose_data/LectureMath_testing_ids_labels.pickle", "rb") as f:
    sample_names, labels = pickle.load(f)

labels = np.array(labels)

# Label distribution
print("Label distribution:")
counts = Counter(labels.tolist())
total = len(labels)
for lab_idx in sorted(counts):
    pct = 100 * counts[lab_idx] / total
    print(f"  label {lab_idx}: {counts[lab_idx]:>7}  ({pct:5.2f}%)")

print(f"\nTotal segments: {total}")
print(f"Unique labels : {sorted(counts.keys())}")


with open("/Users/kritikabhat/Documents/MastersResearch/Code/pose_data/LectureMath_testing_ids_labels.pickle", "rb") as f:
    sample_names, labels = pickle.load(f)

# Extract unique video IDs from test set
test_videos = set()
for name in sample_names:
    parts = name.rsplit("_", 2)
    if len(parts) == 3:
        test_videos.add(parts[0])

print("Videos in test set:", sorted(test_videos))

# Check which of your 5 are in it
my_5 = ["00000_000_001", "00000_000_002", "00000_000_003", "00000_000_005","00000_001_001"]
for v in my_5:
    print(f"  {v}: {'IN test set ✅' if v in test_videos else 'NOT in test set ❌'}")