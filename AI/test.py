import os

label_dir = "C:/Users/Group8/Desktop/dataset/labels/train"
class_counts = [0, 0, 0, 0]

for file in os.listdir(label_dir):
    if file.endswith(".txt"):
        with open(os.path.join(label_dir, file), 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1

print("Class frequencies in training set:")
for i, count in enumerate(class_counts):
    print(f"Class {i}: {count}")



