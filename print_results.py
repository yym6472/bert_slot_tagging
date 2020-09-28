import os
import json

results = []
for filename in os.listdir("./output"):
    if not os.path.exists(os.path.join("./output", filename, "test_metrics.txt")):
        continue
    with open(os.path.join("./output", filename, "test_metrics.txt"), "r") as f:
        results.append((json.loads(f.readline().strip()[25:].replace("'", "\""))["f1"], filename))

results = sorted(results, key=lambda x: x[0], reverse=True)
for score, setting in results:
    print(f"{score:.6f} {setting}")