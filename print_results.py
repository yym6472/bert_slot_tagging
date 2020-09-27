import os
import json

for filename in os.listdir("./output"):
    if not os.path.exists(os.path.join("./output", filename, "test_metrics.txt")):
        continue
    with open(os.path.join("./output", filename, "test_metrics.txt"), "r") as f:
        print(filename, json.loads(f.readline().strip()[25:].replace("'", "\""))["f1"])
    