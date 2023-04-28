import os
from utils.constants import LOG

run_list = [
    "./test_model.py",
    "./baseline_train.py",
]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
for filename in run_list:
    LOG(f"RUNNING {filename}")
    os.system(f"python {filename}")
