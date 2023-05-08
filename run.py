import os
from utils.constants import LOG

run_list = [
    "./test_model.py",
    "./prompt_entail_train.py",
]

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"
for filename in run_list:
    LOG(f"RUNNING {filename}")
    os.system(f"python {filename}")
