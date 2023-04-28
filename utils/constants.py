import torch

CLS_TOKEN = 101
SEP_TOKEN = 102
MASK_TOKEN = 103
PAD_TOKEN = 0

SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, PAD_TOKEN]
SPECIAL_ID = -1

GRAM = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NULL_LABEL = "O"
LABEL_ENTITY = {
    "LOC": "地名",
    "ORG": "组织",
    "PER": "人名",
    "GPE": "地缘政治实体名"
}

LOG = lambda msg: print(f"\033[1;31m{msg}\033[0m")
