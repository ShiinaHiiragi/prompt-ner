import torch

CLS_TOKEN = 101
SEP_TOKEN = 102
MASK_TOKEN = 103
PAD_TOKEN = 0

SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, PAD_TOKEN]
SPECIAL_ID = -1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LABEL_ENTITY = {
    "LOC": "地名",
    "ORG": "组织",
    "PER": "人名",
    "GPE": "地缘政治实体"
}
