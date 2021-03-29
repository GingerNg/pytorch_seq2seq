from enum import Enum

class MthStep(Enum):
    fold_split = "fold_split"
    process_data = "process_data"
    train = "train"

class STEPTYPE(Enum):
    train = "01"
    test = "02"
    build_vocab = "03"
    infer = "04"