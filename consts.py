import os

import numpy as np

DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SEED = 42
RNG = np.random.default_rng(seed=SEED)
