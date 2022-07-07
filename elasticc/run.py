import os
from train_models import Model

path_to_trainingset = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
    "elasticc_feature_trainingset_v3",
)

m = Model(
    stage="2a", path_to_trainingset=path_to_trainingset, n_iter=1, random_state=42
)
m.split_sample()
m.train()
m.evaluate()
