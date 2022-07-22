import os, time
import logging
from train_models import Model

path_to_trainingset = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
    "elasticc_feature_trainingset_v6.pkl",
)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

for i, stage in enumerate(["2a"]):

    m = Model(
        stage=stage,
        path_to_trainingset=path_to_trainingset,
        n_iter=200,
        random_state=65 + i,
        grid_search_sample_size=10000,
    )
    m.split_sample()
    # m.train()
    m.evaluate()
