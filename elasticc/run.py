import os
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

m = Model(
    stage="1",
    path_to_trainingset=path_to_trainingset,
    n_iter=1,
    random_state=42,
    one_alert_per_stock=False,
)
m.split_sample()
# m.train()
# m.evaluate()
