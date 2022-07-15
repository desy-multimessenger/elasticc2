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


m = Model(
    stage="1",
    path_to_trainingset=path_to_trainingset,
    n_iter=100,
    random_state=60,
    grid_search_sample_size=20000,
)
m.split_sample()

t_start = time.time()

m.train()

t_end = time.time()

logger.info("------------------------------------")
logger.info("           FITTING DONE             ")
logger.info(f"  This took {t_end-t_start} seconds")
logger.info("------------------------------------")

m.evaluate()
