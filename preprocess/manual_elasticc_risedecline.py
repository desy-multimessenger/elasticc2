#!/usr/bin/env python
# coding: utf-8

import sys

from ampel.lsst.alert.load.ElasticcTrainingsetLoader import ElasticcTrainingsetLoader
from ampel.lsst.alert.ElasticcAlertSupplier import ElasticcAlertSupplier
from ampel.lsst.ingest.LSSTDataPointShaper import LSSTDataPointShaper
from ampel.contrib.hu.t2.T2TabulatorRiseDecline import T2TabulatorRiseDecline
from ampel.contrib.hu.t2.T2TabulatorRiseDecline import T2TabulatorRiseDeclineBase
from ampel.contrib.hu.t2.T2ElasticcRedshiftSampler import T2ElasticcRedshiftSampler
from ampel.dev.DevAmpelContext import DevAmpelContext
from ampel.log.AmpelLogger import AmpelLogger

from ampel.lsst.view.LSSTT2Tabulator import LSSTT2Tabulator
import pandas as pd



model = sys.argv[1]
run = int(sys.argv[2])


AMPEL_CONF = '/home/jnordin/github/ampel83elasticc2/Ampel-HU-astro/ampel_conf.yaml'



ctx = DevAmpelContext.load(
    config = AMPEL_CONF,
    db_prefix = 'dumpme'
)
logger = AmpelLogger.get_logger()



fpath = f'/home/jnordin/data/elasticc2/elasticc2_v1_june7/ELASTICC2_TRAIN_01_{model}/ELASTICC2_TRAIN_01_NONIaMODEL0-00{run:02d}'

# Configure an alert loader
config = {'file_path': fpath}
alertloader = ElasticcTrainingsetLoader(**config)

# Configure an alert supplier (not sure this is needed?)
config = {'loader': {'unit':'ElasticcTrainingsetLoader', 'config':{'file_path':fpath}}}
supplier = ElasticcAlertSupplier(**config)
shaper = LSSTDataPointShaper(logger=logger)

# Configure T2 units which are needed
config = {'tabulator':[{'unit':'LSSTT2Tabulator', 'config':{'zp':27.5}}], 'logger': logger}
t2rise = T2TabulatorRiseDecline(**config)
t2rise.post_init()

config = {'logger': logger}
t2z = T2ElasticcRedshiftSampler(**config)




summary = []
hostinfo = {}



for k, alert in enumerate(supplier):

    stock = alert.stock
    dps = shaper.process(alert.datapoints, stock)

    risdec = t2rise.process({}, dps )
    # Do not bloat with non-detection.
    if not risdec['success']:
        continue

    risdec['stock'] = stock
    summary.append( risdec )

    if not stock in hostinfo:
        dpobj = [dp for dp in dps if 'LSST_OBJ' in dp['tag']]
        hostinfo[stock] = t2z.process(dpobj[0] )

    if len(summary) % 1000==0:
        print(model, len(summary))



print('alert entries', len(summary))
print('host entries', len(hostinfo))


dfh = pd.DataFrame.from_dict(hostinfo, orient='index')
outf = f'hostinfo_{model}_{run:02d}.csv'
dfh.to_csv(outf)
dfs = pd.DataFrame.from_dict(summary)
outs = f'risedec_{model}_{run:02d}.csv'
dfs.to_csv(outs)
