import subprocess
import glob
import re
import pandas as pd
import json
from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np
import sys

# Second stage of feature processing.
# This will for a specific taxonomy model possibly:
# - Cut down to a specific ndet or t_lc range.
# - Subselect a set of alerts for each unique transient
# - Split into a training and validation sample. 

# Run this each time for one taxclass
run_tax = int(sys.argv[1])
print(run_tax)

# Settings for early sample (in practice prob only used to ndet ~5)
#ndetrange = [0, 10]
#timerange = [0, 10]
#maxalert = 10
# Settings for late timeset
#ndetrange = [6, 99999]
#timerange = [6, 99999]
#maxalert = 10
# Settings for a smallish testset
ndetrange = [0, 99999]
timerange = [0, 99999]
maxalert = 3

valfrac = 0.2
random_state = 42

df = pd.read_parquet('features_{}.parquet'.format(run_tax))

print('startsize', df.shape)

im = (df['ndet']>=ndetrange[0]) & (df['ndet']<=ndetrange[1]) & (df['t_lc']>=timerange[0]) & (df['t_lc']<=timerange[1])
df = df[im]

print('after cuts', df.shape)

stocks = df['stock'].unique()
print('transients', len(stocks))

# Define training and validation subsets
valstock = np.random.choice( stocks, size=int(valfrac*len(stocks)), replace=False)

# Maybe do both things at once? 
# apply works but is very slow - a direct loop faster?
#dfgroup = df.groupby('stock').apply(
#	lambda x: x if x.shape[0]<=maxalert else x.sample(n=maxalert) )

train, validate = [], []
dfgroup = df.groupby('stock')
k = 0
for groupstock, group in dfgroup:
	if k % 10000==0:
		print(k, groupstock, group.shape)
	if groupstock in valstock:
		savelist = validate
	else:
		savelist = train
	if group.shape[0]>maxalert:
		savelist.append( group.sample(n=maxalert, random_state=random_state) )
	else:
		savelist.append( group )
		
	k+=1
		
df_train = pd.concat(train)
df_val = pd.concat(validate)

df_train.to_parquet('features_{}_ndet{}-{}_tlc{}-{}_maxalert{}_train.parquet'.format(run_tax, *ndetrange, *timerange, maxalert))
df_val.to_parquet('features_{}_ndet{}-{}_tlc{}-{}_maxalert{}_validate.parquet'.format(run_tax, *ndetrange, *timerange, maxalert))



plt.figure()
plt.subplot(3,1,1)
_, bins, __ = plt.hist(df['stock'].value_counts())

plt.subplot(3,1,2)
plt.hist(df_train['stock'].value_counts().append(df_val['stock'].value_counts()),bins=bins)
plt.hist(df_val['stock'].value_counts(),bins=bins)

plt.subplot(3,1,3)
_, bins, __ = plt.hist(df['t_lc'])
plt.hist(df_train['t_lc'],bins=bins)
plt.hist(df_val['t_lc'],bins=bins)
plt.show()



