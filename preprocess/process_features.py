import subprocess
import glob
import re
import pandas as pd
import json

# First stage of processing features:
# - Will combine different models beloning to the same taxonomy
# - Will combine RiseDecline features with stock info such as redshift

# Run this each time for one taxclass
run_tax = 2246

# Storing counts 
# 2222: Got 954288 feature sets for 118640 transients.
# 2223: Got 1570977 feature sets for 163032 transients.
# 2224: Got 2313110 feature sets for 285403 transients.
# 2225: Got 197306 feature sets for 26735 transients.
# 2226: Got 210834 feature sets for 27568 transients.
# 2232: Got 10156 feature sets for 4233 transients.
# 2233: Got 1298 feature sets for 870 transients.
# 2234: Got 59477 feature sets for 8085 transients.
# 2235: Got 1210348 feature sets for 18388 transients.
# 2242: Got 1922874 feature sets for 68089 transients.
# 2243: Got 775449 feature sets for 65662 transients.
# 2244: Got 152107 feature sets for 6822 transients.
# 2245: Got 64639 feature sets for 7443 transients.
# 2246: Got 2281738 feature sets for 63646 transients.
# 2322: Got 2438976 feature sets for 14440 transients.
# 2323: Got 3195739 feature sets for 15024 transients.
# 2324: Got 4354477 feature sets for 22054 transients.
# 2325: Got 3335128 feature sets for 68993 transients.
# 2332: Got 3203889 feature sets for 28140 transients.




# Finall model names
modfiles = glob.glob('hostinfo*_01.csv')
models = [re.search('hostinfo_(.+)_01.csv', modf)[1] for modf in modfiles]

# Taxonomy dict from https://github.com/LSSTDESC/elasticc/blob/main/taxonomy/taxonomy.ipynb
tax = {
	'SNIa-SALT3':2222, 
	'SNIc+HostXT_V19':2223, 
	'SNIb+HostXT_V19':2223, 
	'SNIc-Templates':2223, 
	'SNIcBL+HostXT_V19':2223, 
	'SNIb-Templates':2223, 
	'SNIIn-MOSFIT':2224, 
	'SNII-Templates':2224, 
	'SNIIb+HostXT_V19':2224, 
	'SNII-NMF':2224, 
	'SNII+HostXT_V19':2224, 
	'SNIIn+HostXT_V19':2224, 
	'SNIax':2225, 
	'SNIa-91bg':2226, 
	'KN_K17':2232, 
	'KN_B19':2232, 
	'Mdwarf-flare':2233,
	'dwarf-nova':2234, 
	'uLens-Single_PyLIMA':2235, 
	'uLens-Single-GenLens':2235, 
	'uLens-Binary':2235, 
	'SLSN-I_no_host':2242, 
	'SLSN-I+host':2242, 
	'TDE':2243, 
	'ILOT':2244, 
	'CART':2245, 
	'PISN':2246, 
	'Cepheid':2322, 
	'RRL':2323, 
	'd-Sct':2324, 
	'EB':2325, 
	'CLAGN':2332, 
	}
zkind = {'HOSTGAL_ZQUANT':1,'HOSTGAL2_ZQUANT':2,'HOSTGAL_ZSPEC':3,'default':0}
dtypes = {'bool_pure':"boolean", 'bool_rise':"boolean", 'bool_fall':"boolean", 'bool_peaked':"boolean", 'bool_fastrise':"boolean",'bool_fastfall':"boolean", 'bool_hasgaps':"boolean", 'success':"boolean"}

# Parse through all files
dftot = None
for modelname, modelid in tax.items():
	if not modelid==run_tax:
		continue
	print(modelname)
	
	for k in range(1,31):
		print('starting', k)
	
		hostfile = 'hostinfo_'+modelname+'_{:02d}.csv'.format(k)
		df = pd.read_csv(hostfile)
		htab = []
		for i, row in df.iterrows():			
			if row['z_source']=='default':
				htab.append({'stock':row['Unnamed: 0'],'host_sep':row['host_sep'],
				'galaxy_color':row['galaxy_color'],'z_source':0,'z':0} )
			else:
				htab.append({'stock':row['Unnamed: 0'],'host_sep':row['host_sep'],
				'galaxy_color':row['galaxy_color'],
				'z_source':zkind[row['z_source']],'z':json.loads(row['z_samples'])[1]})
		df_host = pd.DataFrame.from_dict(htab)

		featfile = 'risedec_'+modelname+'_{:02d}.csv'.format(k)
		print(featfile)
		df = pd.read_csv(featfile,dtype=dtypes)
		
		# Columns we can remove
		df = df.drop(columns=['Unnamed: 0','success'])

		# Start merge
		df = pd.merge( df, df_host, 
				how='inner', on='stock' )
		print(df.shape)
		
		
		#break
		if dftot is not None:
			dftot = pd.concat([dftot,df])
		else:
			dftot = df
		print(dftot.shape)
		
#		if k==2:
#			break
				
print('trying to save model', run_tax)
print('Got {} feature sets for {} transients.'.format(dftot.shape[0],len(dftot['stock'].unique())))
dftot.to_parquet('features_{}.parquet'.format(run_tax))

