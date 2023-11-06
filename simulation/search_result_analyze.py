
import yaml
from itertools import product
import pandas as pd
import os
import numpy as np

result_dir = "./log_search/vc_node_factor_1.0"
config_dir = "./data/Venus/config" 
dataset = "Venus_Sept"
vcs = ["vcvGl", "vcHvQ", "vcJsw"]


#get name of config: Venus_Sept_0 -> 0
configs_names = [int(subdir.split('_')[-1]) for subdir in os.listdir(result_dir)]
configs_names.sort()

jcts = []
for name in configs_names:
    jct_csv = os.path.join(result_dir, f"{dataset}_{name}", "jct_avg_consolidate.csv")
    jct_pd = pd.read_csv(jct_csv)
    jct_pd.columns = ['vc', 'jct']
    jct = [jct_pd[jct_pd['vc']==vc]['jct'].values[0] for vc in vcs]

    jcts.append(jct)

# best config name for each vc
best_config_names = np.argmin(jcts, axis=0)



#generate the best config for cluster
base_config_path = os.path.join("./data/Venus/config", "0.yaml")
f = open(base_config_path, 'r')
best_config = yaml.safe_load(f)

parameters = list(best_config.keys())
for i, vc in enumerate(vcs):
    f = open(os.path.join(f"./data/Venus/config", f"{best_config_names[i]}.yaml") , 'r')
    best_config_for_vc = yaml.safe_load(f)
    for parameter in parameters:
        best_config[parameter][vc] = best_config_for_vc[parameter][vc]
    
import pprint
print("Best config for cluster")
pprint.pprint(best_config)







    
    
