
import yaml
from itertools import product
import pandas as pd
import os

dataset = "Venus"
vc_nums = f"./data/{dataset}/vc_config.csv"
config_dir = f"./data/{dataset}/config"

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

vc_df = pd.read_csv( vc_nums, index_col=0)
vcs = vc_df.index.tolist()

vc_filter = ["vcvGl", "vcHvQ", "vcJsw"]

for vc in vc_filter:
    if vc not in vcs:
        raise ValueError(f"vc {vc} is not in the vc list {vc_filter}")


schedulers = ['lucid-fixed', 'lucid-continue', 'lucid-nogpu', 'lucid-alwaysgpu']
# thresholds = [0.1, 0.2, 0.3, 0.4]
thresholds = [0.25]
scale = [1,2,5,10]

    

for i, (scheduler, scale, threshold) in enumerate(product(schedulers, scale, thresholds)) :
    scheduler_dict = {vc:scheduler for vc in vc_filter}
    threshold_dict = {vc: threshold for vc in vc_filter}
    trace_scale = {vc: scale for vc in vc_filter}
    cluster_scale = {vc: scale for vc in vc_filter}
    
    vc_config = {"threshold":threshold_dict, 
                "scheduler":scheduler_dict,
                "trace_scale":trace_scale,
                "cluster_scale":cluster_scale
                }
    
    f = open(f"./data/{dataset}/config/{i}.yaml", 'w')
    yaml.dump(vc_config, f)