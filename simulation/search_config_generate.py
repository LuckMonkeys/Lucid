
import yaml
from itertools import product
import pandas as pd
import os

dataset = "Venus"
vc_nums = f"./data/{dataset}/vc_config.csv"
config_dir = "./data/Venus/config"

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

vc_df = pd.read_csv( vc_nums, index_col=0)
vcs = vc_df.index.tolist()

vc_filter = ["vcvGl", "vcHvQ", "vcJsw"]
vc_trace_scale = {"vcvGl":5, "vcHvQ":5, "vcJsw":5}
vc_cluster_scale = {"vcvGl":2, "vcHvQ":2, "vcJsw":2}

for vc in vc_filter:
    if vc not in vcs:
        raise ValueError(f"vc {vc} is not in the vc list {vc_filter}")


schedulers = ['lucid-fixed', 'lucid-continue', 'lucid-nogpu', 'lucid-alwaysgpu']
thresholds = [0.1, 0.2, 0.3, 0.4]

    

for i, (scheduler, threshold) in enumerate(product(schedulers, thresholds)) :
    scheduler_dict = {vc:scheduler for vc in vc_filter}
    threshold_dict = {vc: threshold for vc in vc_filter}
    
    vc_config = {"threshold":threshold_dict, "scheduler":scheduler_dict}
    vc_config["trace_scale"] = vc_trace_scale
    vc_config["cluster_scale"] = vc_cluster_scale
    
    f = open(f"./data/Venus/config/{i}.yaml", 'w')
    yaml.dump(vc_config, f)