
import yaml
from itertools import product
import pandas as pd
import os
import numpy as np

result_dir = "./log_search/vc_node_factor_1.0"
config_dir = "./data/Venus/config" 
dataset = "Venus_Sept"
vcs = ["vcvGl", "vcHvQ", "vcJsw"]

def get_best_config_for_cluster(result_dir, config_dir, dataset, vcs):
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
    base_config_path = os.path.join(config_dir, "0.yaml")
    f = open(base_config_path, 'r')
    best_config = yaml.safe_load(f)

    parameters = list(best_config.keys())
    for i, vc in enumerate(vcs):
        f = open(os.path.join(config_dir, f"{best_config_names[i]}.yaml") , 'r')
        best_config_for_vc = yaml.safe_load(f)
        for parameter in parameters:
            best_config[parameter][vc] = best_config_for_vc[parameter][vc]
        
    import pprint
    print("Best config for cluster")
    pprint.pprint(best_config)




def get_scale_down_results(result_dir, dataset, info, scales=4, exclude_profile_gpu=True):
    import pandas as pd
    
    configs_names = [int(subdir.split('_')[-1]) for subdir in os.listdir(result_dir)]
    configs_names.sort()

    values = []

    for name in configs_names:
            
        dir = os.path.join(result_dir, f"{dataset}_{name}") 
        if exclude_profile_gpu:
            jct_pd = pd.read_csv(f"{dir}/jct_avg_consolidate_execludeProfileGPU.csv")
        else:
            jct_pd = pd.read_csv(f"{dir}/jct_avg_consolidate.csv")

        jct_pd.columns = ['index', 'jct']

        values.append(jct_pd[jct_pd['index'] == "all"].values[0][1])
        
    df_dict = {"info":info}
    df_dict.update({f"scale_{scales[i]}":values[j:j+len(scales)] for i, j in enumerate(range(0, len(values), len(scales)))})
    df = pd.DataFrame.from_dict(df_dict)
    
    print(df)

schedulers = ['lucid-fixed', 'lucid-continue', 'lucid-nogpu', 'lucid-alwaysgpu']
scales = [1,2,5,10]

get_scale_down_results(result_dir, dataset, info=schedulers, scales=scales)



    
    
