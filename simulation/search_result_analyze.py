
#%%
import yaml
from itertools import product
import pandas as pd
import os
import numpy as np
# from natsort import 

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

    for name in configs_names[:len(info)*len(scales)]:
            
        dir = os.path.join(result_dir, f"{dataset}_{name}") 
        if exclude_profile_gpu:
            jct_pd = pd.read_csv(f"{dir}/jct_avg_consolidate_execludeProfileGPU.csv")
        else:
            jct_pd = pd.read_csv(f"{dir}/jct_avg_consolidate.csv")

        jct_pd.columns = ['index', 'jct']

        values.append(jct_pd[jct_pd['index'] == "all"].values[0][1])
    # print(values) 
    # values = values[:len(info)*len(scales)]
    df_dict = {"scales":scales}
    df_dict.update({f"{info[i]}":values[j:j+len(scales)] for i, j in enumerate(range(0, len(values), len(scales)))})
    # print(df_dict)
    df = pd.DataFrame.from_dict(df_dict)
    # df.to_csv(f"scale_down_results.csv", index=False)
    
    print(df)
    
result_dir = "./log_search/vc_node_factor_1.0"
# config_dir = "./data/Venus/config" 
dataset = "Venus_Sept"
# vcs = ["vcvGl", "vcHvQ", "vcJsw"]


#%%
#### result for all datasets
schedulers = ['lucid-fixed', 'lucid-continue', 'lucid-nogpu', 'lucid-alwaysgpu']
scales = [1,2,4,5]
for dataset, dir_name in zip(['Venus_Sept', "Philly", "MLaas"], ["venus", "philly", "mlaas"]):
    result_dir = f"./log_search_{dir_name}/vc_node_factor_1.0"
    print(f"Dataset: {dataset} node factor 1.0")
    get_scale_down_results(result_dir, dataset, info=schedulers, scales=scales)
print("Dataset: Philly node factor 0.8") 
result_dir = f"./log_search_philly/vc_node_factor_0.8"
dataset="Philly"
get_scale_down_results(result_dir, dataset, info=schedulers, scales=scales)

#### result for scale 1
#%%
schedulers = ['lucid-fixed', 'lucid-continue', 'lucid-nogpu', 'lucid-alwaysgpu', 'lucid']
scales = [1]
result_dir = "./log_search_test/vc_node_factor_0.8"
dataset = "Philly"
get_scale_down_results(result_dir, dataset, info=schedulers, scales=scales)
    
