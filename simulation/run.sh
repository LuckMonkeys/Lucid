#!/bin/bash

# python simulator.py -e='Venus_Sept' -t='./data/Venus' --sweep
# 
# 

# python simulator.py -s lucid-node-scale --node_scaling_num=0
# python simulator.py -s lucid-node-scale --node_scaling_num=1
# python simulator.py -s lucid-node-scale --node_scaling_num=2
# python simulator.py -s lucid-node-scale --node_scaling_num=3
# 

# vc_nodes_factor=(0.5 1.5 2.0 2.5 3.0)
# vc_nodes_factor=(1.5 2.0 2.5 3.0)
# vc_nodes_factor=(10.0 20.0 30.0)
# vc_nodes_factor=(0.8 0.9 1 1.2) #  1.5 2.0 )
# vc_nodes_factor=(1.0 1.2 1.5 2.0)

# for factor in "${vc_nodes_factor[@]}"
# do
#     # python simulator.py -s lucid --vc_nodes_factor=$factor --learning_method='perfect'& 
#     python simulator.py -s lucid-fixed --vc_nodes_factor=$factor & 
#     python simulator.py -s lucid-continue --vc_nodes_factor=$factor & 
#     python simulator.py -s  lucid-nogpu --vc_nodes_factor=$factor --learning_method='fixed' & 
#     python simulator.py -s  lucid-alwaysgpu --vc_nodes_factor=$factor --learning_method='perfect'& 
#     # 
#     wait 
# done


## search with scaled trance and cluster
#1. generate config yaml
python search_config_generate.py
 
#2. run search with config yaml
max_config_idx=3
for i in $(seq 0 $max_config_idx)
do
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Venus_Sept' --trace-dir="./data/Venus" --log-dir="./log_search" --search_config="./data/Venus/config/$i.yaml"
done
