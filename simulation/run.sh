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
# python search_config_generate.py
 
#2. run search with config yaml
max_config_idx=15
# for i in $(seq 0 $max_config_idx)
for ((i=0; i<=max_config_idx; i+=4));
do
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Venus_Sept' --trace-dir="./data/Venus" --log-dir="./log_search_venus" --search_config="./data/Venus/config/$i.yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Venus_Sept' --trace-dir="./data/Venus" --log-dir="./log_search_venus" --search_config="./data/Venus/config/$((i+1)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Venus_Sept' --trace-dir="./data/Venus" --log-dir="./log_search_venus" --search_config="./data/Venus/config/$((i+2)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Venus_Sept' --trace-dir="./data/Venus" --log-dir="./log_search_venus" --search_config="./data/Venus/config/$((i+3)).yaml" --filter_profile_job=True&
    wait
done

for ((i=0; i<=max_config_idx; i+=4));
do
    python simulator.py -s search --vc_nodes_factor=0.8 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$i.yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=0.8 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$((i+1)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=0.8 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$((i+2)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=0.8 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$((i+3)).yaml" --filter_profile_job=True&
    wait
done

for ((i=0; i<=max_config_idx; i+=4));
do
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$i.yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$((i+1)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$((i+2)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='Philly' --trace-dir="./data/Philly" --log-dir="./log_search_philly" --search_config="./data/Philly/config/$((i+3)).yaml" --filter_profile_job=True&
    wait
done

for ((i=0; i<=max_config_idx; i+=4));
do
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='MLaas' --trace-dir="./data/MLaas" --log-dir="./log_search_mlaas" --search_config="./data/MLaas/config/$i.yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='MLaas' --trace-dir="./data/MLaas" --log-dir="./log_search_mlaas" --search_config="./data/MLaas/config/$((i+1)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='MLaas' --trace-dir="./data/MLaas" --log-dir="./log_search_mlaas" --search_config="./data/MLaas/config/$((i+2)).yaml" --filter_profile_job=True &
    python simulator.py -s search --vc_nodes_factor=1.0 --experiment-name='MLaas' --trace-dir="./data/MLaas" --log-dir="./log_search_mlaas" --search_config="./data/MLaas/config/$((i+3)).yaml" --filter_profile_job=True&
    wait
done
