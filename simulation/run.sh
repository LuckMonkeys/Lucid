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
vc_nodes_factor=(1 1.5 2.0 )
# vc_nodes_factor=(1)

for factor in "${vc_nodes_factor[@]}"
do
    python simulator.py -s lucid --vc_nodes_factor=$factor
    python simulator.py -s  lucid-alwaysgpu --vc_nodes_factor=$factor
done


