#!/bin/bash
export OMP_NUM_THREADS=4
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
for ((i=0; i<7; i++))
            do
                startCore=$[${i}*4]
                endCore=$[${startCore}+3]
                numactl --physcpubind=${startCore}-${endCore} --membind=0 python /home2/tvm/TLCBench/benchmark_autoscheduler.py --logdir=/home2/tvm/TLCBench/experiment_res/mxnet_resnet50_v1_bs1_4core& # >> /home2/tvm/TLCBench/experiment_res/0907/native_${i}.txt&
            done

