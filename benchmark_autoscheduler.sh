#!/bin/bash -x

record_root="tuning_records/2021-11-17"
target='llvm -mcpu=cascadelake -model=platinum-8280'
dtype='float32'

network_list=('ResNet50_v1b')
cores_list=('4')    # multi-instances should be the last one
batch_list=('1')

repeat=1
physical_cores=28

network='ResNet50_v1b'
for i in $(seq 1 ${#cores_list[@]}); do
    num_cores=${cores_list[$i-1]}
    batch_size=${batch_list[$i-1]}

    export OMP_NUM_THREADS=${num_cores}
    export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
    
    num_groups=$[${physical_cores}/${num_cores}]
    for j in $(seq 0 $[${num_groups}-1]); do
        start_core=$[${j}*${num_cores}]
        end_core=$[$[${j}+1]*${num_cores}-1]
        record_dir="${record_root}/${network}_${num_cores}cores_bs${batch_size}/"
        benchmark_log="${record_root}/benchmark_autoscheduler_${network}_${num_cores}cores_bs${batch_size}_cores${start_core}-${end_core}.log"
        printf "=%.0s" {1..100}; echo
        echo "benchmarking autoscheduler using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
        echo "using tuning records at ${record_dir}"
        echo "saving logs to ${benchmark_log}"; echo
        
        if [ ${num_groups} == 1 ]
        then
            numactl --physcpubind=${start_core}-${end_core} --membind=0 \
            python benchmark_autoscheduler.py \
                --network=${network} \
                --batch-size=${batch_size} \
                --target="${target}" \
                --dtype=${dtype} \
                --logdir=${record_dir} \
                --repeat=${repeat} \
                | tee ${benchmark_log} 2>&1
            echo "done benchmarking autoscheduler using ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: ${start_core}-${end_core}"
        else
            numactl --physcpubind=${start_core}-${end_core} --membind=0 \
            python benchmark_autoscheduler.py \
                --network=${network} \
                --batch-size=${batch_size} \
                --target="${target}" \
                --dtype=${dtype} \
                --logdir=${record_dir} \
                --repeat=${repeat} \
                | tee ${benchmark_log} 2>&1 &
        fi
    done
done
echo "benchmarking sessions lanched, please wait for the python runs."
