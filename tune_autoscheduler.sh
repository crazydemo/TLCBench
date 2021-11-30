#!/bin/bash -x

record_root="tuning_records/`date +'%F'`"
target='llvm -mcpu=cascadelake -model=platinum-8280'
dtype='float32'

network_list=('ResNet50_v1b')
cores_list=('28' '4')
batch_list=('1' '1')

mkdir -p ${record_root}
for network in ${network_list[@]}; do
    for i in $(seq 1 ${#cores_list[@]}); do
        num_cores=${cores_list[$i-1]}
        batch_size=${batch_list[$i-1]}

        record_dir="${record_root}/${network}_${num_cores}cores_bs${batch_size}/"
        tuning_log="${record_root}/tune_autoscheduler_${network}_${num_cores}cores_bs${batch_size}.log"
        echo; printf '=%.0s' {1..100}; echo
        echo "tuning ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: 0-$[${num_cores}-1]"
        echo "saving records to ${record_dir}"
        echo "saving logs to ${tuning_log}"

        export OMP_NUM_THREADS=${num_cores}
        export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
        numactl --physcpubind=0-$[${num_cores}-1] --membind=0 \
        python tune_autoscheduler.py \
            --network=${network} \
            --batch-size=${batch_size} \
            --target="${target}" \
            --dtype=${dtype} \
            --logdir=${record_dir} \
            | tee ${tuning_log} 2>&1
        echo "done tuning ${network} with ${num_cores} cores and batchsize=${batch_size} on cores: 0-$[${num_cores}-1]"
    done
done
