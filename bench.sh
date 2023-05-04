echo "tuning with 32 cores and batchsize=1"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
numactl --physcpubind=0-31 --membind=0 \
python benchmark_autoscheduler.py \
    --network="MLP2" \
    --batch-size=-1 \
    --dtype="int8" \
    --logdir="tmp_logs_layers"
