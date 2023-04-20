echo "tuning with 32 cores and batchsize=1"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
numactl --physcpubind=0-31 --membind=0 \
python tune_autoscheduler.py \
    --network="MLP1" \
    --batch-size=32 \
    --dtype="int8" \
    --logdir="tmp_logs_layers" \
     | tee "tune_layers.log"
