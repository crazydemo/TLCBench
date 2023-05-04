echo "tuning with 32 cores and batchsize=1"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
numactl --physcpubind=0-31 --membind=0 \
python tune_autoscheduler.py \
    --network="MHA4" \
    --batch-size=32 \
    --dtype="int8" \
    --logdir="tmp_logs_layers" \
     | tee "tune_layers_mha4_int8_bs32.log"
     
numactl --physcpubind=0-31 --membind=0 \
python tune_autoscheduler.py \
    --network="MHA4" \
    --batch-size=64 \
    --dtype="int8" \
    --logdir="tmp_logs_layers" \
     | tee "tune_layers_mha4_int8_bs64.log"
     
numactl --physcpubind=0-31 --membind=0 \
python tune_autoscheduler.py \
    --network="MHA4" \
    --batch-size=128 \
    --dtype="int8" \
    --logdir="tmp_logs_layers" \
     | tee "tune_layers_mha4_int8_bs128.log"
