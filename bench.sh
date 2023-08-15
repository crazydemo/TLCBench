echo "tuning with 32 cores and batchsize=1"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
python benchmark_autoscheduler.py     --network="MLP1"     --batch-size=32     --dtype="float32"     --logdir="tmp_logs_layers"    --profiling=True
