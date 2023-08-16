echo "tuning with 32 cores and batchsize=1"
export OMP_NUM_THREADS=32
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
# FLOAT32
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=32 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=64 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=128 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=256 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=512 --dtype="float32" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=32 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=64 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=128 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=256 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=512 --dtype="float32" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA1" --batch-size=32 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA1" --batch-size=64 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA1" --batch-size=128 --dtype="float32" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA2" --batch-size=32 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA2" --batch-size=64 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA2" --batch-size=128 --dtype="float32" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA3" --batch-size=32 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA3" --batch-size=64 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA3" --batch-size=128 --dtype="float32" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA4" --batch-size=32 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA4" --batch-size=64 --dtype="float32" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA4" --batch-size=128 --dtype="float32" --logdir="tmp_logs_layers"

# INT8
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=32 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=64 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=128 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=256 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP1" --batch-size=512 --dtype="int8" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=32 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=64 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=128 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=256 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MLP2" --batch-size=512 --dtype="int8" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA1" --batch-size=32 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA1" --batch-size=64 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA1" --batch-size=128 --dtype="int8" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA2" --batch-size=32 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA2" --batch-size=64 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA2" --batch-size=128 --dtype="int8" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA3" --batch-size=32 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA3" --batch-size=64 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA3" --batch-size=128 --dtype="int8" --logdir="tmp_logs_layers"

numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA4" --batch-size=32 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA4" --batch-size=64 --dtype="int8" --logdir="tmp_logs_layers"
numactl --physcpubind=0-31 --membind=0 python benchmark_autoscheduler.py --network="MHA4" --batch-size=128 --dtype="int8" --logdir="tmp_logs_layers"
