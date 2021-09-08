# TLCBench

Benchmark scripts for TVM

## Content
- [Requirement](#requirement)
- [Intel CPU](#intel-cpu)
- [NVIDIA GPU](#nvidia-gpu)


## Requirement
Tested with  
TVM commit id: dfe4cebbdadab3d4e6e6ba3951276a51a4ffeaf6 (May. 14, 2021)  
mxnet==1.8.0.post0  
gluonnlp==0.10.0  

### Benchmark
- Commands for AutoScheduler bs=1 / bs=128 28core
```bash
OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 numactl --physcpubind=0-27 --membind=0 python benchmark_autoscheduler.py
```

- Commands for AutoScheduler multi-instance
```bash
./multi_instance.sh
```

### Tuning
- Commands for AutoScheduler
```bash
# Tune one network
OMP_NUM_THREADS=28 KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 numactl --physcpubind=0-27 --membind=0 python python tune_autoscheduler.py --network resnet_50 --target "llvm -mcpu=skylake-avx512 -model=platinum-8124m"
