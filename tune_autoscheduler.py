import os
import argparse

import tvm
from tvm import relay, auto_scheduler

from utils import get_network, make_network_key

network_to_n_trials = {
    # CPU
    ("resnet_50", 1, "float32", "llvm"): 22000,
    ("resnet_50", 1, "bf16", "llvm"): 22000,
    ("mobilenet_v2", 1, "float32", "llvm"): 16000,
    ("bert", 1, "float32", "llvm"): 12000,
    ("bert", 1, "bf16", "llvm"): 12000,
    ("MLP1", 32, "float32", "llvm"): 2400,
    ("MLP1", 64, "float32", "llvm"): 2400,
    ("MLP1", 128, "float32", "llvm"): 2400,
    ("MLP1", 256, "float32", "llvm"): 2400,
    ("MLP1", 512, "float32", "llvm"): 2400,
    ("MLP2", 32, "float32", "llvm"): 4000,
    ("MLP2", 64, "float32", "llvm"): 4000,
    ("MLP2", 128, "float32", "llvm"): 4000,
    ("MLP2", 256, "float32", "llvm"): 4000,
    ("MLP2", 512, "float32", "llvm"): 4000,
    ("MHA1", 32, "float32", "llvm"): 4000,
    ("MHA1", 64, "float32", "llvm"): 4000,
    ("MHA1", 128, "float32", "llvm"): 4000,
    ("MHA2", 32, "float32", "llvm"): 4000,
    ("MHA2", 64, "float32", "llvm"): 4000,
    ("MHA2", 128, "float32", "llvm"): 4000,
    ("MHA3", 32, "float32", "llvm"): 4000,
    ("MHA3", 64, "float32", "llvm"): 4000,
    ("MHA3", 128, "float32", "llvm"): 4000,
    ("MHA4", 32, "float32", "llvm"): 4000,
    ("MHA4", 64, "float32", "llvm"): 4000,
    ("MHA4", 128, "float32", "llvm"): 4000,

    ("MLP1", 32, "int8", "llvm"): 4000,
    ("MLP1", 64, "int8", "llvm"): 4000,
    ("MLP1", 128, "int8", "llvm"): 4000,
    ("MLP1", 256, "int8", "llvm"): 4000,
    ("MLP1", 512, "int8", "llvm"): 4000,
    ("MLP2", 32, "int8", "llvm"): 7000,
    ("MLP2", 64, "int8", "llvm"): 7000,
    ("MLP2", 128, "int8", "llvm"): 7000,
    ("MLP2", 256, "int8", "llvm"): 7000,
    ("MLP2", 512, "int8", "llvm"): 7000,
    ("MHA1", 32, "int8", "llvm"): 5000,
    ("MHA1", 64, "int8", "llvm"): 5000,
    ("MHA1", 128, "int8", "llvm"): 5000,
    ("MHA2", 32, "int8", "llvm"): 5000,
    ("MHA2", 64, "int8", "llvm"): 5000,
    ("MHA2", 128, "int8", "llvm"): 5000,
    ("MHA3", 32, "int8", "llvm"): 5000,
    ("MHA3", 64, "int8", "llvm"): 5000,
    ("MHA3", 128, "int8", "llvm"): 5000,
    ("MHA4", 32, "int8", "llvm"): 5000,
    ("MHA4", 64, "int8", "llvm"): 5000,
    ("MHA4", 128, "int8", "llvm"): 5000,

    # GPU
    ("resnet_50", 1, "float32", "cuda"): 20000,
    ("mobilenet_v2", 1, "float32", "cuda"): 16000,
    ("bert", 1, "float32", "cuda"): 12000,
}


def auto_scheduler_tune(network, batch_size, dtype, target, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    layout = "NHWC"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )

    if dtype=="bf16":
        mod = relay.transform.ToMixedPrecision("bfloat16")(mod)

    n_trials = network_to_n_trials[(network, batch_size, dtype, str(target.kind))]

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True, timeout=1000),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
    else:
        min_repeat_ms = 450 if network in ["bert"] else 300
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            repeat=1, min_repeat_ms=min_repeat_ms, timeout=10
        )
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print(
            "========== Task %d  (workload key: %s) =========="
            % (idx, task.workload_key)
        )
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(tuning_opt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet_50", "mobilenet_v2", "bert", "MLP1", "MLP2", "MHA1", "MHA2", "MHA3", "MHA4", "all"],
        default="MHA1",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8358 -mcpu=icelake-server",#"llvm -model=platinum-8124m -mcpu=skylake-avx512",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="int8", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs_layers/", help="Log file directory."
    )
    args = parser.parse_args()

    if args.network == "all":
        networks = ["MLP1", "MLP2", "MHA1", "MHA2", "MHA3", "MHA4"]
    else:
        networks = [args.network]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    for network in networks:
        for batch_size in batch_sizes:
            if "MHA" in network and batch_size in [256, 512]:
                continue
            for dtype in dtypes:
                network_key = make_network_key(network, batch_size, dtype)
                print("Tune %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )

                auto_scheduler_tune(network, batch_size, dtype, target, log_file)
