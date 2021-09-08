from datetime import time
import os
import argparse

import tvm
from tvm import relay, auto_scheduler

from utils import get_network, make_network_key

network_to_n_trials = {
    # CPU
    ("resnet_50", 128, "float32", "llvm"): 22000,
    ("resnet_v1_torch", 1, "float32", "llvm"): 22000,
    ("inception_v3_torch", 1, "float32", "llvm"): 44000,##task 54
    ("faster_rcnn_torch", 1, "float32", "llvm"): 44000,## task 51
    ("mask_rcnn_torch", 1, "float32", "llvm"): 44000,##54
    ("mobilenet_v2", 1, "float32", "llvm"): 16000,
    ("bert", 64, "float32", "llvm"): 12000,
    # GPU
    ("resnet_50", 1, "float32", "cuda"): 20000,
    ("mobilenet_v2", 1, "float32", "cuda"): 16000,
    ("bert", 1, "float32", "cuda"): 12000,
}


def auto_scheduler_tune(network, batch_size, dtype, target, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    if "mask_rcnn" in network:
        layout = "NCHW"
    else:
        layout = "NHWC"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )

    n_trials = network_to_n_trials[(network, batch_size, dtype, str(target.kind))]

    if "cpu" in target.keys:
        tuning_opt = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
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
        choices=["resnet_50", "mobilenet_v2", "bert", "resnet_v1_torch", "inception_v3_torch", "faster_rcnn_torch",\
             "mask_rcnn_torch", "all"],
        default="resnet_50",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8124m -mcpu=skylake-avx512",#llvm -model=platinum-8124m -mcpu=skylake-avx512
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="experiment_res/mxnet_resnet50_v1_bs128_28core", help="Log file directory."
    )
    args = parser.parse_args()

    if args.network == "all":
        networks = ["faster_rcnn_torch", "mask_rcnn_torch"]
    else:
        networks = [args.network]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    import datetime
    time_lst = []
    for network in networks:
        start = datetime.datetime.now()
        print(start)
        for batch_size in batch_sizes:
            for dtype in dtypes:
                network_key = make_network_key(network, batch_size, dtype)
                print("Tune %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )

                auto_scheduler_tune(network, batch_size, dtype, target, log_file)
        end = datetime.datetime.now()
        print(end)
        print(end-start)
        time_lst.append(end-start)
    print(time_lst)
