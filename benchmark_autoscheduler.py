import os
import argparse

import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.debugger.debug_executor import GraphModuleDebug
import time

from utils import get_network, make_network_key

def benchmark(network, batch_size, dtype, target, log_file, repeat, profiling, inC, outC):
    layout = "NHWC"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout, inC, outC
    )
    if dtype == "int8":
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    assert os.path.exists(log_file), "The log file '%s' does not exist." % log_file
    print("Use log file %s" % log_file)

    # Build module
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True},
        ):
            lib = tvm.relay.build(mod, target=target, params=params)
    ctx = tvm.device(str(target), 0)
    if not profiling:
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
    else:
        module = GraphModuleDebug(
            lib["debug_create"]("default", ctx),
            [ctx],
            lib.graph_json,
            dump_root="./tvmdbg",
        )
    
    # Feed input data
    data = np.random.uniform(size=input_shape)
    module.set_input(input_name, data)        
    module.set_input(**params)

    if profiling:
        # execute
        for _ in range(repeat):
            module.run()
    else:
        # Evaluate
        ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=3000, repeat=repeat)

    if profiling:
        return np.array(0)
    else:
        return np.array(ftimer().results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["MLP1", "MLP2", "MHA1", "MHA2", "MHA3", "MHA4", "single_matmul", "all"],
        default="single_matmul",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="The batch size")
    parser.add_argument("--inC", type=int, default=13, help="The batch size")
    parser.add_argument("--outC", type=int, default=512, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8358 -mcpu=icelake-server",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs_layers/", help="Log file directory."
    )
    parser.add_argument(
        "--profiling", type=bool, default=False, help="Whether to use profiling."
    )
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.network == "all":
        networks = ["MLP1", "MLP2", "MHA1", "MHA2", "MHA3", "MHA4"]
    else:
        networks = [args.network]
    if args.batch_size == -1:
        batch_sizes = [32, 64, 128, 256, 512]
    else:
        batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    # Benchmark
    result_messages = []
    for network in networks:
      for batch_size in batch_sizes:
         if "MHA" in network and batch_size in [256, 512]:
             continue
         for dtype in dtypes:
            matmul_shapes = ((args.inC, args.outC),)
            for inC, outC in matmul_shapes:
                network_key = make_network_key(network, batch_size, dtype, inC, outC)
                print("Benchmark %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )
                prof_res = benchmark(
                    network, batch_size, dtype, target, log_file, args.repeat, args.profiling, inC, outC
                )
                prof_res *= 1000  # convert to millisecond
                message = "%-18s %-12s %-19s (%s)" % (
                    network_key,
                    batch_size,
                    "%.3f ms" % np.mean(prof_res),
                    "%.3f ms" % np.std(prof_res),
                )
                result_messages.append(message)

    # Print result
    print("-------------------------------------------------------------")
    print(
        "%-18s %-12s %-20s"
        % ("Network Name", "Batch size", "Mean Inference Time (std dev)")
    )
    print("-------------------------------------------------------------")
    for line in result_messages:
        print(line)
    print("-------------------------------------------------------------")

