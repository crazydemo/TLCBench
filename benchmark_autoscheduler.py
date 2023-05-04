import os
import argparse

import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_runtime as runtime

from utils import get_network, make_network_key

def benchmark(network, batch_size, dtype, target, log_file, repeat):
    layout = "NHWC"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )
    '''
    if dtype == "int8":
        with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
            mod = relay.quantize.quantize(mod, params)
    '''
    assert os.path.exists(log_file), "The log file '%s' does not exist." % log_file
    print("Use log file %s" % log_file)

    if network in ["bert"]:
        # Build module
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = tvm.relay.build(mod, target=target, params=params)

        ctx = tvm.device(str(target), 0)
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))

        # Feed input data
        seq_length = input_shape[0][1]
        data = np.random.uniform(size=input_shape[0])
        token_types = np.random.uniform(size=input_shape[1])
        valid_length = np.array([seq_length] * batch_size)
        if dtype == "bf16":
            data = np.left_shift(data.astype("uint32"), 16).view("<f4")
            token_types = np.left_shift(token_types.astype("uint32"), 16).view("<f4")
            valid_length = np.left_shift(valid_length.astype("uint32"), 16).view("<f4")
        module.set_input(data0=data, data1=token_types, data2=valid_length)
    else:
        # Build module
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = tvm.relay.build(mod, target=target, params=params)

        ctx = tvm.device(str(target), 0)
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))
        # Feed input data
        data = np.random.uniform(size=input_shape)
        module.set_input(input_name, data)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=3000, repeat=repeat)
    return np.array(ftimer().results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=["resnet_50", "mobilenet_v2", "bert", "MLP1", "MLP2", "MHA1", "MHA2", "MHA3", "MHA4", "all"],
        default="MLP1",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8358 -mcpu=icelake-server",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="int8", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="tmp_logs_layers/", help="Log file directory."
    )
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.network == "all":
        #networks = ["resnet_50", "mobilenet_v2", "bert"]
        networks = ["MHA1", "MHA2", "MHA3", "MHA4"]
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
                network_key = make_network_key(network, batch_size, dtype)
                print("Benchmark %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )
                prof_res = benchmark(
                    network, batch_size, dtype, target, log_file, args.repeat
                )

                prof_res *= 1000  # convert to millisecond
                message = "%-18s %-12s %-19s (%s)" % (
                    network,
                    batch_size,
                    "%.4f ms" % np.mean(prof_res),
                    "%.4f ms" % np.std(prof_res),
                )
                result_messages.append(message)
                print(message)

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
