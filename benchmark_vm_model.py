import os
import argparse

import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.contrib.graph_executor as runtime

from utils import get_network, make_network_key
from tvm.runtime.vm import VirtualMachine

import time

def benchmark(network, batch_size, dtype, target, log_file, repeat):
    if "mask_rcnn" in network:
        layout = "NCHW"
    else:
        layout = "NHWC"
    mod, params, input_name, input_shape, output_shape = get_network(
        network, batch_size, dtype, layout
    )

    assert os.path.exists(log_file), "The log file '%s' does not exist." % log_file
    print("Use log file %s" % log_file)

    # batch_size_ = batch_size
    # # if isany:
    # batch_size_ = relay.Any()
    # mod, params, data_shape, out_shape = get_net(batch_size_)
    exe =  relay.vm.compile(mod, target="llvm", params=params)
    ctx = tvm.device(str(target), 0)
    vm = VirtualMachine(exe, ctx)
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    input_list = [data_tvm]
    results = vm.invoke("main", input_list)
    start_time = time.time()
    time_lst = []
    for i in range(1200):
        if i==200:
            tic = time.time()
        vm.invoke("main", input_list)
    with_fuse_fps = 1000 * batch_size / (time.time() - tic)
    print(with_fuse_fps)
        #end_time = time.time()
        #tvm_time = end_time - start_time
        #start_time = time.time()
        #time_lst.append(tvm_time)
    #return time_lst

    # with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
    #     vm_exec = relay.vm.compile(mod, target=target, params=params)
    # ctx = tvm.device(str(target), 0)
    # vm = VirtualMachine(vm_exec, ctx)
    # # vm.init(ctx)
    # inputs = {}
    # data = np.random.uniform(size=input_shape)
    # for e, i in zip(input_name, data):
    #     inputs[e] = i
    # # print(inputs)
    # result = vm.invoke("main", **inputs)
    # return np.array(result)

    # if network in ["bert"]:
    #     # Build module
    #     with auto_scheduler.ApplyHistoryBest(log_file):
    #         with tvm.transform.PassContext(
    #             opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    #         ):
    #             lib = relay.build(mod, target=target, params=params)

    #     ctx = tvm.device(str(target), 0)
    #     module = runtime.GraphModule(lib["default"](ctx))

    #     # Feed input data
    #     seq_length = input_shape[0][1]
    #     data = np.random.uniform(size=input_shape[0])
    #     token_types = np.random.uniform(size=input_shape[1])
    #     valid_length = np.array([seq_length] * batch_size)
    #     module.set_input(data0=data, data1=token_types, data2=valid_length)
    # else:
    #     # Build module
    #     with auto_scheduler.ApplyHistoryBest(log_file):
    #         with tvm.transform.PassContext(
    #             opt_level=3, config={"relay.backend.use_auto_scheduler": True}
    #         ):
    #             lib = relay.build(mod, target=target, params=params)#error occurs
    #     ctx = tvm.device(str(target), 0)
    #     module = runtime.GraphModule(lib["default"](ctx))

    #     # Feed input data
    #     data = np.random.uniform(size=input_shape)
    #     module.set_input(input_name, data)

    # # Evaluate
    # ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    # return np.array(ftimer().results)


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
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -model=platinum-8124m -mcpu=skylake-avx512",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument(
        "--logdir", type=str, default="/home2/tvm/TLCBench/saved_logs/2020_01_11", help="Log file directory."
    )
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.network == "all":
        networks = ["resnet_v1_torch", "inception_v3_torch"]
    else:
        networks = [args.network]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    # Benchmark
    result_messages = []
    for network in networks:
        for batch_size in batch_sizes:
            for dtype in dtypes:
                network_key = make_network_key(network, batch_size, dtype)
                print("Benchmark %s ..." % network_key)

                log_file = os.path.join(
                    args.logdir, "autoscheduler", target.model, network_key + ".json"
                )
                prof_res = benchmark(
                    network, batch_size, dtype, target, log_file, args.repeat
                )
    '''

                prof_res *= 1000  # convert to millisecond
                message = "%-18s %-12s %-19s (%s)" % (
                    network,
                    batch_size,
                    "%.2f ms" % np.mean(prof_res),
                    "%.2f ms" % np.std(prof_res),
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
    '''
