import os

import tvm
from tvm import relay
x_scale = y_scale = z_scale = 0.00784314
x_zero_point = y_zero_point = z_zero_point = 127

x_zp = relay.const(x_zero_point, "int32")
y_zp = relay.const(y_zero_point, "int32")
x_s = relay.const(x_scale, "float32")
y_s = relay.const(y_scale, "float32")


def get_network(name, batch_size, dtype, layout):
    """Get the symbol definition and random weight of a network"""
    input_name = "data"
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        import mxnet
        import gluoncv
        n_layer = int(name.split("_")[1])
        block = gluoncv.model_zoo.get_model("ResNet50_v1b", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype="float32"
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
        
    elif name == "mobilenet_v2":
        import mxnet

        multiplier = 1
        block = mxnet.gluon.model_zoo.vision.get_mobilenet_v2(
            multiplier, pretrained=True
        )
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "bert":
        import gluonnlp

        seq_length = 128

        # Instantiate a BERT classifier using GluonNLP
        model_name = "bert_12_768_12"
        dataset = "book_corpus_wiki_en_uncased"
        model, _ = gluonnlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False,
        )

        # Convert the MXNet model into TVM Relay format
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size,),
        }
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        print(mod)
        input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
                fn, params
            ),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    elif name == "MLP1":
        import numpy as np
        input_shape = (batch_size, 13)
        output_shape = (batch_size, 128)
        kernel1_shape = (13, 512)
        kernel2_shape = (512, 256)
        kernel3_shape = (256, 128)

        x = relay.var("data", shape=input_shape, dtype="float32")
        kernel1 = relay.var("kernel1", shape=kernel1_shape, dtype="float32")
        kernel2 = relay.var("kernel2", shape=kernel2_shape, dtype="float32")
        kernel3 = relay.var("kernel3", shape=kernel3_shape, dtype="float32")

        if dtype=="int8":
            x = relay.qnn.op.quantize(x, relay.const(0.078), relay.const(0), out_dtype="uint8")
            kernel1 = relay.qnn.op.quantize(kernel1, relay.const(0.07), relay.const(0), out_dtype="int8")
            kernel2 = relay.qnn.op.quantize(kernel2, relay.const(0.07), relay.const(0), out_dtype="int8")
            kernel3 = relay.qnn.op.quantize(kernel3, relay.const(0.07), relay.const(0), out_dtype="int8")
        if dtype=="float32":
            out = relay.nn.matmul(x, kernel1)
            out = relay.nn.relu(out)
            out = relay.nn.matmul(out, kernel2)
            out = relay.nn.relu(out)
            out = relay.nn.matmul(out, kernel3)
            out = relay.nn.relu(out)
            mod = tvm.IRModule.from_expr(out)
        else:
            out = relay.qnn.op.dense(
                x, relay.transpose(kernel1, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            out = relay.qnn.op.dense(
                out, relay.transpose(kernel2, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            out = relay.qnn.op.dense(
                out, relay.transpose(kernel3, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            mod = tvm.IRModule.from_expr(out)
            mod = relay.transform.InferType()(mod)
            mod = relay.qnn.transform.CanonicalizeOps()(mod)
        print(mod)
        input_shapes = {"data": input_shape, "kernel1": kernel1_shape, "kernel2": kernel2_shape, "kernel3": kernel3_shape}
        param_lst = ["kernel1", "kernel2", "kernel3"]
        params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype("float32") for x in param_lst}

    elif name == "MLP2":
        import numpy as np
        input_shape = (batch_size, 479)
        output_shape = (batch_size, 1)
        kernel1_shape = (479, 1024)
        kernel2_shape = (1024, 1024)
        kernel3_shape = (1024, 512)
        kernel4_shape = (512, 256)
        kernel5_shape = (256, 1)
        x = relay.var("data", shape=input_shape, dtype="float32")
        kernel1 = relay.var("kernel1", shape=kernel1_shape, dtype="float32")
        kernel2 = relay.var("kernel2", shape=kernel2_shape, dtype="float32")
        kernel3 = relay.var("kernel3", shape=kernel3_shape, dtype="float32")
        kernel4 = relay.var("kernel4", shape=kernel4_shape, dtype="float32")
        kernel5 = relay.var("kernel5", shape=kernel5_shape, dtype="float32")
        if dtype=="int8":
            x = relay.qnn.op.quantize(x, relay.const(0.078), relay.const(0), out_dtype="uint8")
            kernel1 = relay.qnn.op.quantize(kernel1, relay.const(0.07), relay.const(0), out_dtype="int8")
            kernel2 = relay.qnn.op.quantize(kernel2, relay.const(0.07), relay.const(0), out_dtype="int8")
            kernel3 = relay.qnn.op.quantize(kernel3, relay.const(0.07), relay.const(0), out_dtype="int8")
            kernel4 = relay.qnn.op.quantize(kernel4, relay.const(0.07), relay.const(0), out_dtype="int8")
            kernel5 = relay.qnn.op.quantize(kernel5, relay.const(0.07), relay.const(0), out_dtype="int8")
        if dtype=="float32":
            out = relay.nn.matmul(x, kernel1)
            out = relay.nn.relu(out)
            out = relay.nn.matmul(out, kernel2)
            out = relay.nn.relu(out)
            out = relay.nn.matmul(out, kernel3)
            out = relay.nn.relu(out)
            out = relay.nn.matmul(out, kernel4)
            out = relay.nn.relu(out)
            out = relay.nn.matmul(out, kernel5)
            out = relay.nn.relu(out)
            mod = tvm.IRModule.from_expr(out)
        else:
            out = relay.qnn.op.dense(
                x, relay.transpose(kernel1, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            out = relay.qnn.op.dense(
                out, relay.transpose(kernel2, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            out = relay.qnn.op.dense(
                out, relay.transpose(kernel3, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            out = relay.qnn.op.dense(
                out, relay.transpose(kernel4, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            out = relay.qnn.op.dense(
                out, relay.transpose(kernel5, axes=[1, 0]),
                x_zp, y_zp, x_s, y_s,
                None,
                out_dtype="int32"
            )
            out = relay.qnn.op.requantize(out, x_s, x_zp, y_s, y_zp, out_dtype="int8")
            out = relay.nn.relu(out)
            mod = tvm.IRModule.from_expr(out)
            mod = relay.transform.InferType()(mod)
            mod = relay.qnn.transform.CanonicalizeOps()(mod)
        print(mod)
        input_shapes = {"data": input_shape, "kernel1": kernel1_shape, "kernel2": kernel2_shape, "kernel3": kernel3_shape,
                        "kernel4": kernel4_shape, "kernel5": kernel5_shape}
        param_lst = ["kernel1", "kernel2", "kernel3", "kernel4", "kernel5"]
        params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype("float32") for x in param_lst}
    elif "MHA" in name:
        import numpy as np
        seq_len, total_dim, num_head = 128, 768, 8
        if name == "MHA2":
            seq_len, total_dim, num_head = 128, 768, 12
        elif name == "MHA3":
            seq_len, total_dim, num_head = 384, 1024, 8
        elif name == "MHA4":
            seq_len, total_dim, num_head = 512, 1024, 16
        head_dim = int(total_dim / num_head)
        input_shape = (batch_size, num_head, seq_len, head_dim)
        output_shape = (batch_size, seq_len, num_head, head_dim)
        matmul1_kernel_shape = (batch_size * num_head, head_dim, seq_len)
        fscore_shape = (1,)
        add_shape = (batch_size, 1, 1, seq_len)
        matmul2_kernel_shape = (batch_size * num_head, seq_len, head_dim)
        x = relay.var("data", shape=(input_shape), dtype=dtype)
        matmul1_kernel = relay.var("matmul1_kernel", shape=(matmul1_kernel_shape), dtype=dtype)
        f_score_div = relay.var("f_score_div", shape=(fscore_shape), dtype=dtype)
        f_score_add = relay.var("f_score_add", shape=(add_shape), dtype=dtype)
        matmul2_kernel = relay.var("matmul2_kernel", shape=(matmul2_kernel_shape), dtype=dtype)
        x = relay.reshape(x, [batch_size *num_head, seq_len, head_dim])
        out = relay.nn.batch_matmul(x, matmul1_kernel, transpose_a=False, transpose_b=False)
        out = relay.divide(out, f_score_div)
        out = relay.reshape(out, [batch_size, num_head, seq_len, seq_len])
        out = relay.add(out, f_score_add)
        out = relay.nn.softmax(out, axis=3)
        out = relay.reshape(out, [batch_size * num_head, seq_len, seq_len])
        out = relay.nn.batch_matmul(out, matmul2_kernel, transpose_a=False, transpose_b=False)
        out = relay.reshape(out, [batch_size, num_head, seq_len, head_dim])
        out = relay.transpose(out, [0, 2, 1, 3])
        mod = tvm.IRModule.from_expr(out)
        print(mod)
        input_shapes = {"data": input_shape, "matmul1_kernel": matmul1_kernel_shape, "f_score_div": fscore_shape, "f_score_add": add_shape,
                        "matmul2_kernel": matmul2_kernel_shape}
        param_lst = ["matmul1_kernel", "f_score_div", "f_score_add", "matmul2_kernel"]
        params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in param_lst}
        params["f_score_div"] = np.array(np.sqrt(seq_len).astype(dtype))
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_name, input_shape, output_shape


def make_network_key(network_name, batch_size, dtype):
    return "%s-B%s-%s" % (network_name, batch_size, dtype)


def use_graph_tuner(network_name, batch_size, dtype, target):
    """Return whether use graph tuner for a network on a target"""
    # Only use graph tuner for CNNs on CPUs
    return "cpu" in target.keys and not (network_name in ["bert"])


def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod
