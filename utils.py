import os

import tvm
from tvm import relay


def get_network(name, batch_size, dtype, layout):
    """Get the symbol definition and random weight of a network"""
    input_name = "data"
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        import mxnet

        n_layer = int(name.split("_")[1])
        block = mxnet.gluon.model_zoo.vision.get_resnet(1, n_layer, pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "ResNet50_v1b":
        import gluoncv
        block = gluoncv.model_zoo.get_model("ResNet50_v1b", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
           mod = convert_to_nhwc(mod)
        else:
           assert layout == "NCHW"
    elif name == "InceptionV3":
        import gluoncv
        input_shape = (batch_size, 3, 300, 300)
        block = gluoncv.model_zoo.get_model("InceptionV3", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )

        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "VGG11_bn":
        import gluoncv
        block = gluoncv.model_zoo.get_model("VGG11_bn", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "DenseNet121":
        import gluoncv
        block = gluoncv.model_zoo.get_model("DenseNet121", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
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
    elif "faster_rcnn" in name:
        import torch
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model = model.eval()
        x = [torch.rand(3, 800, 800)]
        predictions = model(x)

        # We grab the TorchScripted model via tracing
        input_shape = [1, 3, 800, 800]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        # import numpy as np
        # in_size = 800

        # input_shape = (1, 3, in_size, in_size)


        # def do_trace(model, inp):
        #     model_trace = torch.jit.trace(model, inp)
        #     model_trace.eval()
        #     return model_trace


        # def dict_to_tuple(out_dict):
        #     if "masks" in out_dict.keys():
        #         return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
        #     return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


        # class TraceWrapper(torch.nn.Module):
        #     def __init__(self, model):
        #         super().__init__()
        #         self.model = model

        #     def forward(self, inp):
        #         out = self.model(inp)
        #         return dict_to_tuple(out[0])


        # model_func = torchvision.models.detection.fasterrcnn_resnet50_fpn
        # model = TraceWrapper(model_func(pretrained=True))

        # model.eval()
        # inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

        # with torch.no_grad():
        #     # out = model(inp)
        #     script_module = do_trace(model, inp)
        
        
        # input_name = "input0"
        # shape_list = [(input_name, input_shape)]
        # mod, params = relay.frontend.from_pytorch(script_module, shape_list)

        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    
    elif "mask_rcnn" in name:
        import torch
        import torchvision
        import numpy as np
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = model.eval()
        x = [torch.rand(3, 800, 800)]
        predictions = model(x)

        # We grab the TorchScripted model via tracing
        input_shape = [1, 3, 800, 800]
        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()

        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        # in_size = 800

        # input_shape = (1, 3, in_size, in_size)


        # def do_trace(model, inp):
        #     model_trace = torch.jit.trace(model, inp)
        #     model_trace.eval()
        #     return model_trace


        # def dict_to_tuple(out_dict):
        #     if "masks" in out_dict.keys():
        #         return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
        #     return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


        # class TraceWrapper(torch.nn.Module):
        #     def __init__(self, model):
        #         super().__init__()
        #         self.model = model

        #     def forward(self, inp):
        #         out = self.model(inp)
        #         return dict_to_tuple(out[0])


        # model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
        # model = TraceWrapper(model_func(pretrained=True))

        # model.eval()
        # inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

        # with torch.no_grad():
        #     out = model(inp)
        #     script_module = do_trace(model, inp)
        
        
        # input_name = "input0"
        # shape_list = [(input_name, input_shape)]
        # mod, params = relay.frontend.from_pytorch(script_module, shape_list)

        # target = "llvm"

        # with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
        #     vm_exec = relay.vm.compile(mod, target=target, params=params)

        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
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
