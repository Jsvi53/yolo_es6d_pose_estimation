# -*- coding:utf-8 -*-
import os  # 导入os模块，用于文件操作
import tensorrt as trt  # 导入TensorRT模块
import time
import argparse

parser = (argparse.ArgumentParser())  # 创建一个 ArgumentParser 对象，可以通过该对象添加参数，然后通过 parse_args() 方法解析添加的参数。
parser.add_argument("--simplify", type=bool, default=False, help="simplify onnx")  # 添加参数
opt = parser.parse_args()  # 解析添加的参数


# 假设TRT_LOGGER是一个已经定义好的日志对象
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_version = [8.5]


def convert_tensorrt_engine(
    onnx_fn,  # ONNX模型文件的路径
    trt_fn,  # 输出的TensorRT引擎文件的路径
    max_batch_size,  # 最大批处理大小
    fp16=True,  # 是否启用FP16精度模式
    int8_calibrator=False,  # INT8校准器，如果需要INT8优化则提供
    workspace=2,  # TensorRT构建器的工作空间大小，单位字节
    verbose=True,
):
    # 设置网络创建标志，以显式地指定批处理维度
    network_creation_flag = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # 使用with语句创建TensorRT构建器和网络，并使用ONNX解析器解析ONNX模型
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_creation_flag) as network, builder.create_builder_config() as builder_config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder_config.max_workspace_size = workspace * (1024 * 1024)
        if fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if int8_calibrator:
            builder_config.set_flag(trt.BuilderFlag.INT8)
        if verbose:
            builder_config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        # speed up the engine build for trt major version >= 8
        # 1. disable cudnn tactic
        # 2. load global timing cache
        if trt_version[0] >= 8:
            tactic_source = builder_config.get_tactic_sources() & ~(1 << int(trt.TacticSource.CUDNN))
            builder_config.set_tactic_sources(tactic_source)

        # 打开ONNX文件并解析
        with open(onnx_fn, "rb") as f:
            # 如果解析失败，打印错误信息
            if not parser.parse(f.read()):
                print("got {} errors: ".format(parser.num_errors))
                for i in range(parser.num_errors):
                    e = parser.get_error(i)
                    print(e.code(), e.desc(), e.node())
                return
            else:
                print("parse successful")  # 解析成功

        # 打印网络输入信息
        print("inputs: ", network.num_inputs)
        # inputs = [network.get_input(i) for i in range(network.num_inputs)]
        # opt_profiles = create_optimization_profiles(builder, inputs)
        # add_profiles(config, inputs, opt_profiles)

        # 打印每个输入的详细信息
        for i in range(network.num_inputs):
            print(i, network.get_input(i).name, network.get_input(i).shape)

        # 打印网络输出信息
        print("outputs: ", network.num_outputs)
        # 打印每个输出的详细信息
        for i in range(network.num_outputs):
            output = network.get_output(i)
            print(i, output.name, output.shape)

        build_start_time = time.time()
        engine = builder.build_serialized_network(network, builder_config)
        build_time_elapsed = time.time() - build_start_time
        TRT_LOGGER.log(
            TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed)
        )

        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(trt_fn))
        with open(trt_fn, "wb") as fout:
            fout.write(engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")


def main():
    global opt
    onnx_model_path = "models/es6d/es6d_simplified.onnx"
    trt_engine_path = "models/es6d/es6d_simplified.engine"
    batch_size = 1
    use_fp16 = True
    calibrator = None
    workspace_size = 2
    convert_tensorrt_engine(
        onnx_model_path,
        trt_engine_path,
        batch_size,
        use_fp16,
        calibrator,
        workspace_size,
        verbose=True,
    )


if __name__ == "__main__":
    main()
