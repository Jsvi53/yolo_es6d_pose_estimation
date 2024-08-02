import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import os


def load_engine(engine_file_path, TRT_LOGGER=trt.Logger()):
    assert os.path.exists(engine_file_path)
    # print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# inference pipeline，即 inference 基本流程


def infer(engine, all_data):
    # 创建 execution context 对象，并初始化各种信息
    with engine.create_execution_context() as context:
        # 存储所有内存和数据的字典
        mem_dict = {}
        # 为每个绑定分配内存
        bindings = []
        for binding in engine:
            binding_index = engine.get_binding_index(binding)
            binding_shape = engine.get_tensor_shape(binding)
            size = trt.volume(context.get_binding_shape(binding_index))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                data = np.ascontiguousarray(all_data[binding_index])
                mem_dict[f"input_buffer{binding_index}"] = data
                mem = cuda.mem_alloc(data.nbytes)  # type: ignore
                mem_dict[f"input_memory{binding_index}"] = mem
                bindings.append(int(mem))
                # print(f"Input {binding_index}: shape={data.shape}, dtype={data.dtype}")
            else:
                mem_dict[f"output_buffer{binding_index}"] = cuda.pagelocked_empty(size, dtype)
                mem = cuda.mem_alloc(mem_dict[f"output_buffer{binding_index}"].nbytes)
                mem_dict[f"output_memory{binding_index}"] = mem
                bindings.append(int(mem))
                # print(f"Output {binding_index}: shape={mem_dict[f'output_buffer{binding_index}'].shape}, dtype={dtype}")

        # 创建CUDA流
        stream = cuda.Stream()

        # 复制输入数据到设备内存
        for binding_index in range(engine.num_bindings):
            if engine.binding_is_input(engine[binding_index]):
                cuda.memcpy_htod_async(mem_dict[f"input_memory{binding_index}"], mem_dict[f"input_buffer{binding_index}"], stream)

        # 执行推理
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # 复制输出数据到主机内存
        for binding_index in range(engine.num_bindings):
            if not engine.binding_is_input(engine[binding_index]):
                cuda.memcpy_dtoh_async(mem_dict[f"output_buffer{binding_index}"], mem_dict[f"output_memory{binding_index}"], stream)

        # 同步流
        stream.synchronize()

        # 收集输出数据
        outputs = [mem_dict[f"output_buffer{binding_index}"] for binding_index in range(engine.num_bindings) if not engine.binding_is_input(engine[binding_index])]
        # for i, output in enumerate(outputs):
        #     print(f"Output {i}: {output}")

        return outputs
