#!/usr/bin/env python3
import argparse
import statistics
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def make_model(path: str, bits: int, size: int, block_size: int, accuracy_level: int):
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((size, size), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["a", "weight"], ["result"], name="matmul")],
        "matmul_nbits_benchmark",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [size, size])],
        [helper.make_tensor_value_info("result", TensorProto.FLOAT, [size, size])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
    )
    quantizer = MatMulNBitsQuantizer(
        model,
        bits=bits,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=accuracy_level,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(path)


def benchmark(path: str, size: int, threads: int, iterations: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    a = np.ones((size, size), dtype=np.float32)
    for _ in range(10):
        session.run(None, {"a": a})
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.run(None, {"a": a})
        samples.append((time.perf_counter_ns() - start) / 1e6)
    median_ms = statistics.median(samples)
    gops = 2.0 * size**3 / (median_ms * 1.0e6)
    print(
        f"threads={threads} median={median_ms:.3f} ms "
        f"effective_throughput={gops:.1f} GOPS"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, choices=[2, 4, 8])
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--accuracy-level", type=int, default=0, choices=range(5))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 30])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--model", default="/tmp/ort_matmul_nbits.onnx")
    args = parser.parse_args()
    make_model(
        args.model, args.bits, args.size, args.block_size, args.accuracy_level
    )
    for threads in args.threads:
        benchmark(args.model, args.size, threads, args.iterations)


if __name__ == "__main__":
    main()
