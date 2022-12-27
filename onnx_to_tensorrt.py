#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import tensorrt as trt

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_file_path, engine_file_path, input_image_WH):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(
        network, TRT_LOGGER
    ) as parser, trt.Runtime(
        TRT_LOGGER
    ) as runtime:
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print(
                "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
            )
            exit(0)
        print("Loading ONNX file from path {}...".format(onnx_file_path))
        with open(onnx_file_path, "rb") as model:
            print("Beginning ONNX file parsing")
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
        network.get_input(0).shape = [1, 3, input_image_WH[0], input_image_WH[1]]
        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine


def main(onnx_file_path, engine_file_path, input_shape):
    """Create a TensorRT engine for ONNX-based YOLOv3-608 and run inference."""
    build_engine(onnx_file_path, engine_file_path, input_shape)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('onnx_file_path')
    parser.add_argument('engine_file_path')
    parser.add_argument('--input_image_path', default='./pictures/zidane.jpg')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image size the yolo was trained at')

    args = parser.parse_args()
    main(args.onnx_file_path, args.engine_file_path, args.img_size)
