from collections import OrderedDict, namedtuple

import cv2
import numpy as np
import tensorrt as trt
import torch
from matplotlib import pyplot as plt


class TRT_engine():
    def __init__(self, engine, image_size) -> None:
        """
        :param engine: serialized TensorRT engine
        :param image_size: the size to rescale to for input
        """
        self.imgsz = image_size
        self.weight = engine
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False

        # Converts a TensorRT datatype to the equivalent numpy type.
        def nptype(trt_type):
            """
            Returns the numpy-equivalent of a TensorRT :class:`DataType` .

            :arg trt_type: The TensorRT data type to convert.

            :returns: The equivalent numpy type.
            """

            mapping = {
                trt.float32: np.float32,
                trt.float16: np.float16,
                trt.int8: np.int8,
                trt.int32: np.int32,
                trt.bool: bool,
                trt.uint8: np.uint8,
            }
            if trt_type in mapping:
                return mapping[trt_type]
            raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")

        for index in range(self.model.num_bindings):
            self.name = self.model.get_tensor_name(index)
            self.dtype = nptype(self.model.get_tensor_dtype(self.name))
            self.shape = tuple(self.model.get_tensor_shape(self.name))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data,
                                                    int(self.data.data_ptr()))
            if self.model.get_tensor_mode(self.name) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self, im, color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img, self.r, self.dw, self.dh

    def preprocess(self, image):
        self.img, self.r, self.dw, self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img, 0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict_tensor(self, img, threshold):
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores = self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        num = int(nums[0])
        new_bboxes = []
        for i in range(num):
            if (scores[i] < threshold):
                continue
            xmin = (boxes[i][0] - self.dw) / self.r
            ymin = (boxes[i][1] - self.dh) / self.r
            xmax = (boxes[i][2] - self.dw) / self.r
            ymax = (boxes[i][3] - self.dh) / self.r
            new_bboxes.append([classes[i], scores[i], xmin, ymin, xmax, ymax])
        return new_bboxes

    def predict(self, img, threshold):
        img = self.preprocess(img)
        return self.predict_tensor(img, threshold)


class TRT_engine_8_4_1_5():
    def __init__(self, weight, img_size, debug=False) -> None:
        self.imgsz = img_size
        self.weight = weight
        self.debug = debug
        self.device = torch.device('cuda:0')
        self.init_engine()

    def init_engine(self):
        # Infer TensorRT Engine
        self.Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(self.weight, 'rb') as self.f, trt.Runtime(self.logger) as self.runtime:
            self.model = self.runtime.deserialize_cuda_engine(self.f.read())
        self.bindings = OrderedDict()
        self.fp16 = False

        def nptype(trt_type):
            """
            Returns the numpy-equivalent of a TensorRT :class:`DataType` .

            :arg trt_type: The TensorRT data type to convert.

            :returns: The equivalent numpy type.
            """

            mapping = {
                trt.float32: np.float32,
                trt.float16: np.float16,
                trt.int8: np.int8,
                trt.int32: np.int32,
                trt.bool: bool,
                trt.uint8: np.uint8,
            }
            if trt_type in mapping:
                return mapping[trt_type]
            raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")

        for index in range(self.model.num_bindings):
            self.name = self.model.get_binding_name(index)
            self.dtype = nptype(self.model.get_binding_dtype(index))
            self.shape = tuple(self.model.get_binding_shape(index))
            self.data = torch.from_numpy(np.empty(self.shape, dtype=np.dtype(self.dtype))).to(self.device)
            self.bindings[self.name] = self.Binding(self.name, self.dtype, self.shape, self.data,
                                                    int(self.data.data_ptr()))
            if self.model.binding_is_input(index) and self.dtype == np.float16:
                self.fp16 = True
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.model.create_execution_context()

    def letterbox(self, im, color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        new_shape = self.imgsz
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # Scale ratio (new / old)
        self.r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            self.r = min(self.r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * self.r)), int(round(shape[0] * self.r))
        self.dw, self.dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            self.dw, self.dh = np.mod(self.dw, stride), np.mod(self.dh, stride)  # wh padding
        self.dw /= 2  # divide padding into 2 sides
        self.dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        self.img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return self.img, self.r, self.dw, self.dh

    def preprocess(self, image):
        self.img, self.r, self.dw, self.dh = self.letterbox(image)
        self.img = self.img.transpose((2, 0, 1))
        self.img = np.expand_dims(self.img, 0)
        self.img = np.ascontiguousarray(self.img)
        self.img = torch.from_numpy(self.img).to(self.device)
        self.img = self.img.float()
        return self.img

    def predict(self, img, threshold):
        img = self.preprocess(img)
        self.binding_addrs['images'] = int(img.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data[0].tolist()
        boxes = self.bindings['det_boxes'].data[0].tolist()
        scores = self.bindings['det_scores'].data[0].tolist()
        classes = self.bindings['det_classes'].data[0].tolist()
        num = int(nums[0])
        new_bboxes = []
        for i in range(num):
            if (scores[i] < threshold):
                continue
            xmin = max((boxes[i][0] - self.dw) / self.r, 0.)
            ymin = max((boxes[i][1] - self.dh) / self.r, 0.)
            xmax = max((boxes[i][2] - self.dw) / self.r, 0.)
            ymax = max((boxes[i][3] - self.dh) / self.r, 0.)
            new_bboxes.append([classes[i], scores[i], xmin, ymin, xmax, ymax])
        return new_bboxes


def visualize(img, bbox_array):
    for temp in bbox_array:
        xmin = int(temp[2])
        ymin = int(temp[3])
        xmax = int(temp[4])
        ymax = int(temp[5])
        clas = int(temp[0])
        score = temp[1]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (105, 237, 249), 2)
        img = cv2.putText(img, "class:" + str(clas) + " " + str(round(score, 2)), (xmin, int(ymin) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (105, 237, 249), 1)
    return img


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('engine')
    parser.add_argument('image')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='the image size the yolo was trained at')
    args = parser.parse_args()

    trt_engine = TRT_engine(args.engine, args.img_size)
    img = cv2.imread(args.image)

    results = trt_engine.predict(img, threshold=0.5)
    img = visualize(img, results)
    cv2.imshow("img", img)
    cv2.waitKey(0)
