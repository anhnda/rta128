
import onnxruntime as ort 
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import torch
import time 
import contextlib
import collections
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
import tensorrt as trt

import torch
import torchvision.transforms as T 

# print(onnx.helper.printable_graph(mm.graph))
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-ox', '--onnx_file', type=str, default="model.onnx")
    parser.add_argument('-f', '--im_file', type=str, default="000000000139.jpg")
    parser.add_argument('-d', '--device', type=str, default='cuda:0')

    args = parser.parse_args()

    # Load the original image without resizing

    # Resize the image for model input
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None]
    print(im_data.device)
    sess = ort.InferenceSession(args.onnx_file, providers=["CUDAExecutionProvider"])
    print("Available providers:", ort.get_available_providers())
    print("Session providers:", sess.get_providers())

    n_runs = 2
    times = []

    for _ in range(n_runs):
        start = time.perf_counter()
        sess.run(
            output_names=None,
            input_feed={
                'images': im_data.data.numpy(),
                'orig_target_sizes': orig_size.cpu().data.numpy()
            }
        )
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / n_runs
    print(f"Average inference time over {n_runs} runs: {avg_time * 1000:.2f} ms")
