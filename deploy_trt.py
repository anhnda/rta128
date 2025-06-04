"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import contextlib
import collections
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw
import tensorrt as trt

import torch
import torchvision.transforms as T 




class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(self, engine_path, device='cuda:0', backend='torch', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    
    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        '''build binddings
        '''
        Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = list(engine.get_tensor_shape(name))  # Convert to list for mutability
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            
            # Handle dynamic shapes
            dynamic = False
            if -1 in shape:
                dynamic = True
                # Replace -1 with max_batch_size for batch dimension (usually first dimension)
                for j, dim in enumerate(shape):
                    if dim == -1:
                        shape[j] = max_batch_size
                
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 

            else:
                # Only create tensor if all dimensions are positive
                if all(dim > 0 for dim in shape):
                    data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                    bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
                else:
                    # For dynamic shapes, create a placeholder that will be updated later
                    bindings[name] = Binding(name, dtype, shape, None, None)

        return bindings

    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            current_shape = list(blob[n].shape)
            
            # Update binding if shape changed or if it was a dynamic placeholder
            if (self.bindings[n].shape != current_shape or 
                self.bindings[n].data is None):
                
                self.context.set_input_shape(n, current_shape)
                
                # Create new tensor with correct shape
                dtype = self.bindings[n].dtype
                data = torch.empty(current_shape, dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype).to(self.device)
                
                # Update binding
                self.bindings[n] = self.bindings[n]._replace(
                    shape=current_shape, 
                    data=data, 
                    ptr=data.data_ptr()
                )
            
            # Copy input data
            self.bindings[n].data.copy_(blob[n])

        # Update output bindings if needed
        for n in self.output_names:
            if self.bindings[n].data is None:
                # Get the actual output shape after setting input shapes
                output_shape = list(self.context.get_tensor_shape(n))
                dtype = self.bindings[n].dtype
                data = torch.empty(output_shape, dtype=torch.from_numpy(np.array([], dtype=dtype)).dtype).to(self.device)
                
                self.bindings[n] = self.bindings[n]._replace(
                    shape=output_shape,
                    data=data,
                    ptr=data.data_ptr()
                )

        # Update bindings addresses
        self.bindings_addr.update({n: self.bindings[n].ptr for n in self.input_names + self.output_names})
        
        # Execute inference
        self.context.execute_v2(list(self.bindings_addr.values()))
        
        outputs = {n: self.bindings[n].data for n in self.output_names}
        return outputs

    def async_run_cuda(self, blob):
        '''numpy input
        '''
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    
    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)

        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)

    def synchronize(self, ):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

        elif self.backend == 'cuda':
            self.stream.synchronize()
    
    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)

    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 

def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for l, b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=str(lab[l].item()), fill='blue', )

        im.save(f'results_{i}.jpg')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-trt', '--trt-file', type=str, default="modelo.engine")
    parser.add_argument('-f', '--im-file', type=str, default="000000000285.jpg")
    parser.add_argument('-d', '--device', type=str, default='cuda:0')

    args = parser.parse_args()

    m = TRTInference(args.trt_file, device=args.device)

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None]

    blob = {
        'images': im_data.to(args.device), 
        'orig_target_sizes': orig_size.to(args.device),
    }
    m.warmup(blob,10)
    v = m.speed(blob,500)
    print(1.0/v)
    # draw([im_pil], output['labels'], output['boxes'], output['scores'])