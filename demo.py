import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession("model.onnx")

# Get input name and shape
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print("Input name:", input_name)
print("Input shape:", input_shape)

# Dummy input (replace with real data)
dummy_input = np.random.rand(*[s if isinstance(s, int) else 1 for s in input_shape]).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: dummy_input})

print("Output shape:", [o.shape for o in outputs])
