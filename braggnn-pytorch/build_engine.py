import engine as eng
import argparse
from onnx import ModelProto
import tensorrt as trt 
 
name = '0center-gpu-opset11' 
engine_name = '{}_16384_FP16_TRT7.plan'.format(name)
onnx_path = '/home/nikilr2/trt_tutorial/braggnn-pytorch/{}.onnx'.format(name)
batch_size = 16384

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

shape = [batch_size , d0, d1, d2]
engine = eng.build_engine(onnx_path, shape= shape)
eng.save_engine(engine, engine_name)
print(shape)
