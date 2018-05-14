import cntk as C
import numpy as np
import onnx
import os

model_file = 'model.onnx'
data_dir = 'test_data_set_0'

def SaveTensorProto(file_path, variable, data):
    tp = onnx.TensorProto()
    tp.name = variable.uid
    for i in range(len(variable.dynamic_axes)):
        tp.dims.append(1) # pad 1 for the each dynamic axis
    for d in variable.shape:
        tp.dims.append(d)
    tp.data_type = onnx.TensorProto.FLOAT
    tp.raw_data = data.tobytes()
    with open(file_path, 'wb') as f:
        f.write(tp.SerializeToString())
        
def SaveData(test_data_dir, prefix, variables, data_list):
    if isinstance(data_list, np.ndarray):
        data_list = [data_list]
    for (i, d), v in zip(enumerate(data_list), variables):
        SaveTensorProto(os.path.join(test_data_dir, '{0}_{1}.pb'.format(prefix, i)), v, d)

def Save(dir, func, inputs, outputs):
    if not os.path.exists(dir):
        os.makedirs(dir)
    func.save(os.path.join(dir,model_file), C.ModelFormat.ONNX)
    
    test_data_dir = os.path.join(dir, data_dir)
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)

    SaveData(test_data_dir, 'input', func.arguments, inputs)
    SaveData(test_data_dir, 'output', func.outputs, outputs)
        
def GenSimple():
    x = C.input_variable((1,3,)) # TODO: fix CNTK exporter bug with shape (3,)
    y = C.layers.Embedding(2)(x) + C.parameter((-1,))
    data_x = np.random.rand(*x.shape).astype(np.float32)
    data_y = y.eval(data_x)
    Save('test_simple', y, data_x, data_y)

if __name__=='__main__':
    np.random.seed(0)
    GenSimple()
