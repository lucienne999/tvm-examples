# 非常偷懒的复制粘贴
# 仅使用 graph exec

import torch # somehow should put front tvm, otherwise appear `free(): invalid pointer`
import tvm
from tvm import relay
from torch import nn
import numpy as np
import os
import tempfile


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        return x, x
    
    
@torch.no_grad()
def init_weights(m: nn.Module):
    for weights in m.parameters():
        torch.nn.init.uniform_(weights)

pt_mlp = MLP()
pt_mlp.apply(init_weights)
device_str = "cuda"
inputs = np.random.random(size=(1, 784)).astype(np.float32)
pt_inputs = torch.from_numpy(inputs)
with torch.no_grad():
    pt_outputs = pt_mlp(pt_inputs)

input_info = [((1, 784), "float32")]
traced_model = torch.jit.trace(pt_mlp.cpu(), pt_inputs)  # torch.jit.trace 执行有条件，可以查看 pytorch 的文档
mod, params = relay.frontend.from_pytorch(traced_model, [("inp0", (1, 784))])  
dev = tvm.cuda()

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(
            mod,
            target=tvm.target.Target(device_str),
            params=params
        )
    lib.export_library("./mlp1_cuda.so")


from tvm import runtime
dtype = "float32"
nd_inputs = tvm.nd.array(inputs)


from tvm.contrib import graph_executor
loaded_lib = runtime.load_module("./mlp1_cuda.so")  
module = graph_executor.GraphModule(lib["default"](dev))
module.set_input("inp0", nd_inputs)
module.run()
h = []
h.append(module.get_output(0))
h.append(module.get_output(1))

# 数值验证

pt2np_outputs0 = pt_outputs[0].numpy()
pt2np_outputs1 = pt_outputs[1].numpy()
np.testing.assert_allclose(pt2np_outputs0, h[0].numpy(), atol=1e-6, rtol=1e-6)
np.testing.assert_allclose(pt2np_outputs1, h[1].numpy(), atol=1e-6, rtol=1e-6)
    
print("[INFO] export done, numerical check passed.")

