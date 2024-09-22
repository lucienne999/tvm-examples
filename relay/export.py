import torch # somehow should put front tvm, otherwise appear `free(): invalid pointer`
import tvm
from tvm import relay
from torch import nn
import numpy as np
import os
import tempfile

# step1: 定义你的 pytorch 模型
# 本例子中是一个非常简单的 mlp
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

# 准备输入和输出，进行转换后的数值校验
inputs = np.random.random(size=(1, 784)).astype(np.float32)
pt_inputs = torch.from_numpy(inputs)
with torch.no_grad():
    pt_outputs = pt_mlp(pt_inputs)

# step2: 导出模型
input_info = [((1, 784), "float32")]
traced_model = torch.jit.trace(pt_mlp.cpu(), pt_inputs)  # torch.jit.trace 执行有条件，可以查看 pytorch 的文档
mod, params = relay.frontend.from_pytorch(traced_model, [("inp0", (1, 784))])  
dev = tvm.cpu()

# mod 即转换后的 TVM IR
# 取消注释查看 IR
# mod.show()


# step2-1: 使用 graph exec 导出
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(
            mod,
            target=tvm.target.Target("llvm"),
            params=params
        )
    lib.export_library("./mlp1.so")


# step2-2: 使用 vm 导出, , 他们的加载会有所不同
from tvm.relay import vm
with tvm.transform.PassContext(opt_level=3):
    exe = vm.compile(mod, "llvm", params=params)
    exe.mod.export_library("./mlp2.so") 


# step3
from tvm import runtime
dtype = "float32"
nd_inputs = tvm.nd.array(inputs)

# step3-1
from tvm.contrib import graph_executor
loaded_lib = runtime.load_module("./mlp1.so")  
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
    
    
# step3-2: 使用 vm 加载
from tvm.runtime import vm as vm_rt
loaded_lib = runtime.load_module("./mlp2.so")  
vm = vm_rt.VirtualMachine(loaded_lib, tvm.cpu())  # vm 管理输入和输出


# 运行导出的模型
vm.set_input("main",  nd_inputs)
vm.invoke_stateful("main") 
h = vm.get_outputs()

# 也可以这样
# h = vm.invoke("main", inputs)


# 数值验证

pt2np_outputs0 = pt_outputs[0].numpy()
pt2np_outputs1 = pt_outputs[1].numpy()
np.testing.assert_allclose(pt2np_outputs0, h[0].numpy(), atol=1e-6, rtol=1e-6)
np.testing.assert_allclose(pt2np_outputs1, h[1].numpy(), atol=1e-6, rtol=1e-6)
    
print("[INFO] export done, numerical check passed.")

