为什么有这个仓库？

1. 虽然 TVM 官方的文档日益完善，但是给的例子都太局部。在学习的时候，对一个 pytorch 模型如何部署上板，略一头雾水。本仓库记录一些端到端的部署例子。
2. TVM 中的概念众多，不结合例子并不好理解。直接读源码工程量又太大。
3. 学习过程，写了很多零零散散的脚本。想系统记录一下。

如有不对请大佬多多指正。

下面是本仓库包含例子的使用说明：

- 使用 relay 来部署模型。

  前言：relay 和 relax 是 tvm 的两种中间表达，relax 是后续推出的版本，包含了更多的特性（有兴趣看 TVM 官方的文档）；因为 relay 存在时间比较长，找到的很多教程也是基于 relay 来构建的；
  
  作为第一个例子，还是从 relay 开始。

  1. 首先，从 python 端导出我们的模型，本例子提供了简单的 mlp: 

  ```
  cd relay
  python export.py 
  ```

  2. 得到 mlp.so, 在上述 python 脚本中，我们已经加载运行了，在接触的业务中通常需要用 cpp 加载

  ```
  make relay_cpu
  ./rely_example
  ```

  执行后打印了结果；

  3. 更常见的需求是运行在 cuda 上，需要对代码修改：

  - export 需要指定设备
  - 需要额外数据的拷贝
    - TVM 中可用 DLManagedTensor 来管理不同设备的数据，当然也可以用 cudamemcpy
    - 在下面的例子中，实现了两种方式

  ```
  python export_gpu.py
  make relay_cuda
  ./relay_cuda_example



  ```


  