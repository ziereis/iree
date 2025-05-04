import torch
import time
from pathlib import Path
import numpy
import iree.runtime as ireert


a = torch.rand(16, 16, dtype=torch.float16)
b = torch.rand(16, 16, dtype=torch.float16)


config = ireert.Config("cuda")
vmm = ireert.VmModule.mmap(config.vm_instance, "matmul.vmfb")
ctx = ireert.SystemContext(vm_modules=[vmm], config=config)
# rt = AtticRuntime("contract_nv_sync.vmfb", driver=AtticRTDriver.CUDA)
func = ctx.modules.module["matmul"]


res = func(a, b)
torch.set_printoptions(threshold=10_000)
numpy.set_printoptions(threshold=10_000)
print(res.to_host())

print(torch.matmul(a.to("cuda"), b.to("cuda")))
