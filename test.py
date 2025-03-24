import torch
import time
from pathlib import Path
import numpy

from attic.runtime.runtime import AtticRuntime, AtticRTDriver, AtticFunction


a = torch.rand(16, 16, dtype=torch.float16)
b = torch.rand(16, 16, dtype=torch.float16)


rt = AtticRuntime("tileandfuse_nv_sync.vmfb", driver=AtticRTDriver.CUDA)
# rt = AtticRuntime("contract_nv_sync.vmfb", driver=AtticRTDriver.CUDA)
func = rt.get_function_by_name("matmul")

res = func(a, b)
torch.set_printoptions(threshold=10_000)
numpy.set_printoptions(threshold=10_000)
print(res)

print(torch.matmul(a.to("cuda"),b.to("cuda")))
