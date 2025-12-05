import time
import torch
from rich.progress import track

for i in track(range(20), description="For example:"):
    time.sleep(0.05)

x = torch.rand(5, 3)
print("There is a gpu available!" if torch.cuda.is_available() else "problem with the gpu")

