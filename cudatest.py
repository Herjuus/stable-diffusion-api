import torch

torch.cuda.empty_cache()

print("Torch version:",torch.__version__)

print("Is CUDA enabled?",torch.cuda.is_available())