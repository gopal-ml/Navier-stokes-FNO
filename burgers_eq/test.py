from Navier.core.fno import FNO1d
from torch.utils.data import DataLoader
import torch

# model = FNO1d(
#     modes=16,
#     width=64
# )

# x = torch.randn(4, 2, 1024)

# y = model(x)

# print(y.shape)

data = torch.load("burgers_dataset.pt")

print(torch.isnan(data["a"]).any())
print(torch.isnan(data["u"]).any())

print(data["a"].shape)
print(data["u"].shape)

print(data["a"].mean(), data["a"].std())
print(data["u"].mean(), data["u"].std())



loader = DataLoader(data, batch_size=4)

batch = next(iter(loader))

model = FNO1d(
    modes=12,
    width=32
)

y = model(batch["a"])

print(batch["a"].shape)
print(batch["u"].shape)
print(y.shape)
