import _bootstrap  # noqa: F401
import numpy as np

from trident.data.pipeline import Dataset, Iterator, TorchTensorizer


rows = [{"x": np.asarray([value], np.float32),
         "target": np.asarray([2 * value], np.float32)}
        for value in range(1, 9)]
iterator = Iterator(
    Dataset(rows), batch_size=4, workers=2, prefetch_batches=2,
    memory_budget_mb=16, tensorizer=TorchTensorizer())

import torch

model = torch.nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
batch = next(iter(iterator.to_torch_dataloader()))
loss = torch.nn.functional.mse_loss(model(batch["x"]), batch["target"])
optimizer.zero_grad()
loss.backward()
optimizer.step()
print("loss:", float(loss))
