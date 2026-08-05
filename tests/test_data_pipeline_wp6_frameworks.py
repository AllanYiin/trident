import unittest

import numpy as np

from trident.data.pipeline import (
    DataProvider, Dataset, DatasetSchema, FieldSpec, Iterator,
    TensorFlowTensorizer, TorchTensorizer,
)


def regression_rows():
    return [
        {"x": np.asarray([value], np.float32),
         "target": np.asarray([2.0 * value], np.float32)}
        for value in (1.0, 2.0, 3.0, 4.0)
    ]


class WP6FrameworkTests(unittest.TestCase):
    def test_pytorch_dataloader_one_training_step(self):
        import torch

        iterator = Iterator(Dataset(regression_rows()), batch_size=4,
                            tensorizer=TorchTensorizer())
        loader = iterator.to_torch_dataloader()
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        before = model.weight.detach().clone()
        batch = next(iter(loader))
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(batch["x"]), batch["target"])
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.weight.detach()))

    def test_tensorflow_dataset_one_training_step_and_short_tail(self):
        import tensorflow as tf

        iterator = Iterator(Dataset(regression_rows() + regression_rows()[:1]),
                            batch_size=4)
        dataset = iterator.to_tensorflow_dataset(prefetch=1)
        weight = tf.Variable([[0.0]], dtype=tf.float32)
        batches = list(dataset)
        self.assertEqual([int(batch["x"].shape[0]) for batch in batches], [4, 1])
        with tf.GradientTape() as tape:
            prediction = tf.matmul(batches[0]["x"], weight)
            loss = tf.reduce_mean(tf.square(prediction - batches[0]["target"]))
        gradient = tape.gradient(loss, weight)
        weight.assign_sub(0.05 * gradient)
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(loss))))
        self.assertNotEqual(float(weight.numpy()[0, 0]), 0.0)

    def test_tensorflow_tensorizer_keeps_batch_mapping(self):
        import tensorflow as tf

        batch = next(iter(Iterator(Dataset(regression_rows()), batch_size=2,
                                   tensorizer=TensorFlowTensorizer())))
        self.assertIsInstance(batch["x"], tf.Tensor)
        self.assertEqual(tuple(batch["x"].shape), (2, 1))

    def test_trident_training_plan_adapter_registration_and_batch_contract(self):
        from trident.optims.trainers import TrainingPlan

        schema = DatasetSchema([
            FieldSpec("x", kind="data", metadata={"role": "input"}),
            FieldSpec("target", kind="label", metadata={"role": "target"}),
        ])
        provider = DataProvider(train=Dataset(regression_rows(), schema=schema))
        adapter = provider.for_trident_trainer("train", batch_size=2)
        plan = TrainingPlan().with_data_loader(adapter).with_batch_size(2)
        self.assertIs(plan._dataloaders.value_list[-1], adapter)
        wrapped = next(iter(adapter))
        names = [field.name for field in wrapped.keys()]
        self.assertEqual(names, ["x", "target"])
        self.assertEqual(adapter.traindata.data.symbol, "x")
        self.assertEqual(adapter.traindata.label.symbol, "target")


if __name__ == "__main__":
    unittest.main()
