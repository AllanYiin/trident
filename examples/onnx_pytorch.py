
from six import string_types
from typing import Any, List, Text, Dict, Set
from onnx import ModelProto, ValueInfoProto

import onnx.checker
import onnx
from onnx import optimizer
import onnx.checker
# Load the ONNX model
model = onnx.load("zz_mean.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)
print(model)


onnx.save(model, 'model.onnx')

print(model.graph)

all_passes = optimizer.get_available_passes()
for p in all_passes:
    print('\t{}'.format(p))
print()

# Pick one pass as example
passes = ['fuse_consecutive_transposes']

# Apply the optimization on the original serialized model
optimized_model = optimizer.optimize(model, passes)


onnx.save(optimized_model, 'model.onnx')
