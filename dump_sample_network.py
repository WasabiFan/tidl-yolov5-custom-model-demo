"""
Utility script which saves a simple model (pre-trained ResNet-50 classifier from PyTorch) via ONNX.
"""

import sys

import torch
from torchvision.models import resnet50

if __name__ == "__main__":
    _, out_file = sys.argv
    
    # model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3*224*224, 1))
    model = resnet50(pretrained=True)
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(
        model,                     # model being run
        x,                         # model input (or a tuple for multiple inputs)
        out_file,                  # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}}
        )

    import onnx
    onnx_model = onnx.load(out_file)
    onnx.checker.check_model(onnx_model)
