import os
import sys
import shutil

import onnxruntime as rt
import onnx
import onnx.shape_inference
import numpy as np

os.environ["TIDL_RT_PERFSTATS"] = "1"

if __name__ == "__main__":
    _, model_path, out_dir_path = sys.argv

    tidl_tools_path = os.environ["TIDL_TOOLS_PATH"]

    os.makedirs(out_dir_path, exist_ok=True)

    out_model_name = os.path.splitext(os.path.basename(model_path))[0] + "_with_shapes.onnx"
    out_model_path = os.path.join(out_dir_path, out_model_name)
    onnx.shape_inference.infer_shapes_path(model_path, out_model_path)
    
    artifacts_dir = os.path.join(out_dir_path, "intermediate_artifacts")
    shutil.rmtree(artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=False)

    so = rt.SessionOptions()
    
    print("Available execution providers : ", rt.get_available_providers())
    
    print(artifacts_dir)
    compilation_options = {
        "platform": "J7",
        "version": "8.2",

        "tidl_tools_path": tidl_tools_path,
        "artifacts_folder": artifacts_dir,

        "tensor_bits": 8,
        # "import": "no",
        
        "debug_level": 300,
        
        # note: to add advanced options here, start it with 'advanced_options:'
        # example 'advanced_options:pre_batchnorm_fold':1
        "advanced_options:calibration_frames": 1,
        # "advanced_options:calibration_iterations": XXX,
    }
    
    desired_eps = ['TIDLCompilationProvider','CPUExecutionProvider']
    sess = rt.InferenceSession(
        out_model_path,
        providers=desired_eps,
        provider_options=[compilation_options, {}],
        sess_options=so
    )
    
    input_details, = sess.get_inputs()
    batch_size, channel, height, width = input_details.shape
    print(input_details.shape)
    assert isinstance(batch_size, str) or batch_size == 1
    assert channel == 3
    input_name = input_details.name
    input_type = input_details.type
    
    print(f'Input "{input_name}": {input_type}')

    assert input_type == 'tensor(float)'
    
    
    # dummy_data = np.random.standard_normal(size = (1, channel, height, width))
    # dummy_data = (dummy_data - np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None]) / np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None]
    dummy_data = np.ones((1, channel, height, width))

    dummy_data = dummy_data.astype(np.single)
    print(dummy_data.dtype)

    # TODO: de-mean and normalize a proper image
    
    input_data = dummy_data
    sess.run(None, {input_name: input_data})
    
    print("Exiting -- pay no attention to the SIGSEGV behind the curtain")
