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

    artifacts_dir = os.path.join(out_dir_path, "tidl_output")
    try:
        shutil.rmtree(artifacts_dir)
    except FileNotFoundError:
        pass

    os.makedirs(artifacts_dir, exist_ok=False)

    so = rt.SessionOptions()
    print("Available execution providers : ", rt.get_available_providers())

    num_calibration_frames = 1
    num_calibration_iterations = 1
    compilation_options = {
        "platform": "J7",
        "version": "8.2",

        "tidl_tools_path": tidl_tools_path,
        "artifacts_folder": artifacts_dir,

        "tensor_bits": 8,
        # "import": "no",
        
        "model_type": "OD",
        'object_detection:meta_arch_type': 6,
        'object_detection:meta_layers_names_list': os.path.splitext(model_path)[0] + ".prototxt", # Note: if this file is omitted, TIDL framework crashes due to buffer overflow
        'advanced_options:output_feature_16bit_names_list': '168, 370, 432, 494, 556', # TODO: copied from official samples

        "debug_level": 300,

        # note: to add advanced options here, start it with 'advanced_options:'
        # example 'advanced_options:pre_batchnorm_fold':1
        "advanced_options:calibration_frames": num_calibration_frames,
        "advanced_options:calibration_iterations": num_calibration_iterations,
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

    for i in range(num_calibration_frames):
        # YOLOv5 normalizes RGB 8-bit-depth [0, 255] into [0, 1]
        # TODO: use proper image data
        dummy_data = np.random.random_sample((1, channel, height, width)).astype(np.single)

        input_data = dummy_data
        sess.run(None, {input_name: input_data})
    
    print("Compilation complete")
