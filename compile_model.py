"""
Compiles an ONNX model into TI's proprietary format, quantizing to 8-bit precision in the process.

Must be run on an x86 host.

Args:
  - Model path. Path to the trained model in ONNX format. There must be a ".prototxt" file of the same name alongside it.
  - Calibration images directory. Path to a folder containing sample images to use when calibrating the model.
  - Output directory. tTarget for the compiled output and intermediate files.
"""

import os
import sys
import shutil

import onnxruntime as rt
import onnx
import onnx.shape_inference
import numpy as np
from PIL import Image

os.environ["TIDL_RT_PERFSTATS"] = "1"

if __name__ == "__main__":
    _, model_path, calibration_images_path, out_dir_path = sys.argv

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

    calibration_images = [ os.path.join(calibration_images_path, name) for name in os.listdir(calibration_images_path) ]

    num_calibration_frames = len(calibration_images)
    num_calibration_iterations = 50 # TODO: Probably more than necessary, but 50 is the default.
    # Documentation on available options: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md
    compilation_options = {
        "platform": "J7",
        "version": "8.2",

        "tidl_tools_path": tidl_tools_path,
        "artifacts_folder": artifacts_dir,

        "tensor_bits": 8,
        
        # YOLO-specific configuration. 
        "model_type": "OD",
        'object_detection:meta_arch_type': 6,
        # The below are copied from the linked sample code and are specific to TI's trained "small" model variant and 384px input dimension.
        # See the note in the README for my thoughts on this parameter. You should _probably_ look in your .prototxt and replicate those numbers here.
        # https://github.com/TexasInstruments/edgeai-benchmark/blob/16e57a65e7aa2802a6ac286be297ecc5cad93344/configs/detection.py#L184
        # 'advanced_options:output_feature_16bit_names_list': '168, 370, 680, 990, 1300',

        # Note: if this parameter is omitted, the TIDL framework crashes due to buffer overflow rather than giving an error
        'object_detection:meta_layers_names_list': os.path.splitext(model_path)[0] + ".prototxt",

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
    print(f"Input shape: {input_details.shape}")

    assert isinstance(batch_size, str) or batch_size == 1
    assert channel == 3

    input_name = input_details.name
    input_type = input_details.type

    print(f'Input "{input_name}": {input_type}')

    assert input_type == 'tensor(float)'

    for image_path in calibration_images:
        # YOLOv5 normalizes RGB 8-bit-depth [0, 255] into [0, 1]
        input_data = np.asarray(Image.open(image_path).resize((width, height))).transpose((2, 0, 1)) / 255
        input_data = input_data.astype(np.float32)
        input_data = np.expand_dims(input_data, 0)

        sess.run(None, {input_name: input_data})
