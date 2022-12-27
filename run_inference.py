"""
Runs a pre-compiled model via TIDL. Computes an emperical average time per inference pass and saves the sample detections.

Args:
  - Model path. Path to the ONNX model file. Ideally, use the "_with_shapes.onnx" output during compilation.
  - TIDL artifacts directory. The TIDL metadata directory generated during compilation. compile_model.py in this repo calls it "tidl_output".
  - Sample data path. Path to a directory containing sample images to run through the model.

Will print the average time taken to run inference, and save the detections in a directory called "sample_detections".
"""

import os
import sys
import time
import shutil

import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw

# os.environ["TIDL_RT_PERFSTATS"] = "1"

CONFIDENCE_THRESHOLD = 0.5

def render_boxes(image_path, inference_width, inference_height, output):
    assert len(output.shape) == 3
    output_count = output.shape[1]
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for i in range(output_count):
        x1, y1, x2, y2, confidence, class_idx_float = output[0, i, :]
        if confidence <= CONFIDENCE_THRESHOLD:
            continue

        x1 = x1 / inference_width * image.width
        y1 = y1 / inference_height * image.height
        x2 = x2 / inference_width * image.width
        y2 = y2 / inference_height * image.height

        # Yes, TI outputs the class index as a float...
        class_draw_color = {
            0.: (255, 50, 50),
            1.: (50, 50, 255),
            # TODO: if using more than two classes, pick some more colors...
        }[class_idx_float]

        draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline=class_draw_color, width=3)

    os.makedirs("sample_detections/", exist_ok=True)
    image.save(os.path.join("sample_detections", os.path.splitext(os.path.basename(image_path))[0] + ".png"))

if __name__ == "__main__":
    _, model_path, artifacts_dir, test_images_dir = sys.argv

    so = rt.SessionOptions()

    print("Available execution providers : ", rt.get_available_providers())

    runtime_options = {
        "platform": "J7",
        "version": "8.2",

        "artifacts_folder": artifacts_dir,
        #"enableLayerPerfTraces": True,
    }

    desired_eps = ['TIDLExecutionProvider', 'CPUExecutionProvider']
    sess = rt.InferenceSession(
        model_path,
        providers=desired_eps,
        provider_options=[runtime_options, {}],
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

    test_image_paths = [ os.path.join(test_images_dir, name) for name in os.listdir(test_images_dir) ]
    test_image_data = []
    for image_path in test_image_paths:
        # YOLOv5 normalizes RGB 8-bit-depth [0, 255] into [0, 1]
        input_data = np.asarray(Image.open(image_path).resize((width, height))).transpose((2, 0, 1)) / 255

        input_data = input_data.astype(np.float32)
        input_data = np.expand_dims(input_data, 0)

        test_image_data.append(input_data)

    # Effective inference latency computation: the amount of time it takes, as observed from user code.
    # Note that the configuration I've tested offloads the non-maximum suppression (NMS) and YOLO output extraction to the TIDL runtime.
    # This means that execution will take longer when more objects are in view. Use representative images for timing purposes.
    NUM_TIMING_REPS = 200
    start = time.time()
    for it in range(NUM_TIMING_REPS):
        for i, input_data in enumerate(test_image_data):
            output = sess.run(None, {input_name: input_data})
    end = time.time()
    TOTAL_EXECUTIONS = NUM_TIMING_REPS * len(test_image_data)
    per_frame_ish = (end-start)/TOTAL_EXECUTIONS*1000
    print(f"Time per inference (ms): {per_frame_ish}")

    for image_path, input_data in zip(test_image_paths, test_image_data):
        # Assumes one output head (postprocessed+YOLO extraction complete) and one image per batch.
        detections, = sess.run(None, {input_name: input_data})
        render_boxes(image_path, width, height, detections[0, :, :, :])
