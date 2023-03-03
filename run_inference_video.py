"""
Runs a pre-compiled model via TIDL. Saves the sample detections.

Intended to be run on the embedded device.

Takes input as a video file and outputs a video file (.avi). To run detections on loose images, see "run_inference_images.py".

Args:
  - Model path. Path to the ONNX model file. Ideally, use the "_with_shapes.onnx" output during compilation.
  - TIDL artifacts directory. The TIDL metadata directory generated during compilation. compile_model.py in this repo calls it "tidl_output".
  - Sample data path. Path to a video file.

Will print the average time taken to run each loop, and save the detections as a video file called "sample_detections.avi".
"""

import os
import sys
import time
from tqdm import tqdm

import onnxruntime as rt
import cv2
import numpy as np

# os.environ["TIDL_RT_PERFSTATS"] = "1"

CONFIDENCE_THRESHOLD = 0.3

def render_boxes(image, inference_width, inference_height, output):
    assert len(output.shape) == 3
    output_count = output.shape[1]

    for i in range(output_count):
        x1, y1, x2, y2, confidence, class_idx_float = output[0, i, :]
        if confidence <= CONFIDENCE_THRESHOLD:
            continue

        x1 = int(round(x1 / inference_width * image.shape[1]))
        y1 = int(round(y1 / inference_height * image.shape[0]))
        x2 = int(round(x2 / inference_width * image.shape[1]))
        y2 = int(round(y2 / inference_height * image.shape[0]))

        # Yes, TI outputs the class index as a float...
        class_draw_color = {
            # Colors for boxes of each class, in (R, G, B) order.
            0.: (255, 50, 50),
            1.: (50, 50, 255),
            # TODO: if using more than two classes, pick some more colors...
        }[class_idx_float]

        # Reverse RGB tuples since OpenCV images default to BGR
        cv2.rectangle(image, (x1, y1), (x2, y2), class_draw_color[::-1], 3)

if __name__ == "__main__":
    _, model_path, artifacts_dir, test_video = sys.argv

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
    reader = cv2.VideoCapture(test_video)
    writer = cv2.VideoWriter(
        "sample_detections.avi",
        cv2.VideoWriter_fourcc(*'XVID'),
        reader.get(cv2.CAP_PROP_FPS),
        (int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    pbar = tqdm(total=reader.get(cv2.CAP_PROP_FRAME_COUNT))

    start = time.time()
    frame_count = 0

    success, frame = reader.read()

    while success is True:
        frame_count += 1

        # YOLOv5 normalizes RGB 8-bit-depth [0, 255] into [0, 1]
        # Model trained with RGB channel order but OpenCV loads in BGR order, so reverse channels.
        input_data = cv2.resize(frame, (width, height)).transpose((2, 0, 1))[::-1, :, :] / 255

        input_data = input_data.astype(np.float32)
        input_data = np.expand_dims(input_data, 0)

        detections, = sess.run(None, {input_name: input_data})
        render_boxes(frame, width, height, detections[0, :, :, :])

        writer.write(frame)

        pbar.update(1)
        success, frame = reader.read()

    writer.release()

    end = time.time()
    per_frame_ish = (end-start)/frame_count*1000
    print(f"Time per loop, incl. I/O and drawing (ms): {per_frame_ish}")
