"""
Runs a pre-compiled model via TIDL. Computes an emperical average time per inference pass and saves the sample detections.

Takes input as loose image files. To run detection on a whole video, see "run_inference_video.py".

Args:
  - Model path. Path to the ONNX model file. Ideally, use the "_with_shapes.onnx" output during compilation.
  - TIDL artifacts directory. The TIDL metadata directory generated during compilation. compile_model.py in this repo calls it "tidl_output".
  - Sample data path. Path to a directory containing sample images to run through the model.

Will print the average time taken to run inference, and save the detections in a directory called "sample_detections".
"""

import os
import sys
import time

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
        # Model trained with RGB channel order but OpenCV loads in BGR order, so reverse channels.
        frame = cv2.imread(image_path)
        input_data = cv2.resize(frame, (width, height)).transpose((2, 0, 1))[::-1, :, :] / 255

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

    os.makedirs("sample_detections/", exist_ok=True)
    for image_path, input_data in zip(test_image_paths, test_image_data):
        # Assumes one output head (postprocessed+YOLO extraction complete) and one image per batch.
        detections, = sess.run(None, {input_name: input_data})

        image = cv2.imread(image_path)
        render_boxes(image, width, height, detections[0, :, :, :])
        cv2.imwrite(os.path.join("sample_detections", os.path.splitext(os.path.basename(image_path))[0] + ".png"), image)
