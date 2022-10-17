import os
import sys
import time
import shutil

import onnxruntime as rt
import numpy as np

os.environ["TIDL_RT_PERFSTATS"] = "1"

if __name__ == "__main__":
    _, model_path, artifacts_dir = sys.argv

    so = rt.SessionOptions()

    print("Available execution providers : ", rt.get_available_providers())

    runtime_options = {
        "platform": "J7",
        "version": "8.2",

        "artifacts_folder": artifacts_dir,
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
    print(input_details.shape)
    assert isinstance(batch_size, str) or batch_size == 1
    assert channel == 3
    input_name = input_details.name
    input_type = input_details.type

    # TODO: why does it say float?
    print(f'Input "{input_name}": {input_type}')

    # assert input_type == 'tensor(float)'

    for i in range(20):
        dummy_data = np.random.standard_normal(size = (1, channel, height, width))
        # Standard torchvision normalization parameters used by the pretrained model
        dummy_data = (dummy_data - np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None]) / np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None]

        dummy_data = dummy_data.astype(np.single)

        # TODO: de-mean and normalize a proper image
        input_data = dummy_data
        output = sess.run(None, {input_name: input_data})
    
    dummy_data = np.random.standard_normal(size = (1, channel, height, width))
    # Standard torchvision normalization parameters used by the pretrained model
    dummy_data = (dummy_data - np.array((0.485, 0.456, 0.406), dtype=np.single)[:, None, None]) / np.array((0.229, 0.224, 0.225), dtype=np.single)[:, None, None]
    dummy_data = dummy_data.astype(np.single)

    start = time.time()
    for i in range(200):
        input_data = dummy_data
        output = sess.run(None, {input_name: input_data})
    end = time.time()
    per_frame_ish = (end-start)/200*1000
    print(output)
    print(per_frame_ish)
