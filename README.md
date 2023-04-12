# TIDL YOLOv5 custom model walkthrough

End-to-end instructions for training, compiling and running a YOLOv5 model on the TDA4VM/BeagleBone
AI-64 with TIDL.

Unofficial and unsupported, but I'll probably help out however I can.

I'm targeting YOLOv5 because it's officially supported by TI. They have a fork of the upstream repo
with customizations for TIDL. This repo is probably most of the way toward compiling other kinds of
models, but I haven't tried them. Feel free to open an issue if you have success.

I have provided Python and C++ inference samples. They both use onnxruntime for evaluation. Observed
inference latency (time taken to run inference) is notably better in C++, but both are reasonably
fast.

**Tested TIDL version: 08_02_00_01**

**Tested BeagleBone AI-64 image: `bbai64-debian-11.6-xfce-edgeai-arm64-2023-02-05-10gb.img.xz`**

The version of the EdgeAI runtime installed on your device must match the version of the TIDl tools
used to compile the model on your PC.

## Overview

This repo includes:
- Training scripts
  - A Jupyter Notebook with necessary steps to train a YOLOv5 model (including TI's customizations)
    and export it in TIDL-compatible format. It can be adapted for any environment but I have tested
    it on Google Colab.
- Compilation scripts
  - `compile_model.py`, a sample utility script which runs TI's compilation and quantization tools on
    a trained model.
  - `docker/Dockerfile`, a Docker container image that you can use to run `compile_model.py` with
    minimal hassle.
- Inference demos
  - `run_inference_images.py`, a Python sample app to run inference on input images, time execution,
    and render the results.
  - `run_inference_video.py`, same as above but on whole video files.
  - `cpp/run_inference_images/`, a reimplementation of the eponymous Python script, in C++.

## Prerequisites

- A GPU-enabled training machine.

  I provide a Jupyter Notebook that can be used with Google Colab.
  If you have your own hardware, that may be easier than messing with Colab. Alternately, you can
  skip training your own model and use TI's provided weights as a proof-of-concept.
- A host machine capable of running x86 Docker images, OR a native x86 Linux machine for model
  compilation.

  Unfortunately, TI's tools require Python 3.6 and only officially support Ubuntu 18.04. As a result,
  I recommend using my provided Docker container. This should work on Windows, Linux and macOS x86
  hosts.

- A TDA4VM or other TIDL-supported chip, to test the results. I developed this on a BeagleBone AI-64.

## Usage

### Step 1: Train a YOLOv5 model

TI has a fork of the YOLOv5 codebase with customizations applied. It can be found [here](https://github.com/TexasInstruments/edgeai-yolov5).
The model architectures are slightly modified for performance reasons, and the export scripts have
been augmented to produce extra data files. As such, I recommend using exclusively their fork (or a
fork thereof) and trained model weights derived from it.

I have prepared a step-by-step guide to train and export an appropriate model. I provide it below
via Google Colab, which you can configure to use free GPU cloud compute resources with timeout
restrictions. Feel free to replicate the steps in your own environment if you'd like to use
something else. Google Colab is not great for "real" model development.

You will need:

- Labeled training data. Either use one of the datasets provided by a third party or make your own.
  Use standard Darknet/YOLO format.

Things to consider:

- There are multiple model flavors/sizes (large, medium, small). Pick the one that suits your needs.
- You can choose the input data resolution. I've only tested fixed, square resolutions. Smaller images mean faster inference.
- There are lots of other hyperparameters you can play with.

Make sure to save:

- The exported `.onnx` (standard format) and `.prototxt` (TI proprietary) files.

**Access the notebook here: https://colab.research.google.com/drive/13p9908P_HMQ0YI5SBWBS2SjKq7PYTykL?usp=sharing**

### Step 2: Compile for TIDL's runtime

TI has an ONNX runtime component for TIDL, but it requires pre-quantized weights and other
metadata. You will now generate this metadata.

TIDL "compiles" models by analyzing data as it flows through the computation graph and making
simplifications. It maps the arbitrary precision floating-point values the model was trained on to
a fixed-point 8- or 16-bit integer representation. This means it must know, for each parameter in
the model, the range and distribution of the numbers that flow through it. When compiling, we
provide a handful of "calibration images" which the tool runs through the model to make these
judgements.

You will want to prepare a folder of images that includes a representative sample of inputs. It
should:
- (TI recommendation) Be around 20 pictures. More is probably better, but also takes longer to run the compilation. I've
  used 5-15 and had good results.
- (My recommendation) Include a variety of backgrounds, styles, lighting, etc. -- whatever variation your problem space
  involves.
- (My recommendation) Have true positives (i.e., most images should include detected objects)
- (My recommendation) Represent most/all of your object classes

It's fine to use data from the training set. I just cherry-picked some images from the training set
that I felt were reasonable.

You will need:
- An x86 host machine to run model compilation. The steps below will focus on the Docker option, but
  if you would prefer to do it natively see the expandable "Alternative: model compilation without
  Docker" section below.

- Calibration data.

  See above for recommendations. The script expects a single flat directory with image files in it.
- The two files from the previous step, in the same directory and names differing only by extension.

Things to consider:
- The script is currently hard-coded to use 50 calibration iterations. It seemed to work fine with 5.
  50 is the default, so kept it that way, but feel free to change it.
  
  More calibration iterations means more waiting. Each one takes 30 seconds to one minute.
- You may want to enable or change the `output_feature_16bit_names_list` parameter.

  This option lists layer names whose outputs will be processed at higher resolution than the
  default 8 bits. TI's "benchmark" repo has magic values of this parameter for each model, but their
  other scripts leave it blank.

  I believe they are attempting to target the output heads: the model processes features with 8-bit
  precision but then would combine and output the final predictions with 16-bit precision. The
  sample values they use _mostly_ fit this theory. They have enabled 16-bit precision on an
  additional very early convolution layer and it is not clear to me why they did so.

  **Regardless, I recommend enabling this parameter in `compile_model.py` and supplying your own
  trained model's output head layer names. These names are listed in the `.prototxt` file.**

Make sure to save:

- The whole output "artifacts" folder. This includes a `.onnx` file and a directory of TIDL spew.

  It is likely that all you need are the `.onnx` and `.bin` files, but I haven't verified this.
  Maybe also one or both of the top-level `.txt` files. Let me know if you find more details.

To perform compilation and calibration, clone this repo. Make sure that Docker is installed and
running. Then run the following:

```bash
cd docker/

# One-time Docker container build:
docker build . -t tidl-model-build-env

# Run compilation:
docker run --rm --shm-size=2gb \
  -v /path/to/my_model_data/:/model_in:ro \
  -v /path/to/calibration_images/:/calibration_images:ro \
  -v $PWD/../:/scripts:ro \
  -v $PWD/artifacts:/model_out \
  tidl-model-build-env \
  python3.6 /scripts/compile_model.py /model_in/last.onnx /calibration_images /model_out
```

Substitute the `/path/to/...` paths with ABSOLUTE (i.e., starting with a drive letter like `C:\` on
Windows or `/` on Linux/macOS) paths to your compiled model and calibration images directory. You
may also want to change the name of the onnx file in the last line of the command if you called it
something other than `last.onnx`.

Don't worry too much about the Docker syntax and parameters if you haven't seen them before. It's
sufficient to change the paths and leave the rest untouched. However, I will note that the `-v`
flags are specifying mappings from your host file system into the Docker container; only the
directories specified with `-v` will be accessible to `compile_model.py`. The syntax is
`-v host_path:container_path:permissions`, so you only want to modify the path up until the colon.

<details>
<summary>Alternative: model compilation without Docker</summary>

To compile without Docker, you will need an x86 Linux PC with Python 3.6. Ideally Ubuntu 18.04. A
virtual machine is fine.

Install the following prerequisites. The listed commands should be sufficient for Ubuntu 18.04.
Please let me know if something is missing.

```bash
sudo apt update
sudo apt install python3-pip python3-setuptools python3-wheel python3.6-dev

sudo apt install cmake build-essential protobuf-compiler libprotobuf-dev

wget https://raw.githubusercontent.com/TexasInstruments/edgeai-tidl-tools/3dc359873f0b80c1e1a0b4eec5d1d02e35d4e532/requirements_pc.txt
python3.6 -m pip install -r requirements_pc.txt

# Make sure to choose the version of the tools corresponding to the version of TIDL on your device.
wget https://github.com/TexasInstruments/edgeai-tidl-tools/releases/download/08_02_00_01-rc1/tidl_tools.tar.gz
# Or, for other versions:
# wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_04_00_00/tidl_tools.tar.gz
# wget https://software-dl.ti.com/jacinto7/esd/tidl-tools/08_06_00_00/TIDL_TOOLS/AM68PA/tidl_tools.tar.gz

tar -xzf tidl_tools.tar.gz
rm tidl_tools.tar.gz

echo 'export TIDL_TOOLS_PATH="$HOME/tidl_tools/"' >> ~/.bash_aliases
echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$TIDL_TOOLS_PATH:$TIDL_TOOLS_PATH/osrt_deps"' >> ~/.bash_aliases
source ~/.bashrc
```

It may be helpful to inspect the Dockerfile in this repo (`docker/Dockerfile`). It targets Ubuntu
20.04 because it requires slightly fewer installs and seems to work fine. You may want to reference
it instead.

To perform compilation and calibration, clone this repo in your x86 environment and run:

```bash
#                          [     ONNX model      ] [calibration data ] [  output directory   ]         
python3.6 compile_model.py my_model_data/last.onnx calibration_images/ my_model_data_compiled/
```

</details>


### Step 3: Try out inference on-device

Note: this would probably work on a variety of devices with TI chips, but I have only tested a
TDA4VM on the BeagleBone AI-64.

#### Python

I have provided two inference scripts: one for loose image files, and one for videos.

The image inference script runs the model on a folder of images. Feel free to use the same folder as
was used for calibration in the previous step.

Whether running on images or videos, is good to also test it on unseen (non-training,
non-calibration) data to confirm that your model generalizes to other data.

Transfer the compiled artifacts directory to your device. Then run:

```bash
# For images:
#                                    [   ONNX model, modified by compilation    ] [   compiled model subdirectory   ] [   data   ]
sudo python3 run_inference_images.py my_model_data_compiled/last_with_shapes.onnx my_model_data_compiled/tidl_output/ test_images/

# For video:
sudo python3 -m pip3 install tqdm

#                                   [   ONNX model, modified by compilation    ] [   compiled model subdirectory   ] [   data    ]
sudo python3 run_inference_video.py my_model_data_compiled/last_with_shapes.onnx my_model_data_compiled/tidl_output/ test_video.mp4
```

_Note: `sudo` used because TIDl attempts to map `/dev/mem` which requires elevated privileges._

The image script will create a directory called `sample_detections/` with copies of all the input
images. The video script will do the same but with a `sample_detections.avi` file. The model's
detections will be drawn on the images. If nothing appears, try decreasing the confidence threshold
constant at the top of `run_inference_*.py`.

The image script will also print an approximate inference time, in milliseconds, per frame. This
number includes the non-maximum suppression and YOLO output extraction, which we've configured to
happen within the TIDL runtime. This means that the time taken will vary based on how many
detections the model outputs. It may be possible to configure a confidence threshold internally via
the `.prototxt` to further limit the variability of this process. I have not explored this.

The video script will print an average loop time, but note that this includes the video loading,
saving, conversions, drawing, etc., which is the bulk of the time for each loop.

#### C++

I have provided a reimplementation of `run_inference_images.py`, in C++. It uses the same
onnxruntime library as the Python version. See `cpp/run_inference_images/README.md` for build and
usage instructions. It takes the same parameters and returns the same results as the Python version.
The printed logs are slightly different, but the behavior is the same.

TI also has C++ samples that do not use onnxruntime. They refer to it as "tidlrt". I have not tested
this API, but it might provide performance benefits by avoiding the ONNX graph entirely. Their
sample for this API is [here](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/2f17e96c7b5a7a8d5c7e904ec4ea6365c5c25b4c/examples/tidlrt_cpp/classification.cpp)
as of writing.

## Performance

I have mainly been working with the "small" model variant, `yolov5s6`, at 320x320 resolution. The
observed inference times are:

- Python: 7.5ms per frame
- C++: 6.5ms per frame

I used my own model and a sample dataset with an average of around four objects per frame.

These measured times are overall latency as observed by the caller, not maximum throughput. They run
with a batch size of 1.

Note that these measurements _do not_ include any of the pre-processing of the input image
(resizing, re-ordering channels), which would add to these totals. I have not attempted to optimize
the performance of the pre-processing so I did not include it.

There is around 0.5ms of memcpy overhead in the C++ version that could be avoided with negligible
effort.

## Tips

### Important TI documentation

The documentation I've found from TI is sparse and difficult to navigate. However, it does exist.

- TIDL model and tool docs: https://software-dl.ti.com/jacinto7/esd/processor-sdk-rtos-jacinto7/08_04_00_06/exports/docs/tidl_j721e_08_04_00_16/ti_dl/docs/user_guide_html/usergroup0.html

- Compilation library and configuration parameters: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md

### Increase weight decay if needed

TI's quantization quality will degrade when the outputs of a layer vary widely in magnitude. It is
best to have small values in all layers. The easiest way to ensure this is during training, with
_regularization_. _Weight decay_ is the default form of regularization included in the YOLOv5 repo
and should do a good job of this. It is enabled by default using the provided starter
hyperparameters.

Although unnecessary, I have been able to increase weight decay even one or two orders of magnitude
above the default (5e-4) before adversely affecting observed mAP. I was also able to apply weight
decay to convolution _biases_, which by default the codebase doesn't do. If you find your model
performing poorly after quantization, consider increasing weight decay and/or including biases in
the decay.

### Getting logs from TIDL

Sometimes, errors in model execution won't be surfaced to the caller. If the error occurred in code
executing on a coprocessor, you can run the following to see the log output:

```
sudo /opt/vision_apps/vx_app_arm_remote_log.out
```
