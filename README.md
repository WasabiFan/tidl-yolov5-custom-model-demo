# TIDL custom model experiments

Work-in-progress scripts and tools to get a custom model compiled with TIDL and running on the
TDA4VM/BeagleBone AI-64.

Included Docker image is intended to be used for compilation. This repo also includes a vscode
configuration for using the Docker image as a dev container. **As of writing, TIDL has extreme
memory errors which mean it crashes and misbehaves under Docker. See [here](https://e2e.ti.com/support/processors-group/processors/f/processors-forum/1160791/tda4vm-bus-error-when-running-forward-pass-through-model-with-tidl_tools-on-pc/4369467)
for discussion.**

The scripts currently use random noise as input into all models, and as sucn don't demonstrate
useful behavior in themselves. However, I hope they are a helpful reference as minimal compilation
and inference steps to be applied to your own models.

Note that this repo runs inference in Python. C++ or another language with tools that give better
runtime performance might be a wiser choice for the TDA4VM platform. So far I have only tried
Python.

## Usage

Due to the above limitation, you will need an x86 Linux installation (optionally a VM) with the
setup steps from the included Dockerfile applied manually.

On the x86 PC:

```
python3.6 dump_sample_network.py sample_resnet.onnx
python3.6 compile_model.py sample_resnet.onnx resnet_artifacts
```

Transfer the `resnet_artifacts` directory to a TDA4VM device (currently only tested on the
BeagleBone AI-64) and run:

```
sudo python3 run_inference.py resnet_artifacts/sample_resnet_with_shapes.onnx resnet_artifacts/tidl_output
```

_Note: `sudo` used because TIDl attempts to map `/dev/mem` which requires elevated privileges._

## Tip: debugging errors from coprocessors

Run `/opt/vision_apps/vx_app_arm_remote_log.out` with root permissions to see logs from the
remote TIDL cores.
