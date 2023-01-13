#include <iterator>
#include <experimental/iterator>

#include <opencv2/opencv.hpp>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
// Provides "Ort*_Tidl" functions:
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
// Note that "Ort*_Cpu" versions compatible with host CPUs are available in:
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

// TIDLRT_*
// #include "itidl_rt.h"
// #include "tivx_mem.h"
#include <TI/tivx_mem.h>
// #include <TI/tivx.h>

void run_inference(std::string model_path, std::string artifacts_path);
void enable_tidl_execution_provider(Ort::SessionOptions& ort_session_options, std::string tidl_artifacts_dir_path);


// onnxruntime has C and C++ API flavors; the C API returns status objects which must be checked and freed.
// Ideally, use the C++ flavor. But some of TI's additions require the C flavor.
#define ORT_ASSERT_SUCCESS(expr)                             \
  do {                                                       \
    OrtStatus *onnx_status = (expr);                         \
    if (onnx_status != NULL) {                               \
      const char* msg = Ort::GetApi().GetErrorMessage(onnx_status); \
      std::cout << "onnxruntime operation \"" << #expr << "\" failed: " << msg << std::endl; \
      Ort::GetApi().ReleaseStatus(onnx_status);                     \
      abort();                                               \
    }                                                        \
  } while (0);

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <artifacts_dir> <test_video_path>" << std::endl;
        std::exit(1);
    }

    std::string model_path = argv[1];
    std::string artifacts_dir = argv[2];
    std::string test_video_path = argv[3];

    run_inference(model_path, artifacts_dir);
}

void run_inference(std::string model_path, std::string artifacts_path) {

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "inference_sample");

    Ort::SessionOptions ort_session_options;
    ort_session_options.SetIntraOpNumThreads(4);
    ort_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // TODO: investigate the other options

    enable_tidl_execution_provider(ort_session_options, artifacts_path);

    Ort::AllocatorWithDefaultOptions ort_allocator;

    Ort::Session session(env, model_path.c_str(), ort_session_options);

    // This sample program assumes one input head and one output head, but the library supports arbitrarily many.
    auto input_name = session.GetInputName(0, &*ort_allocator);
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_tensor_dims = input_tensor_info.GetShape();
    ONNXTensorElementDataType input_tensor_type = input_tensor_info.GetElementType();

    auto output_name = session.GetOutputName(0, &*ort_allocator);
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_tensor_dims = output_tensor_info.GetShape();
    ONNXTensorElementDataType output_tensor_type = output_tensor_info.GetElementType();

    std::cout << "Loaded model from path " << model_path << std::endl;
    std::cout << "\tInput \"" << input_name << "\" dims: [ ";
    std::copy(input_tensor_dims.begin(), input_tensor_dims.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << " ], type: " << input_tensor_type << std::endl;
    std::cout << "\tOutput \"" << output_name << "\" dims: [ ";
    std::copy(output_tensor_dims.begin(), output_tensor_dims.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << " ], type: " << output_tensor_type << std::endl;

    // Note: would be straightforward to support other types, but this sample script will assume float
    assert(input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    assert(output_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    assert(input_tensor_dims.size() == 4);
    assert(output_tensor_dims.size() == 4);

    // TODO: verify order of width/height
    int64_t in_batch, in_channels, in_width, in_height;
    std::tie(in_batch, in_channels, in_width, in_height) = std::make_tuple(input_tensor_dims[0], input_tensor_dims[1], input_tensor_dims[2], input_tensor_dims[3]);
    int64_t out_batch, out_channels, out_width, out_height;
    std::tie(out_batch, out_channels, out_width, out_height) = std::make_tuple(output_tensor_dims[0], output_tensor_dims[1], output_tensor_dims[2], output_tensor_dims[3]);

    assert(in_batch == 1);
    assert(out_batch == in_batch);

    static_assert(sizeof(float) == 4);

    vx_uint32 input_tensor_total_bytes = in_channels * in_width * in_height * sizeof(float);
    void *input_tensor_buffer = tivxMemAlloc(input_tensor_total_bytes, tivx_mem_heap_region_e::TIVX_MEM_EXTERNAL);
    assert(input_tensor_buffer != NULL);

    vx_uint32 output_tensor_total_bytes = out_channels * out_width * out_height * sizeof(float);
    void *output_tensor_buffer = tivxMemAlloc(output_tensor_total_bytes, tivx_mem_heap_region_e::TIVX_MEM_EXTERNAL);
    assert(output_tensor_buffer != NULL);

    // Warning: some references online indicate that TI's tivxMemFree function frees _all_ allocated memory, not just the pointer passed.

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, input_tensor_buffer, input_tensor_total_bytes, input_tensor_dims.data(), input_tensor_dims.size(), input_tensor_type);
    Ort::Value output_tensor = Ort::Value::CreateTensor(memory_info, output_tensor_buffer, output_tensor_total_bytes, output_tensor_dims.data(), output_tensor_dims.size(), output_tensor_type);

    Ort::IoBinding binding(session);
    binding.BindInput(input_name, input_tensor);
    binding.BindOutput(output_name, output_tensor);

    auto run_options = Ort::RunOptions();
    run_options.SetRunLogVerbosityLevel(2);

    session.Run(run_options, binding);

    // TODO: free everything
}

void enable_tidl_execution_provider(Ort::SessionOptions& ort_session_options, std::string tidl_artifacts_dir_path) {
    c_api_tidl_options *tidl_options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
    // TODO: free
    assert(tidl_options);

    ORT_ASSERT_SUCCESS(OrtSessionsOptionsSetDefault_Tidl(tidl_options));

    tidl_artifacts_dir_path.copy(tidl_options->artifacts_folder, sizeof(tidl_options->artifacts_folder) - 1);
    tidl_options->artifacts_folder[sizeof(tidl_options->artifacts_folder) - 1] = '\0';

    tidl_options->debug_level = 3;
    // options->max_pre_empt_delay = 6.0;
    // options->priority = 2;

    ORT_ASSERT_SUCCESS(OrtSessionOptionsAppendExecutionProvider_Tidl(ort_session_options, tidl_options));
}
