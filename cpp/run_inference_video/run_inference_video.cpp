#include <iterator>
#include <experimental/iterator>

#include <opencv2/opencv.hpp>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
// Provides "Ort*_Tidl" functions:
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
// Note that "Ort*_Cpu" versions compatible with host CPUs are available in:
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

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

    // Ort::AllocatorWithDefaultOptions allocator;

    Ort::Session session(env, model_path.c_str(), ort_session_options);

    Ort::TypeInfo type_info = session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_tensor_dims = tensor_info.GetShape();
    ONNXTensorElementDataType input_tensor_type = tensor_info.GetElementType();

    std::cout << "Loaded model. Input dims: [ ";
    std::copy(input_tensor_dims.begin(), input_tensor_dims.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << " ], type: " << input_tensor_type << std::endl;
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
