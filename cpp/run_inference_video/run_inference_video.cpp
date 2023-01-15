#include <iterator>
#include <experimental/iterator>
#include <filesystem>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
// Provides "Ort*_Tidl" functions:
#include <onnxruntime/core/providers/tidl/tidl_provider_factory.h>
// Note that "Ort*_Cpu" versions compatible with host CPUs are available in:
// #include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>

#include <TI/tivx_mem.h>

void run_inference(std::string model_path, std::string artifacts_path, std::string test_images_dir);
void enable_tidl_execution_provider(Ort::SessionOptions& ort_session_options, std::string tidl_artifacts_dir_path);
void load_sample_data(std::string image_dir_path, int input_width, int input_height, std::vector<std::pair<cv::Mat, cv::Mat>>& out_data);


// onnxruntime has C and C++ API flavors; the C API returns status objects which must be checked and freed.
// Ideally, use the C++ flavor. But some of TI's additions require the C flavor.
#define ORT_ASSERT_SUCCESS(expr)                                                                   \
    do {                                                                                           \
        OrtStatus *onnx_status = (expr);                                                           \
        if (onnx_status != NULL) {                                                                 \
            const char* msg = Ort::GetApi().GetErrorMessage(onnx_status);                          \
            std::cout << "onnxruntime operation \"" << #expr << "\" failed: " << msg << std::endl; \
            Ort::GetApi().ReleaseStatus(onnx_status);                                              \
            abort();                                                                               \
        }                                                                                          \
    } while (0);

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <artifacts_dir> <test_images_dir>" << std::endl;
        std::exit(1);
    }

    std::string model_path = argv[1];
    std::string artifacts_dir = argv[2];
    std::string test_images_dir = argv[3];

    run_inference(model_path, artifacts_dir, test_images_dir);
}

void run_inference(std::string model_path, std::string artifacts_path, std::string test_images_dir) {

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
    std::cout << "\tOutput \"" << output_name << "\" dims, reported: [ ";
    std::copy(output_tensor_dims.begin(), output_tensor_dims.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << " ], type: " << output_tensor_type << std::endl;

    // Note: would be straightforward to support other types, but this sample script will assume float
    assert(input_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    assert(output_tensor_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    assert(input_tensor_dims.size() == 4);
    assert(output_tensor_dims.size() >= 2);
    // Last dimension should be the YOLO output vector
    assert(output_tensor_dims[output_tensor_dims.size() - 1] >= 5);

    // TODO: verify order of width/height
    int64_t in_batch, in_channels, in_width, in_height;
    std::tie(in_batch, in_channels, in_width, in_height) = std::make_tuple(input_tensor_dims[0], input_tensor_dims[1], input_tensor_dims[2], input_tensor_dims[3]);
    // int64_t out_batch, out_channels, out_width, out_height;
    // std::tie(out_batch, out_channels, out_width, out_height) = std::make_tuple(output_tensor_dims[0], output_tensor_dims[1], output_tensor_dims[2], output_tensor_dims[3]);

    assert(in_batch == 1);
    // assert(out_batch == in_batch);

    static_assert(sizeof(float) == 4);

    vx_uint32 input_tensor_total_bytes = in_channels * in_width * in_height * sizeof(float);
    void *input_tensor_buffer = tivxMemAlloc(input_tensor_total_bytes, tivx_mem_heap_region_e::TIVX_MEM_EXTERNAL);
    assert(input_tensor_buffer != NULL);

    // vx_uint32 output_tensor_total_bytes = out_channels * out_width * out_height * sizeof(float);
    // void *output_tensor_buffer = tivxMemAlloc(output_tensor_total_bytes, tivx_mem_heap_region_e::TIVX_MEM_EXTERNAL);
    // assert(output_tensor_buffer != NULL);

    // Warning: some references online indicate that TI's tivxMemFree function frees _all_ allocated memory, not just the pointer passed.

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, input_tensor_buffer, input_tensor_total_bytes, input_tensor_dims.data(), input_tensor_dims.size(), input_tensor_type);
    // Ort::Value output_tensor = Ort::Value::CreateTensor(memory_info, output_tensor_buffer, output_tensor_total_bytes, output_tensor_dims.data(), output_tensor_dims.size(), output_tensor_type);

    Ort::IoBinding binding(session);
    binding.BindInput(input_name, input_tensor);
    binding.BindOutput(output_name, memory_info);
    // Note: if output has fixed/known dims, allocate an Ort::Value as with input_tensor and use that instead to pre-allocate the buffer.
    // binding.BindOutput(output_name, output_tensor);

    auto run_options = Ort::RunOptions();
    run_options.SetRunLogVerbosityLevel(2);

    // Run a sample forward pass
    session.Run(run_options, binding);

    auto output = binding.GetOutputValues();
    assert(output.size() == 1);
    
    auto actual_output_tensor_info = output[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> actual_output_tensor_dims = actual_output_tensor_info.GetShape();
    ONNXTensorElementDataType actual_output_tensor_type = actual_output_tensor_info.GetElementType();

    std::cout << "Dummy inference completed:" << std::endl;
    std::cout << "\tOutput \"" << output_name << "\" dims, actual: [ ";
    std::copy(actual_output_tensor_dims.begin(), actual_output_tensor_dims.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << " ], type: " << actual_output_tensor_type << std::endl;

    std::vector<std::pair<cv::Mat, cv::Mat>> image_data;
    load_sample_data(test_images_dir, in_width, in_height, image_data);

    std::cout << "Loaded " << image_data.size() << " images" << std::endl;

    std::cout << "Beginning timing runs..." << std::endl;

    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; i++) {
        for (auto const& [ _, input_data] : image_data) {
            // std::cout << input_data.size() << std::endl;
            // assert(input_data.size().width == in_width);
            // assert(input_data.size().height == in_height);
            // assert(input_data.channels() == in_channels);

            assert(input_data.total() * sizeof(float) == input_tensor_total_bytes);
            // auto copy_start_time = std::chrono::steady_clock::now();
            // TODO: memcpy takes 0.5ms+, could be avoided
            memcpy(input_tensor_buffer, input_data.data, input_data.total() * sizeof(float));
            // TODO: std::copy took 8ms+, very slow
            // std::copy(input_data.begin<float>(), input_data.end<float>(), (float*)input_tensor_buffer);
            // auto copy_end_time = std::chrono::steady_clock::now();
            // auto copy_total_execution = copy_end_time - copy_start_time;
            // std::cout << std::chrono::duration <double, std::micro> (copy_total_execution).count() << " us" << std::endl;

            session.Run(run_options, binding);
        }
    }
    auto end_time = std::chrono::steady_clock::now();
    auto total_execution = end_time - start_time;
    auto per_frame = total_execution / (200 * image_data.size());
    std::cout << std::chrono::duration <double, std::milli> (per_frame).count() << " ms" << std::endl;


    // TODO: free everything
}

void enable_tidl_execution_provider(Ort::SessionOptions& ort_session_options, std::string tidl_artifacts_dir_path) {
    c_api_tidl_options *tidl_options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
    // TODO: free
    assert(tidl_options);

    ORT_ASSERT_SUCCESS(OrtSessionsOptionsSetDefault_Tidl(tidl_options));

    tidl_artifacts_dir_path.copy(tidl_options->artifacts_folder, sizeof(tidl_options->artifacts_folder) - 1);
    tidl_options->artifacts_folder[sizeof(tidl_options->artifacts_folder) - 1] = '\0';

    // tidl_options->debug_level = 3;
    // options->max_pre_empt_delay = 6.0;
    // options->priority = 2;

    ORT_ASSERT_SUCCESS(OrtSessionOptionsAppendExecutionProvider_Tidl(ort_session_options, tidl_options));
}

void load_sample_data(std::string image_dir_path, int input_width, int input_height, std::vector<std::pair<cv::Mat, cv::Mat>>& out_data) {
    for (auto const& dir_entry : std::filesystem::directory_iterator{image_dir_path}) 
    {
        auto image = cv::imread(dir_entry.path());

        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size2i(input_width, input_height));

        // cv::cvtColor(input_tensor, input_tensor, cv::ColorConversionCodes::COLOR_BGR2RGB);
        // input_tensor.convertTo(input_tensor, CV_32FC3, 1/255.0);

        auto input_tensor = cv::dnn::blobFromImage(
            resized_image,
            1/255.0, // scalefactor
            cv::Size2i(input_width, input_height),
            0, // mean
            true, // swapRB
            false, // crop
            CV_32F
        );

        // TODO: verify resized to square
        // TODO: might be h/w swapped

        out_data.push_back(std::make_pair(image, input_tensor));
    }
}
