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
void print_sample_output_metadata(const char *output_name, Ort::Value& output);
void load_sample_data(std::string image_dir_path, int input_width, int input_height, std::map<std::string, std::pair<cv::Mat, cv::Mat>>& out_data);
void render_detections(uint64_t detection_width, uint64_t detection_height, cv::Mat& out_image, Ort::Value& tensor_output);

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
    // TODO: session has many options, take a look at what's available

    enable_tidl_execution_provider(ort_session_options, artifacts_path);

    Ort::Session session(env, model_path.c_str(), ort_session_options);

    Ort::AllocatorWithDefaultOptions ort_allocator;

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
    assert(output_tensor_dims[output_tensor_dims.size() - 1] == 6);

    int64_t in_batch, in_channels, in_width, in_height;
    std::tie(in_batch, in_channels, in_width, in_height) = std::make_tuple(input_tensor_dims[0], input_tensor_dims[1], input_tensor_dims[2], input_tensor_dims[3]);

    assert(in_batch == 1);

    static_assert(sizeof(float) == 4);

    vx_uint32 input_tensor_total_bytes = in_channels * in_width * in_height * sizeof(float);
    void *input_tensor_buffer = tivxMemAlloc(input_tensor_total_bytes, tivx_mem_heap_region_e::TIVX_MEM_EXTERNAL);
    assert(input_tensor_buffer != NULL);
    memset(input_tensor_buffer, 0, input_tensor_total_bytes);

    // Warning: some references online indicate that TI's tivxMemFree function frees _all_ allocated memory, not just the pointer passed.

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info, input_tensor_buffer, input_tensor_total_bytes, input_tensor_dims.data(), input_tensor_dims.size(), input_tensor_type);

    Ort::IoBinding binding(session);
    binding.BindInput(input_name, input_tensor);
    binding.BindOutput(output_name, memory_info);
    // Note: if output has fixed/known dims, allocate an Ort::Value as with input_tensor and use that instead to pre-allocate the buffer.
    // binding.BindOutput(output_name, output_tensor);

    auto run_options = Ort::RunOptions();
    run_options.SetRunLogVerbosityLevel(2);


    // Run a sample forward pass to show the "actual" output dims (for variable-length dims)
    session.Run(run_options, binding);
    auto output = binding.GetOutputValues();
    assert(output.size() == 1);
    print_sample_output_metadata(output_name, output[0]);

    // Load data from provided directory
    std::map<std::string, std::pair<cv::Mat, cv::Mat>> image_data;
    load_sample_data(test_images_dir, in_width, in_height, image_data);
    std::cout << "Loaded " << image_data.size() << " images" << std::endl;

    std::cout << "Beginning timing runs..." << std::endl;

    // Compute emperical average inference latency
    auto start_time = std::chrono::steady_clock::now();
    const int NUM_TIMING_REPS = 20;
    for (int i = 0; i < NUM_TIMING_REPS; i++) {
        for (auto const& [ _, value ] : image_data) {
            auto const& [ __, input_data] = value;

            assert(input_data.total() * sizeof(float) == input_tensor_total_bytes);
            // TODO: memcpy takes 0.5ms+, could be avoided or optimized
            memcpy(input_tensor_buffer, input_data.data, input_data.total() * sizeof(float));
            // TODO: std::copy took 8ms+, very slow
            // std::copy(input_data.begin<float>(), input_data.end<float>(), (float*)input_tensor_buffer);

            session.Run(run_options, binding);
            auto output = binding.GetOutputValues();
            assert(output.size() == 1);
        }
    }
    auto end_time = std::chrono::steady_clock::now();
    auto total_execution = end_time - start_time;
    auto per_frame = total_execution / (NUM_TIMING_REPS * image_data.size());
    std::cout << "Inference time per frame: " << std::chrono::duration <double, std::milli> (per_frame).count() << " ms" << std::endl;

    std::cout << "Rendering sample detections..." << std::endl;

    std::filesystem::path out_dir = { "sample_detections" };
    std::filesystem::create_directory(out_dir);

    for (auto const& [ filename, value ] : image_data) {
        auto const& [ original_image, input_data] = value;

        assert(input_data.total() * sizeof(float) == input_tensor_total_bytes);
        memcpy(input_tensor_buffer, input_data.data, input_data.total() * sizeof(float));

        session.Run(run_options, binding);

        auto ort_output = binding.GetOutputValues();
        assert(ort_output.size() == 1);

        cv::Mat rendered_detections = original_image.clone();
        render_detections(in_width, in_height, rendered_detections, ort_output[0]);

        cv::imwrite(out_dir / filename, rendered_detections);
    }

    std::cout << "Sample detections saved to: " << out_dir.relative_path() << std::endl;

    tivxMemFree(input_tensor_buffer, input_tensor_total_bytes, tivx_mem_heap_region_e::TIVX_MEM_EXTERNAL);
    ort_allocator.Free(input_name);
    ort_allocator.Free(output_name);

    // TODO: still some memory leaks from Session reported by Valgrind, unclear how to resolve them.
    // The below is likely redundant as the Session destructor should implicitly release.
    session.release();
}

void enable_tidl_execution_provider(Ort::SessionOptions& ort_session_options, std::string tidl_artifacts_dir_path) {
    c_api_tidl_options *tidl_options = (c_api_tidl_options *)malloc(sizeof(c_api_tidl_options));
    assert(tidl_options);

    ORT_ASSERT_SUCCESS(OrtSessionsOptionsSetDefault_Tidl(tidl_options));

    tidl_artifacts_dir_path.copy(tidl_options->artifacts_folder, sizeof(tidl_options->artifacts_folder) - 1);
    tidl_options->artifacts_folder[sizeof(tidl_options->artifacts_folder) - 1] = '\0';

    // tidl_options->debug_level = 3;
    // tidl_options->max_pre_empt_delay = 6.0;
    // tidl_options->priority = 2;

    ORT_ASSERT_SUCCESS(OrtSessionOptionsAppendExecutionProvider_Tidl(ort_session_options, tidl_options));
    free(tidl_options);
}

void print_sample_output_metadata(const char *output_name, Ort::Value& output) {
    auto actual_output_tensor_info = output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> actual_output_tensor_dims = actual_output_tensor_info.GetShape();
    ONNXTensorElementDataType actual_output_tensor_type = actual_output_tensor_info.GetElementType();

    std::cout << "Dummy inference completed:" << std::endl;
    std::cout << "\tOutput \"" << output_name << "\" dims, actual: [ ";
    std::copy(actual_output_tensor_dims.begin(), actual_output_tensor_dims.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
    std::cout << " ], type: " << actual_output_tensor_type << std::endl;
}

void load_sample_data(std::string image_dir_path, int input_width, int input_height, std::map<std::string, std::pair<cv::Mat, cv::Mat>>& out_data) {
    for (auto const& dir_entry : std::filesystem::directory_iterator{image_dir_path}) 
    {
        auto image = cv::imread(dir_entry.path());

        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size2i(input_width, input_height));

        auto input_tensor = cv::dnn::blobFromImage(
            resized_image,
            1/255.0, // scalefactor
            cv::Size2i(input_width, input_height),
            0, // mean
            true, // swapRB
            false, // crop
            CV_32F
        );

        out_data.insert_or_assign(dir_entry.path().filename().string(), std::make_pair(image, input_tensor));
    }
}

void render_detections(uint64_t detection_width, uint64_t detection_height, cv::Mat& out_image, Ort::Value& tensor_output) {
    auto tensor_output_info = tensor_output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> tensor_output_dims = tensor_output_info.GetShape();
    ONNXTensorElementDataType tensor_output_type = tensor_output_info.GetElementType();

    assert(tensor_output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    assert(tensor_output_dims.size() == 4);
    int64_t out_batch, out_channels, out_num_entries, out_features;
    std::tie(out_batch, out_channels, out_num_entries, out_features) = std::make_tuple(tensor_output_dims[0], tensor_output_dims[1], tensor_output_dims[2], tensor_output_dims[3]);

    assert(out_batch == 1);
    assert(out_channels == 1);
    // x1, y1, x2, y2, objectness, class_idx
    assert(out_features == 6);

    auto *output_buf = tensor_output.GetTensorData<float>();

    const float CONFIDENCE_THRESHOLD = 0.3;
    for (int entry_idx = 0; entry_idx < out_num_entries; entry_idx++) {
        std::vector<float> features;
        for (int feature_idx = 0; feature_idx < out_features; feature_idx++) {
            features.push_back(output_buf[entry_idx * out_features + feature_idx]);
        }

        if (features[4] < CONFIDENCE_THRESHOLD) {
            continue;
        }

        float box_x1 = features[0] / detection_width * out_image.size().width;
        float box_y1 = features[1] / detection_height * out_image.size().height;
        float box_x2 = features[2] / detection_width * out_image.size().width;
        float box_y2 = features[3] / detection_height * out_image.size().height;

        int class_idx_int = (int)features[5];

        // TODO: if you have more than two classes, modify the below
        // Color channels in BGR order
        cv::Scalar color = class_idx_int == 0 ? cv::Scalar{ 50, 50, 255 }
                            : class_idx_int == 1 ? cv::Scalar{ 255, 50, 50 }
                            : cv::Scalar{ 255, 255, 255 };

        cv::rectangle(
            out_image,
            cv::Point2d(box_x1, box_y1),
            cv::Point2d(box_x2, box_y2),
            color,
            3
        );
    }
}