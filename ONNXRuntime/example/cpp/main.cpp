#include "base.hpp"
#include "onnxruntime_cxx_api.h"
#include <cstring>
#include <filesystem>

#ifdef USE_TENSORRT
#include <tensorrt_provider_options.h>
#endif

#ifdef USE_OPENVINO
#include <openvino_provider_factory.h>
#endif

#ifdef USE_COREML
#include <coreml_provider_factory.h>
#endif

#if defined(__linux__)
#define PLATFORM "Linux"
#elif defined(__APPLE__)
#define PLATFORM "macOS"
#else
#error "Unknown"
#endif

namespace fs = std::filesystem;
static auto rootDir = fs::current_path().parent_path().string();

class ONNXRuntime : public Base {
  public:
    ONNXRuntime(std::string const &ep) {
        // ORT_LOGGING_LEVEL_WARNING, ORT_LOGGING_LEVEL_INFO, ORT_LOGGING_LEVEL_VERBOSE
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "LOG");
        // Create an ONNX Runtime session and load the model into it
        Ort::SessionOptions options;
        setup_provider(options, ep);
        try {
            session = Ort::Session(env, (rootDir + "/Assets/yolov8n.onnx").c_str(), options);
        } catch (Ort::Exception const &e) {
            throw std::runtime_error(e.what());
        }

        Ort::AllocatorWithDefaultOptions allocator;
        // Get the input info
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            // Get the input name (e.g. input_0, input_1, ...)
            inputNamePtrs.emplace_back(session.GetInputNameAllocated(i, allocator));
            inputNames.emplace_back(inputNamePtrs[i].get());
            // Get the input dimension (e.g. [1, 3, 224, 224], [1, 1, 1000] ...)
            inputDims.emplace_back(session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            // Create an input tensor
            inputTensors.emplace_back(
                Ort::Value::CreateTensor<float>(allocator, inputDims[i].data(), inputDims[i].size()));
        }

        // Get the output info
        for (size_t i = 0; i < session.GetOutputCount(); ++i) {
            // Get the output name (e.g. output_0, output_1, ...)
            outputNamePtrs.emplace_back(session.GetOutputNameAllocated(i, allocator));
            outputNames.emplace_back(outputNamePtrs[i].get());
            // Get the output dimension (e.g. [1, 3, 224, 224], [1, 1, 1000] ...)
            outputDims.emplace_back(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            // Create an output tensor
            outputTensors.emplace_back(
                Ort::Value::CreateTensor<float>(allocator, outputDims[i].data(), outputDims[i].size()));
        }
    }

  private:
    void setup_provider(Ort::SessionOptions &options, std::string const &ep) {
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.DisableProfiling();

        if (ep == "TensorrtExecutionProvider") {
#ifdef USE_TENSORRT
            auto root_dir = fs::current_path().parent_path().parent_path().string();
            const auto &api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2 *tensorrt_options;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
            std::vector<const char *> option_keys = {"device_id", "trt_max_workspace_size",
                                                     "trt_engine_cache_enable", "trt_engine_cache_path"};
            std::vector<const char *> option_values = {"0", "3221225472", "1",
                                                       (root_dir + "/Assets").c_str()};
            Ort::ThrowOnError(
                api.SessionOptionsAppendExecutionProvider_TensorRT_V2(options, tensorrt_options));
#else
            std::cout << "TensorRT is not supported." << std::endl;
#endif
        } else if (ep == "CUDAExecutionProvider") {
#ifdef USE_CUDA
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
#else
            std::cout << "CUDA is not supported." << std::endl;
#endif
        } else if (ep == "OpenVINOExecutionProvider") {
#ifdef USE_OPENVINO
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(options, "CPU"));
#else
            std::cout << "OpenVINO is not supported." << std::endl;
#endif
        } else if (ep == "CoreMLExecutionProvider") {
#ifdef USE_COREML
            uint32_t coreml_flags = 0;
            coreml_flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(options, coreml_flags));
#else
            std::cout << "CoreML is not supported." << std::endl;
#endif
        } else if (ep == "CPUExecutionProvider") {
            // CPUExecutionProvider is the default provider
        } else {
            std::cout << "Provider not found: " << ep << std::endl;
        }
    }

    void infer() override {
        try {
            // Fill the input tensor
            float *inputTensorValues = inputTensors[0].GetTensorMutableData<float>();
            // Flatten the input image [1, 3, h, w] => [1, 3 * h * w]
            cvImage = cvImage.reshape(1, 1);
            std::memcpy(inputTensorValues, cvImage.data, cvImage.total() * sizeof(float));

            // Run the inference
            session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputTensors.size(),
                        outputNames.data(), outputTensors.data(), outputTensors.size());

            // Copy output from the outputTensors [1, 84, 8400]
            float *outputTensorValues = outputTensors[0].GetTensorMutableData<float>();
            std::vector<int> outputDim;
            for (auto e : outputDims[0])
                outputDim.push_back(static_cast<int>(e));
            cv::Mat output(outputDim.size(), outputDim.data(), CV_32F, outputTensorValues);
            outputs.clear();
            outputs.emplace_back(output);

        } catch (Ort::Exception const &e) {
            throw std::runtime_error(e.what());
        }
    }

  private:
    Ort::Env env{nullptr}; // https://github.com/microsoft/onnxruntime/issues/5320#issuecomment-700995952
    Ort::Session session{nullptr};

    /*
    to keep the allocated memory => input and output names will be valid???
    another solution without keeping the allocated memory:
    char * buf = new char[1024];
    strcpy(buf, session.GetInputNameAllocated(i, allocator).get());
    */
    std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
    std::vector<Ort::AllocatedStringPtr> outputNamePtrs;

    std::vector<char const *> inputNames;
    std::vector<char const *> outputNames;

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    std::vector<std::vector<int64_t>> inputDims;
    std::vector<std::vector<int64_t>> outputDims;
};

int main() {
    std::string video_path = rootDir + "/Assets/video.mp4";

    auto providers = Ort::GetAvailableProviders();
    for (auto const &ep : providers) {
        std::cout << "Using " << ep << std::endl;
        std::string save_path = rootDir + "/Results/" + PLATFORM + "-ONNXRuntime-Cpp-" + ep + ".mp4";
        ONNXRuntime session(ep);
        session.run(video_path, save_path, false);
    }

    return 0;
}