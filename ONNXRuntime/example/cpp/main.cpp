#include "base.hpp"
#include "onnxruntime_cxx_api.h"
#include <cstdint>
#include <cstring>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
        // options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
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
            spdlog::warn("TensorRT is not supported.");
#endif
        } else if (ep == "CUDAExecutionProvider") {
#ifdef USE_CUDA
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
#else
            spdlog::warn("CUDA is not supported.");
#endif
        } else if (ep == "OpenVINOExecutionProvider") {
#ifdef USE_OPENVINO
            std::unordered_map<std::string, std::string> op;
            op["device_type"] = "CPU";
            options.AppendExecutionProvider_OpenVINO_V2(op);
#else
            spdlog::warn("OpenVINO is not supported.");
#endif
        } else if (ep == "CoreMLExecutionProvider") {
#ifdef USE_COREML
            // https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
            std::unordered_map<std::string, std::string> provider_options;
            provider_options["ModelFormat"] = "NeuralNetwork";
            provider_options["MLComputeUnits"] = "CPUAndNeuralEngine";
            provider_options["RequireStaticInputShapes"] = "1";
            provider_options["EnableOnSubgraphs"] = "0";
            provider_options["AllowLowPrecisionAccumulationOnGPU"] = "1";
            provider_options["SpecializationStrategy"] = "FastPrediction";
            // provider_options["ModelCacheDirectory"] = "/tmp/__cache__";
            options.AppendExecutionProvider("CoreML", provider_options);
#else
            std::cout << "CoreML is not supported." << std::endl;
#endif
        } else if (ep == "CPUExecutionProvider") {
            // CPUExecutionProvider is the default provider
        } else {
            spdlog::warn("Provider not found: {}", ep);
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

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/main", "ONNXRuntime C++ Example");
    options.add_options()("h,help", "Show help")(
        "v,video", "Path to video file",
        cxxopts::value<std::string>()->default_value(rootDir + "/Assets/video.mp4"))(
        "s,save", "Directory to save output video",
        cxxopts::value<std::string>()->default_value(rootDir + "/Results"));
    auto config = options.parse(argc, argv);
    if (config.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    spdlog::cfg::load_env_levels();
    // spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%x %X.%e] [%^%l%$] %v");

    std::string video_path = config["video"].as<std::string>();
    std::string save_dir = config["save"].as<std::string>();
    spdlog::info("Video path: {}", video_path);

    auto providers = Ort::GetAvailableProviders();
    for (auto const &ep : providers) {
        spdlog::info("Using {}", ep);
        std::string save_path = save_dir + "/" + PLATFORM + "-ONNXRuntime-Cpp-" + ep + ".mp4";
        ONNXRuntime session(ep);
        session.run(video_path, save_path);
    }

    return 0;
}