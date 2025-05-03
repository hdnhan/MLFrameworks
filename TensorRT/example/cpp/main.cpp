#include "base.hpp"

#include <cuda_runtime_api.h>

#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

namespace fs = std::filesystem;
static auto rootDir = fs::current_path().parent_path().string();

// Class to extend TensorRT ILogger
class Logger : public nvinfer1::ILogger {
  public:
    Severity mSeverity = Severity::kINFO;
    Logger(Severity severity = Severity::kINFO) : mSeverity(severity) {};

    void error(std::string const &msg) { return log(Severity::kERROR, msg.c_str()); }
    void warning(std::string const &msg) { return log(Severity::kWARNING, msg.c_str()); }
    void info(std::string const &msg) { return log(Severity::kINFO, msg.c_str()); }
    void verbose(std::string const &msg) { return log(Severity::kVERBOSE, msg.c_str()); }

  private:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity > mSeverity)
            return;
        switch (severity) {
        case Severity::kERROR:
            std::cerr << "[ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "[WARNING] " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "[INFO] " << msg << std::endl;
            break;
        case Severity::kVERBOSE:
            std::cout << "[VERBOSE] " << msg << std::endl;
            break;
        default:
            break;
        }
    }
};

class TensorRT : public Base {
  public:
    TensorRT(std::string const &modelName, std::string const &dtype = "f32") : dtype(dtype) {
        mLogger = std::make_unique<Logger>();

        f_onnx = modelName + ".onnx";
        f_engine = modelName + "-" + dtype + ".engine";

        initLibNvInferPlugins(nullptr, "");

        if (!std::filesystem::exists(f_engine)) {
            if (!std::filesystem::exists(f_onnx))
                throw std::runtime_error("Error, onnx file not found at " + f_onnx);
            build();
        }
        if (!mEngine)
            load();

        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        bindings.resize(mEngine->getNbIOTensors());
        for (int i = 0; i < mEngine->getNbIOTensors(); ++i) {
            auto name = mEngine->getIOTensorName(i);
            nvinfer1::Dims dims = mEngine->getTensorShape(name);
            auto size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
            cudaMalloc(&bindings[i], size * sizeof(float));
            if (mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                mInputIndices.emplace_back(i);
                mInputDims.emplace_back(dims);
            } else {
                mOutputIndices.emplace_back(i);
                mOutputDims.emplace_back(dims);
                mOutputs.emplace_back(std::vector<float>(size));
            }
        }
    }

    ~TensorRT() {
        for (void *binding : bindings)
            cudaFree(binding);
    }

  private:
    void build() {
        std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(*mLogger)};
        std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1UL << 30); // 1GiB
        if (builder->platformHasFastFp16() && dtype == "f16")
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        if (builder->platformHasFastInt8() && dtype == "i8") {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }

        auto const flag = 1U << int(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
        std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flag)};
        std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, *mLogger)};

        mLogger->info("Parsing onnx file from " + f_onnx);
        if (!parser->parseFromFile(f_onnx.c_str(), static_cast<int>(mLogger->mSeverity)))
            throw std::runtime_error("Error parsing onnx file.");

        mLogger->info("Building plan from network and config");
        std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*mLogger));
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(plan->data(), plan->size()));

        mLogger->info("Saving serialized engine to " + f_engine);
        std::ofstream file;
        file.open(f_engine, std::ios::binary | std::ios::out);
        file.write((const char *)plan->data(), plan->size());
        file.close();
    }

    void load() {
        std::ifstream file(f_engine, std::ios::binary | std::ios::ate);
        mLogger->info("Loading engine at " + f_engine);

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size))
            throw std::runtime_error("Error reading engine file.");

        mRuntime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*mLogger));
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(buffer.data(), buffer.size()));
    }

    void infer() override {
        // Flatten the input image [1, 3, h, w] => [1, 3 * h * w]
        cvImage = cvImage.reshape(1, 1);
        // Fill the input tensor
        cudaMemcpy(bindings[0], cvImage.data, cvImage.total() * sizeof(float), cudaMemcpyHostToDevice);

        // Run the inference
        if (!mContext->executeV2(bindings.data()))
            throw std::runtime_error("Error running inference.");

        for (size_t i = 0; i < mOutputIndices.size(); ++i) {
            cudaMemcpy(mOutputs[i].data(), bindings[mOutputIndices[i]], mOutputs[i].size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }

        // Copy output from the outputTensors [1, 84, 8400]
        std::vector<int> outputDim;
        for (int i = 0; i < mOutputDims[0].nbDims; ++i)
            outputDim.emplace_back(mOutputDims[0].d[i]);
        cv::Mat output(outputDim.size(), outputDim.data(), CV_32F, mOutputs[0].data());
        outputs.clear();
        outputs.emplace_back(output);
    }

  private:
    std::string f_onnx, f_engine;
    std::string dtype;

    std::unique_ptr<Logger> mLogger;
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

    std::vector<void *> bindings;
    std::vector<int> mInputIndices, mOutputIndices;
    std::vector<nvinfer1::Dims> mInputDims, mOutputDims;
    std::vector<std::vector<float>> mOutputs;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/main", "TensorRT C++ Example");
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
    std::string save_path = save_dir + "/Linux-TensorRT-Cpp.mp4";

    // Load the network
    TensorRT session(rootDir + "/Assets/yolov8n");
    session.run(video_path, save_path);
    return 0;
}