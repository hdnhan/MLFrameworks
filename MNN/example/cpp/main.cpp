#include "base.hpp"
#include <MNN/Interpreter.hpp>
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <string>

#if defined(__linux__)
#define PLATFORM "Linux"
#elif defined(__APPLE__)
#define PLATFORM "macOS"
#else
#error "Unknown"
#endif

namespace fs = std::filesystem;
static auto rootDir = fs::current_path().parent_path().string();

// https://www.yuque.com/mnn/en/create_session
class CppMNN : public Base {
  public:
    CppMNN() {
        // Load the network
        std::string modelPath = rootDir + "/Assets/yolov8n.mnn";
        spdlog::info("Model path: {}", modelPath);
        interpreter = MNN::Interpreter::createFromFile(modelPath.c_str());
        if (!interpreter) {
            spdlog::error("Failed to load model");
            return;
        }
        scheduleConfig = new MNN::ScheduleConfig;
        scheduleConfig->type = MNN_FORWARD_METAL;     // Metal
        scheduleConfig->backupType = MNN_FORWARD_CPU; // CPU
        session = interpreter->createSession(*scheduleConfig);
        if (!session) {
            spdlog::error("Failed to create session");
            return;
        }

        inputTensor = interpreter->getSessionInput(session, NULL);
        if (!inputTensor) {
            spdlog::error("Failed to get input tensor");
            return;
        }
        inputHostTensor = new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE);

        outputTensor = interpreter->getSessionOutput(session, NULL);
        if (!outputTensor) {
            spdlog::error("Failed to get output tensor");
            return;
        }
        outputHostTensor = new MNN::Tensor(outputTensor, MNN::Tensor::CAFFE);
    }

    ~CppMNN() {
        if (session)
            interpreter->releaseSession(session);
        if (interpreter)
            delete interpreter;
        if (scheduleConfig)
            delete scheduleConfig;
        if (inputHostTensor)
            delete inputHostTensor;
        if (outputHostTensor)
            delete outputHostTensor;
    }

  private:
    void infer() override {
        // cv::Mat cvImage [1, 3, h, w]
        auto inputSize = inputTensor->size(); // 1 * 3 * h * w * sizeof(float)
        // Populate the input tensor (if on CPU, don't need create a new tensor and copy real data to it)
        // memcpy(inputTensor->host<float>(), cvImage.data, inputSize); // Comment two lines below
        memcpy(inputHostTensor->host<float>(), cvImage.data, inputSize);
        inputTensor->copyFromHostTensor(inputHostTensor);
        // Run the model
        interpreter->runSession(session);
        // Get the output tensor (if on CPU, don't need create a new tensor and copy real result to it)
        auto outputShape = outputTensor->shape(); // [1, 84, 8400]
        // auto outputData = outputTensor->host<float>(); // Comment two lines below
        outputTensor->copyToHostTensor(outputHostTensor);
        auto outputData = outputHostTensor->host<float>();
        cv::Mat output(outputShape.size(), outputShape.data(), CV_32F, outputData);

        outputs.clear();
        outputs.emplace_back(output);
    }

  private:
    MNN::Interpreter *interpreter;
    MNN::Session *session;
    MNN::ScheduleConfig *scheduleConfig;

    // Input and output
    MNN::Tensor *inputTensor, *outputTensor;
    MNN::Tensor *inputHostTensor, *outputHostTensor;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/main", "MNN C++ Example");
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

#if defined(__APPLE__)
    {
        spdlog::info("Using Metal");
        std::string save_path = save_dir + "/" + PLATFORM + "-MNN-Cpp.mp4";
        CppMNN session;
        session.run(video_path, save_path);
    }
#endif
    return 0;
}