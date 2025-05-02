#include "base.hpp"
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <opencv2/dnn/dnn.hpp>
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

class OpenCV : public Base {
  public:
    OpenCV(bool useCUDA = false) {
        // Load the network
        std::string modelPath = rootDir + "/Assets/yolov8n.onnx";
        spdlog::info("Model path: {}", modelPath);
        model = cv::dnn::readNet(modelPath);

        if (useCUDA) {
            spdlog::debug("Setting CUDA backend and target");
            model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
    }

  private:
    void infer() override {
        // Set the input to the network
        model.setInput(cvImage);
        // Run the forward pass to get output from the output layers
        outputs.clear();
        auto layers = model.getUnconnectedOutLayersNames();
        model.forward(outputs, layers);
    }

  private:
    cv::dnn::Net model;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/main", "OpenCV C++ Example");
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

#if defined(__linux__)
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        spdlog::info("Using CUDA");
        std::string save_path = save_dir + "/" + PLATFORM + "-OpenCV-Cpp-CUDA.mp4";
        OpenCV session(true);
        session.run(video_path, save_path);
    }
#endif

#if defined(__linux__) || defined(__APPLE__)
    {
        spdlog::info("Using CPU");
        std::string save_path = save_dir + "/" + PLATFORM + "-OpenCV-Cpp-CPU.mp4";
        OpenCV session(false);
        session.run(video_path, save_path);
    }
#endif
    return 0;
}