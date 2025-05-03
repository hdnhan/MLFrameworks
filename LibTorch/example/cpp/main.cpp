#include "base.hpp"
#include <cxxopts.hpp>
#include <filesystem>
#include <iostream>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#if defined(__linux__)
#define PLATFORM "Linux"
#elif defined(__APPLE__)
#define PLATFORM "macOS"
#else
#error "Unknown"
#endif

namespace fs = std::filesystem;
static auto rootDir = fs::current_path().parent_path().string();

class LibTorch : public Base {
  public:
    LibTorch(torch::Device device) : device(device) {
        // Load the network
        std::string modelPath = rootDir + "/Assets/yolov8n.torchscript";
        spdlog::info("Model path: {}", modelPath);
        model = torch::jit::load(modelPath);
        model.to(device);
        model.eval();
        spdlog::info("Model loaded");
    }

  private:
    void infer() override {
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        auto input = torch::from_blob(cvImage.data,
                                      {
                                          cvImage.size[0],
                                          cvImage.size[1],
                                          cvImage.size[2],
                                          cvImage.size[3],
                                      },
                                      options);
        auto output = model.forward({input}).toTensor().to(torch::kCPU);
        std::vector<int> outputDim;
        for (auto e : output.sizes())
            outputDim.push_back(static_cast<int>(e));
        cv::Mat outputMat(outputDim.size(), outputDim.data(), CV_32F, output.data_ptr<float>());
        outputs.clear();
        outputs.emplace_back(outputMat.clone());
    }

  private:
    torch::jit::script::Module model;
    torch::Device device = torch::kCPU;
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("./build/main", "LibTorch C++ Example");
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
    if (torch::cuda::is_available()) {
        spdlog::info("Using CUDA");
        std::string save_path = save_dir + "/" + PLATFORM + "-LibTorch-Cpp-CUDA.mp4";
        LibTorch session(torch::kCUDA);
        session.run(video_path, save_path);
    }
#endif

#if defined(__linux__) || defined(__APPLE__)
    {
        spdlog::info("Using CPU");
        std::string save_path = save_dir + "/" + PLATFORM + "-LibTorch-Cpp-CPU.mp4";
        LibTorch session(torch::kCPU);
        session.run(video_path, save_path);
    }
#endif
    return 0;
}