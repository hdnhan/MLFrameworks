#include "base.hpp"
#include <filesystem>

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
        model = cv::dnn::readNet(rootDir + "/Assets/yolov8n.onnx");

        if (useCUDA) {
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

int main() {
    std::string video_path = rootDir + "/Assets/video.mp4";

#if defined(__linux__)
    if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
        std::cout << "Using CUDA" << std::endl;
        std::string save_path = rootDir + "/Results/" + PLATFORM + "-OpenCV-Cpp-CUDA.mp4";
        OpenCV session(true);
        session.run(video_path, save_path, false);
    }
#endif

#if defined(__linux__) || defined(__APPLE__)
    {
        std::cout << "Using CPU" << std::endl;
        std::string save_path = rootDir + "/Results/" + PLATFORM + "-OpenCV-Cpp-CPU.mp4";
        OpenCV session(false);
        session.run(video_path, save_path, false);
    }
#endif
    return 0;
}