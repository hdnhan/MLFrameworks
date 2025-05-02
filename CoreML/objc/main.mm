#include "base.hpp"
#include <CoreML/CoreML.h>
#include <cxxopts.hpp>
#include <filesystem>
#include <spdlog/cfg/env.h>
#include <spdlog/common.h>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

static auto rootDir = fs::current_path().parent_path().string();

class CoreML : public Base {
  public:
    CoreML() {
        auto const path = rootDir + "/Assets/yolov8n.mlmodelc";
        NSString *pathStr = [[NSString alloc] initWithUTF8String:path.c_str()];
        NSURL *modelURL = [NSURL fileURLWithPath:pathStr];

        // Create model configuration
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        // Load the model
        NSError *error = nil;
        model = [[MLModel modelWithContentsOfURL:modelURL configuration:config error:&error] retain];

        if (error) {
            NSLog(@"Error: %@", error);
            throw std::runtime_error("Failed to load model");
        }

        // Create input dictionary
        inputDict = [[NSMutableDictionary alloc] init];
        NSDictionary *inputDesc = model.modelDescription.inputDescriptionsByName;
        for (NSString *key in inputDesc) {
            MLFeatureDescription *desc = inputDesc[key];
            if (!desc.multiArrayConstraint)
                throw std::runtime_error(key.UTF8String + std::string(" is not a multiarray from ") + path);
            NSArray<NSNumber *> *shape = desc.multiArrayConstraint.shape;
            inputDict[key] = [[MLMultiArray alloc] initWithShape:shape
                                                        dataType:desc.multiArrayConstraint.dataType
                                                           error:&error];
            if (error) {
                NSLog(@"Error creating input array for %@: %@", key, error);
                throw std::runtime_error("Failed to create input array");
            }
            NSLog(@"Input name: %@, shape: %@", key, shape);
        }
    }

  private:
    void infer() override {
        // Get the input data
        MLMultiArray *input = inputDict[@"image"];
        memcpy(input.dataPointer, cvImage.data, input.count * sizeof(float));

        NSError *error = nil;
        id<MLFeatureProvider> inputProvider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict error:&error];
        if (error) {
            NSLog(@"Error creating input provider: %@", error);
            return;
        }

        // Perform inference
        id<MLFeatureProvider> prediction = [model predictionFromFeatures:inputProvider error:&error];

        if (error) {
            NSLog(@"Error performing inference: %@", error);
            return;
        }

        // Copy output from the outputTensors [1, 84, 8400]
        std::vector<int> outputDim = {1, 84, 8400};
        MLMultiArray *outputArray = [prediction featureValueForName:@"output"].multiArrayValue;
        cv::Mat output(outputDim.size(), outputDim.data(), CV_32F, outputArray.dataPointer);
        outputs.clear();
        outputs.emplace_back(output);
    }

  private:
    MLModel *model;
    NSMutableDictionary *inputDict;
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
    std::string save_path = save_dir + "/coreml-objc.mp4";
    CoreML session;
    session.run(video_path, save_path);

    return 0;
}