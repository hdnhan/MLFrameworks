#include "base.hpp"
#include <CoreML/CoreML.h>
#include <filesystem>

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
        model = [MLModel modelWithContentsOfURL:modelURL configuration:config error:&error];

        if (error) {
            NSLog(@"Error: %@", error);
            throw std::runtime_error("Failed to load model");
        }

        // Create input dictionary
        inputDict = [NSMutableDictionary dictionary];
        // image: MLMultiArray 1x3x640x640
        inputDict[@"image"] = [[MLMultiArray alloc] initWithShape:@[ @1, @3, @640, @640 ]
                                                         dataType:MLMultiArrayDataTypeFloat32
                                                            error:&error];
    }

  private:
    void infer() override {
        // Get the input data
        MLMultiArray *input = inputDict[@"image"];
        memcpy(input.dataPointer, cvImage.data, 640 * 640 * 3 * sizeof(float));

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

int main() {
    auto const video_path = rootDir + "/Assets/video.mp4";
    auto const save_path = rootDir + "/Results/coreml-objc.mp4";
    CoreML session;
    session.run(video_path, save_path, false);

    return 0;
}