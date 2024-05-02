#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

class Base {
  public:
    Base() = default;
    virtual ~Base() = default;

  private:
    // Preprocess an image to cvImage of shape newShape.
    void preprocess(cv::Mat const &image, cv::Size const &newShape);

    // Run inference on cvImage and store all results in outputs.
    virtual void infer() = 0;

    // Postprocess outputs[0] of shape (1, nc + 4, 8400) to get  bboxes, scores, and classIDs.
    void postprocess(cv::Size const &newShape, cv::Size const &oriShape, float confThres = 0.25,
                     float iouThres = 0.45);

    // Visualize results on original image using bboxes, scores, and classIDs.
    void draw(cv::Mat &image);

    // Warm up the network by running inference on a dummy image.
    void warmUp();

  public:
    void run(std::string const &video_path, std::string const &save_path, bool verbose = false);

  protected:
    // preprocessed image
    cv::Mat cvImage;
    // output of the network
    std::vector<cv::Mat> outputs;

  private:
    // postprocessed results
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> classIDs;
    // colors for different classes
    std::unordered_map<int, cv::Scalar> colors;
};