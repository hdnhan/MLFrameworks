#include <chrono>
#include <opencv2/opencv.hpp>

#include "onnxruntime_cxx_api.h"
#ifdef USE_TENSORRT
#include <tensorrt_provider_factory.h>
#include <tensorrt_provider_options.h>
#endif

class OrtSessionManager {
   public:
    OrtSessionManager(std::string const& model_path, std::string const& ep) : ep(ep) {
        // Create an ONNX Runtime session and load the model into it
        Ort::SessionOptions options;
        setup_provider(options);
        // ORT_LOGGING_LEVEL_WARNING, ORT_LOGGING_LEVEL_INFO, ORT_LOGGING_LEVEL_VERBOSE
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "LOG");
        try {
            session = Ort::Session(env, model_path.c_str(), options);
        } catch (Ort::Exception const& e) {
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

    void run() {
        try {
            // Run the inference
            session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(), inputTensors.size(),
                        outputNames.data(), outputTensors.data(), outputTensors.size());
        } catch (Ort::Exception const& e) {
            throw std::runtime_error(e.what());
        }
    }

   private:
    void setup_provider(Ort::SessionOptions& options) {
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.DisableProfiling();

        if (ep == "TensorrtExecutionProvider") {
#ifdef USE_TENSORRT
            const auto& api = Ort::GetApi();
            OrtTensorRTProviderOptionsV2* tensorrt_options;
            Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&tensorrt_options));
            tensorrt_options->device_id = 0;
            tensorrt_options->trt_max_workspace_size = 3221225472;  // 3 * 1024 * 1024 * 1024
            tensorrt_options->trt_engine_cache_enable = true;
            tensorrt_options->trt_engine_cache_path = "/workspace/Assets";
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
        } else if (ep == "CPUExecutionProvider") {
            // CPUExecutionProvider is the default provider
        } else {
            std::cout << "Provider not found: " << ep << std::endl;
        }
    }

   private:
    Ort::Env env{nullptr};  // https://github.com/microsoft/onnxruntime/issues/5320#issuecomment-700995952
    Ort::Session session{nullptr};

    std::string ep;  // Execution provider

    /*
    to keep the allocated memory => input and output names will be valid???
    another solution without keeping the allocated memory:
    char * buf = new char[1024];
    strcpy(buf, session.GetInputNameAllocated(i, allocator).get());
    */
    std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
    std::vector<Ort::AllocatedStringPtr> outputNamePtrs;

    std::vector<char const*> inputNames;
    std::vector<char const*> outputNames;

   public:
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    std::vector<std::vector<int64_t>> inputDims;
    std::vector<std::vector<int64_t>> outputDims;
};

class OrtInference {
   public:
    OrtInference(std::string const& model_path, std::string const& ep) : session_manager(model_path, ep) {}

    /**
     * Preprocess the input image
     * @param image: Input image
     * @param newShape: New shape of the image (height, width)
     */
    void preprocess(cv::Mat& image, cv::Size const& newShape) {
        // Swap R <-> B
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        // Resize but keep aspect ratio
        float h = image.rows, w = image.cols;
        float height = newShape.height, width = newShape.width;
        float ratio = std::min(width / w, height / h);
        cv::resize(image, image, cv::Size(static_cast<int>(w * ratio), static_cast<int>(h * ratio)));
        // Pad to newShape
        float dh = (height - image.rows) / 2.f;
        float dw = (width - image.cols) / 2.f;
        int top = std::round(dh - 0.1), bottom = std::round(dh + 0.1);
        int left = std::round(dw - 0.1), right = std::round(dw + 0.1);
        cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                           cv::Scalar(114, 114, 114));
        // Normalize
        image.convertTo(image, CV_32FC3, 1.f / 255);
        cv::Scalar mean(0, 0, 0);  // 0.485, 0.456, 0.406
        cv::Scalar std(1, 1, 1);   // 0.229, 0.224, 0.225
        cv::subtract(image, mean, image);
        cv::divide(image, std, image);

        // Fill the input tensor
        float* inputTensorValues = session_manager.inputTensors[0].GetTensorMutableData<float>();
        // [h, w, 3] => [h * w, 3] => [3, h * w]
        image = image.reshape(1, image.size().area()).t();
        std::memcpy(inputTensorValues, image.data, image.total() * sizeof(float));
    }

    void infer() { session_manager.run(); }

    /**
     * Postprocess the output of the network.
     * @param newShape: New shape of the image (height, width).
     * @param oriShape: Original shape of the image (height, width).
     * @param confThres: Confidence threshold.
     * @param iouThres: IoU threshold.
     * @return: A tuple of 3 vectors: bboxes, scores, classIds.
     */
    std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> postprocess(
        cv::Size const& newShape, cv::Size const& oriShape, float confThres = 0.25, float iouThres = 0.45) {
        // Copy output from the outputTensors [1, 84, 8400]
        float* outputTensorValues = session_manager.outputTensors[0].GetTensorMutableData<float>();
        std::vector<int> outputDim;
        for (auto e : session_manager.outputDims[0]) outputDim.push_back(static_cast<int>(e));
        cv::Mat output(outputDim.size(), outputDim.data(), CV_32F, outputTensorValues);

        output = output.reshape(1, output.size[1]).t();  // (1, 84, 8400) -> (8400, 84)
        std::vector<cv::Rect> bboxes;                    // (8400, 4)
        std::vector<float> scores;                       // (8400,)
        std::vector<int> class_ids;                      // (8400,)
        bboxes.reserve(output.rows);
        scores.reserve(output.rows);
        class_ids.reserve(output.rows);
        double val;
        int idx[2];  // dims: 1 x 80 => idx[0] is always 0
        for (int i = 0; i < output.rows; ++i) {
            // cxcywh to xywh
            float cx = output.at<float>(i, 0), cy = output.at<float>(i, 1);
            float w = output.at<float>(i, 2), h = output.at<float>(i, 3);
            cv::Rect bbox{int(cx - w / 2), int(cy - h / 2), int(w), int(h)};
            cv::minMaxIdx(output.row(i).colRange(4, output.cols), nullptr, &val, nullptr, idx);
            bboxes.emplace_back(bbox);
            scores.emplace_back(val);
            class_ids.emplace_back(idx[1]);
        }

        // Batched NMS
        std::vector<int> keep;
        cv::dnn::NMSBoxesBatched(bboxes, scores, class_ids, confThres, iouThres, keep, 1.f, 300);

        // Scale to original shape
        float gain = std::min(newShape.width / static_cast<float>(oriShape.width),
                              newShape.height / static_cast<float>(oriShape.height));  // gain = old / new
        int padw = std::round((newShape.width - oriShape.width * gain) / 2.f - 0.1);
        int padh = std::round((newShape.height - oriShape.height * gain) / 2.f - 0.1);

        std::vector<cv::Rect> bboxes_res;
        std::vector<float> scores_res;
        std::vector<int> class_ids_res;
        for (int i : keep) {
            scores_res.push_back(scores[i]);
            class_ids_res.push_back(class_ids[i]);

            int x1 = std::max(int((bboxes[i].x - padw) / gain), 0);
            int y1 = std::max(int((bboxes[i].y - padh) / gain), 0);
            int x2 = std::min(int((bboxes[i].x + bboxes[i].width - padw) / gain), oriShape.width);
            int y2 = std::min(int((bboxes[i].y + bboxes[i].height - padh) / gain), oriShape.height);
            bboxes_res.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        }
        return {bboxes_res, scores_res, class_ids_res};
    }

   private:
    OrtSessionManager session_manager;
};

static std::unordered_map<int, cv::Scalar> colors;
/**
 * Draw the bounding boxes on the image.
 * @param image: Input image.
 * @param bboxes: Bounding boxes.
 * @param scores: Scores.
 * @param classIds: Class IDs.
 */
void draw(cv::Mat& image, std::vector<cv::Rect> const& bboxes, std::vector<float> const& scores,
          std::vector<int> const& classIds) {
    for (size_t i = 0; i < bboxes.size(); i++) {
        if (!colors.count(classIds[i]))
            colors[classIds[i]] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        auto color = colors[classIds[i]];
        cv::rectangle(image, bboxes[i], color, 2);
        std::string label = cv::format("%.2f", scores[i]);
        cv::putText(image, label, cv::Point(bboxes[i].x, bboxes[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2);
    }
}

void warm_up(OrtInference& OrtSess) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Size newShape(640, 640);  // (width, height)
    OrtSess.preprocess(image, newShape);
    OrtSess.infer();
    auto [bboxes, scores, classIds] = OrtSess.postprocess(newShape, image.size());
}

void run(std::string const& ep, bool verbose = false) {
    OrtInference OrtSess("/workspace/Assets/yolov8n.onnx", ep);
    std::string out_path = "/workspace/Results/onnxruntime-cpp-" + ep + ".mp4";

    // Warm up
    for (int i = 0; i < 10; ++i) warm_up(OrtSess);

    // Load video
    cv::VideoCapture cap("/workspace/Assets/video.mp4");
    cv::VideoWriter out;

    cv::Size newShape(640, 640);  // (width, height)

    std::chrono::high_resolution_clock::time_point start, end;
    double t_pre = 0, t_infer = 0, t_post = 0;
    double t_pres = 0, t_infers = 0, t_posts = 0;
    int count = 0;
    while (cap.isOpened()) {
        cv::Mat frame;
        cap.read(frame);
        if (frame.empty()) {
            break;
        }

        cv::Size oriShape = frame.size();
        cv::Mat image = frame.clone();

        // Preprocess
        start = std::chrono::high_resolution_clock::now();
        OrtSess.preprocess(image, newShape);
        end = std::chrono::high_resolution_clock::now();
        t_pre = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_pres += t_pre;

        // Inference
        start = std::chrono::high_resolution_clock::now();
        OrtSess.infer();
        end = std::chrono::high_resolution_clock::now();
        t_infer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_infers += t_infer;

        // Postprocess
        start = std::chrono::high_resolution_clock::now();
        auto [bboxes, scores, classIds] = OrtSess.postprocess(newShape, oriShape);
        end = std::chrono::high_resolution_clock::now();
        t_post = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_posts += t_post;

        double fps = 1e6 / (t_pre + t_infer + t_post);
        draw(frame, bboxes, scores, classIds);
        std::string fps_str = cv::format("FPS: %.2f", fps);
        cv::putText(frame, fps_str, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        count++;
        if (verbose)
            std::cout << std::fixed << std::setprecision(3) << count << " -> Preprocess: " << t_pre / 1e3
                      << " ms, Inference: " << t_infer / 1e3 << " ms, Postprocess: " << t_post / 1e3
                      << " ms, FPS: " << fps << std::endl;

        if (!out.isOpened())
            out.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, frame.size());
        out.write(frame);
    }

    if (cap.isOpened()) cap.release();
    if (out.isOpened()) out.release();

    std::cout << "\n" << ep << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Preprocess: " << t_pres / 1e3 / count
              << " ms, Inference: " << t_infers / 1e3 / count << " ms, Postprocess: " << t_posts / 1e3 / count
              << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "FPS: " << 1e6 * count / (t_pres + t_infers + t_posts)
              << std::endl;
}

int main() {
    auto providers = Ort::GetAvailableProviders();
    for (auto const& p : providers) run(p);
    return 0;
}