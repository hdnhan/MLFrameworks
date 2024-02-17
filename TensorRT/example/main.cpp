#include <cuda_runtime_api.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

// Class to extend TensorRT ILogger
class Logger : public nvinfer1::ILogger {
   public:
    Severity mSeverity = Severity::kINFO;
    Logger(Severity severity = Severity::kINFO) : mSeverity(severity){};

    void error(std::string const& msg) { return log(Severity::kERROR, msg.c_str()); }
    void warning(std::string const& msg) { return log(Severity::kWARNING, msg.c_str()); }
    void info(std::string const& msg) { return log(Severity::kINFO, msg.c_str()); }
    void verbose(std::string const& msg) { return log(Severity::kVERBOSE, msg.c_str()); }

   private:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > mSeverity) return;
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

class TRTSessionManager {
   public:
    TRTSessionManager(std::string const& modelName, std::string const& dtype = "f32") : dtype(dtype) {
        mLogger = std::make_unique<Logger>();

        f_onnx = modelName + ".onnx";
        f_engine = modelName + "-" + dtype + ".engine";

        initLibNvInferPlugins(nullptr, "");

        if (!std::filesystem::exists(f_engine)) {
            if (!std::filesystem::exists(f_onnx))
                throw std::runtime_error("Error, onnx file not found at " + f_onnx);
            build();
        }
        if (!mEngine) load();

        mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        bindings.resize(mEngine->getNbBindings());
        for (int i = 0; i < mEngine->getNbBindings(); ++i) {
            nvinfer1::Dims dims = mEngine->getBindingDimensions(i);
            auto size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>());
            cudaMalloc(&bindings[i], size * sizeof(float));
            if (mEngine->bindingIsInput(i)) {
                mInputIndices.emplace_back(i);
                mInputDims.emplace_back(dims);
            } else {
                mOutputIndices.emplace_back(i);
                mOutputDims.emplace_back(dims);
                outputs.emplace_back(std::vector<float>(size));
            }
        }
    }

    ~TRTSessionManager() {
        for (void* binding : bindings) cudaFree(binding);
    }

    void run() {
        if (!mContext->executeV2(bindings.data())) throw std::runtime_error("Error running inference.");
        for (size_t i = 0; i < mOutputIndices.size(); ++i) {
            cudaMemcpy(outputs[i].data(), bindings[mOutputIndices[i]], outputs[i].size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
        }
    }

   private:
    void build() {
        std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(*mLogger)};
        std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

        config->setMaxWorkspaceSize(1 << 30);  // 1GiB
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
        if (builder->platformHasFastFp16() && dtype == "f16") config->setFlag(nvinfer1::BuilderFlag::kFP16);
        if (builder->platformHasFastInt8() && dtype == "i8") {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }

        auto const explicitBatch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
        std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, *mLogger)};

        mLogger->info("Parsing onnx file from " + f_onnx);
        if (!parser->parseFromFile(f_onnx.c_str(), static_cast<int>(mLogger->mSeverity)))
            throw std::runtime_error("Error parsing onnx file.");

        mLogger->info("Building plan from network and config");
        std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*mLogger)};
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()));

        mLogger->info("Saving serialized engine to " + f_engine);
        std::ofstream file;
        file.open(f_engine, std::ios::binary | std::ios::out);
        file.write((const char*)plan->data(), plan->size());
        file.close();
    }

    void load() {
        std::ifstream file(f_engine, std::ios::binary | std::ios::ate);
        mLogger->info("Loading engine at " + f_engine);

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) throw std::runtime_error("Error reading engine file.");

        std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(*mLogger)};
        mEngine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    }

   private:
    std::string f_onnx, f_engine;
    std::string dtype;

    std::unique_ptr<Logger> mLogger;
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;

   public:
    std::vector<void*> bindings;
    std::vector<int> mInputIndices, mOutputIndices;
    std::vector<nvinfer1::Dims> mInputDims, mOutputDims;
    std::vector<std::vector<float>> outputs;
};

class TRTInference {
   public:
    TRTInference(std::string const& model_path) : session_manager(model_path) {}

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

        // [h, w, 3] => [h * w, 3] => [3, h * w]
        image = image.reshape(1, image.size().area()).t();
        cudaMemcpy(session_manager.bindings[0], image.data, image.total() * sizeof(float),
                   cudaMemcpyHostToDevice);
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
        std::vector<int> outputDim;
        for (int i = 0; i < session_manager.mOutputDims[0].nbDims; ++i)
            outputDim.emplace_back(session_manager.mOutputDims[0].d[i]);
        cv::Mat output(outputDim.size(), outputDim.data(), CV_32F, session_manager.outputs[0].data());

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
    TRTSessionManager session_manager;
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

void warm_up(TRTInference& TRTSess) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Size newShape(640, 640);  // (width, height)
    TRTSess.preprocess(image, newShape);
    TRTSess.infer();
    auto [bboxes, scores, classIds] = TRTSess.postprocess(newShape, image.size());
}

void run(bool verbose = false) {
    TRTInference TRTSess("/workspace/Assets/yolov8n");
    std::string out_path = "/workspace/Results/tensorrt-cpp.mp4";

    // Warm up
    for (int i = 0; i < 10; ++i) warm_up(TRTSess);

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
        TRTSess.preprocess(image, newShape);
        end = std::chrono::high_resolution_clock::now();
        t_pre = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_pres += t_pre;

        // Inference
        start = std::chrono::high_resolution_clock::now();
        TRTSess.infer();
        end = std::chrono::high_resolution_clock::now();
        t_infer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_infers += t_infer;

        // Postprocess
        start = std::chrono::high_resolution_clock::now();
        auto [bboxes, scores, classIds] = TRTSess.postprocess(newShape, oriShape);
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

    std::cout << std::fixed << std::setprecision(3) << "Preprocess: " << t_pres / 1e3 / count
              << " ms, Inference: " << t_infers / 1e3 / count << " ms, Postprocess: " << t_posts / 1e3 / count
              << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "FPS: " << 1e6 * count / (t_pres + t_infers + t_posts)
              << std::endl;
}

int main() {
    run();
    return 0;
}