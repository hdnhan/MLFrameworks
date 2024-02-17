#include <chrono>
#include <opencv2/opencv.hpp>

/**
 * Preprocess the input image
 * @param image: Input image
 * @param newShape: New shape of the image (height, width)
 */
void preprocess(cv::Mat& image, cv::Size const& newShape) {
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
    cv::dnn::blobFromImage(image, image, 1.f / 255, cv::Size(), cv::Scalar(), true);  // 4D blob from image
}

/**
 * Runs the input image through the network.
 * @param image: Input image (1, 3, h, w).
 * @param model: The network.
 * @return: The output of the network (1, nc + 4, 8400).
 */
cv::Mat infer(cv::Mat const& image, cv::dnn::Net& model) {
    // Sets the input to the network.
    model.setInput(image);
    // Runs the forward pass to get output of the output layers.
    std::vector<cv::Mat> outputs;
    auto layers = model.getUnconnectedOutLayersNames();
    model.forward(outputs, layers);
    return outputs[0];
}

/**
 * Postprocess the output of the network.
 * @param output: The output of the network (1, nc + 4, 8400).
 * @param newShape: New shape of the image (height, width).
 * @param oriShape: Original shape of the image (height, width).
 * @param confThres: Confidence threshold.
 * @param iouThres: IoU threshold.
 * @return: A tuple of 3 vectors: bboxes, scores, classIds.
 */
std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>> postprocess(cv::Mat& output,
                                                                                    cv::Size const& newShape,
                                                                                    cv::Size const& oriShape,
                                                                                    float confThres = 0.25,
                                                                                    float iouThres = 0.45) {
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

void warm_up(cv::dnn::Net& model) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Size newShape(640, 640);
    preprocess(image, newShape);
    cv::Mat output = infer(image, model);
    auto [bboxes, scores, classIds] = postprocess(output, newShape, image.size());
}

void run(bool use_cuda, bool verbose = false) {
    // Load the network
    cv::dnn::Net model = cv::dnn::readNet("/workspace/Assets/yolov8n.onnx");

    // Warm up
    for (int i = 0; i < 10; ++i) warm_up(model);

    std::string out_path = "/workspace/Results/opencv-cpp-cpu.mp4";
    if (use_cuda) {
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        out_path = "/workspace/Results/opencv-cpp-cuda.mp4";
    }

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
        preprocess(image, newShape);
        end = std::chrono::high_resolution_clock::now();
        t_pre = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_pres += t_pre;

        // Inference
        start = std::chrono::high_resolution_clock::now();
        cv::Mat detections = infer(image, model);
        end = std::chrono::high_resolution_clock::now();
        t_infer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_infers += t_infer;

        // Postprocess
        start = std::chrono::high_resolution_clock::now();
        auto [bboxes, scores, classIds] = postprocess(detections, newShape, oriShape);
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

    std::cout << "\nCUDA: " << use_cuda << std::endl;
    std::cout << std::fixed << std::setprecision(3) << "Preprocess: " << t_pres / 1e3 / count
              << " ms, Inference: " << t_infers / 1e3 / count << " ms, Postprocess: " << t_posts / 1e3 / count
              << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "FPS: " << 1e6 * count / (t_pres + t_infers + t_posts)
              << std::endl;
}

int main() {
    bool use_cuda = cv::cuda::getCudaEnabledDeviceCount() > 0;

    if (use_cuda) run(true);
    run(false);

    return 0;
}
