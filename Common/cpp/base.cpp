#include "base.hpp"

void Base::preprocess(cv::Mat const &image, cv::Size const &newShape) {
    // Resize but keep aspect ratio
    float h = image.rows, w = image.cols;
    float height = newShape.height, width = newShape.width;
    float ratio = std::min(width / w, height / h);
    cv::resize(image, cvImage, cv::Size(static_cast<int>(w * ratio), static_cast<int>(h * ratio)));
    // Pad to newShape
    float dh = (height - cvImage.rows) / 2.f;
    float dw = (width - cvImage.cols) / 2.f;
    int top = std::round(dh - 0.1), bottom = std::round(dh + 0.1);
    int left = std::round(dw - 0.1), right = std::round(dw + 0.1);
    cv::copyMakeBorder(cvImage, cvImage, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    cv::dnn::blobFromImage(cvImage, cvImage, 1.f / 255, cv::Size(), cv::Scalar(),
                           true); // 4D blob from image
}

void Base::postprocess(cv::Size const &newShape, cv::Size const &oriShape, float confThres, float iouThres) {
    cv::Mat output = outputs[0];
    output = output.reshape(1, output.size[1]).t(); // (1, 84, 8400) -> (8400, 84)

    std::vector<cv::Rect> _bboxes; // (8400, 4)
    std::vector<float> _scores;    // (8400,)
    std::vector<int> _classIDs;    // (8400,)

    _bboxes.reserve(output.rows);
    _scores.reserve(output.rows);
    _classIDs.reserve(output.rows);

    double val;
    int idx[2]; // dims: 1 x 80 => idx[0] is always 0
    for (int i = 0; i < output.rows; ++i) {
        // cxcywh to xywh
        float cx = output.at<float>(i, 0), cy = output.at<float>(i, 1);
        float w = output.at<float>(i, 2), h = output.at<float>(i, 3);
        cv::Rect bbox{int(cx - w / 2), int(cy - h / 2), int(w), int(h)};
        cv::minMaxIdx(output.row(i).colRange(4, output.cols), nullptr, &val, nullptr, idx);
        _bboxes.emplace_back(bbox);
        _scores.emplace_back(val);
        _classIDs.emplace_back(idx[1]);
    }

    // Batched NMS
    std::vector<int> keep;
    cv::dnn::NMSBoxesBatched(_bboxes, _scores, _classIDs, confThres, iouThres, keep, 1.f, 300);

    // Scale to original shape
    float gain = std::min(newShape.width / static_cast<float>(oriShape.width),
                          newShape.height / static_cast<float>(oriShape.height)); // gain = old / new
    int padw = std::round((newShape.width - oriShape.width * gain) / 2.f - 0.1);
    int padh = std::round((newShape.height - oriShape.height * gain) / 2.f - 0.1);

    // Reset previous results
    bboxes.clear();
    scores.clear();
    classIDs.clear();
    for (int i : keep) {
        scores.push_back(_scores[i]);
        classIDs.push_back(_classIDs[i]);

        int x1 = std::max(int((_bboxes[i].x - padw) / gain), 0);
        int y1 = std::max(int((_bboxes[i].y - padh) / gain), 0);
        int x2 = std::min(int((_bboxes[i].x + _bboxes[i].width - padw) / gain), oriShape.width);
        int y2 = std::min(int((_bboxes[i].y + _bboxes[i].height - padh) / gain), oriShape.height);
        bboxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }
}

void Base::draw(cv::Mat &image) {
    for (size_t i = 0; i < bboxes.size(); i++) {
        if (!colors.count(classIDs[i]))
            colors[classIDs[i]] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
        auto color = colors[classIDs[i]];
        cv::rectangle(image, bboxes[i], color, 2);
        std::string label = cv::format("%.2f", scores[i]);
        cv::putText(image, label, cv::Point(bboxes[i].x, bboxes[i].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2);
    }
}

void Base::warmUp() {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::Size newShape(640, 640);
    preprocess(image, newShape);
    infer();
    postprocess(newShape, image.size());
}

void Base::run(std::string const &video_path, std::string const &save_path, bool verbose) {
    // Warm up
    for (int i = 0; i < 10; ++i)
        warmUp();

    // Load video
    cv::VideoWriter writer;
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open the video file." << std::endl;
        return;
    }

    cv::Size newShape(640, 640); // (width, height)
    std::chrono::high_resolution_clock::time_point start, end;
    double t_pre = 0, t_infer = 0, t_post = 0;
    double t_pres = 0, t_infers = 0, t_posts = 0;
    int count = 0;
    cv::Mat frame;
    while (cap.read(frame) && !frame.empty()) {
        // Preprocess
        start = std::chrono::high_resolution_clock::now();
        preprocess(frame, newShape);
        end = std::chrono::high_resolution_clock::now();
        t_pre = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_pres += t_pre;

        // Inference
        start = std::chrono::high_resolution_clock::now();
        infer();
        end = std::chrono::high_resolution_clock::now();
        t_infer = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_infers += t_infer;

        // Postprocess
        start = std::chrono::high_resolution_clock::now();
        postprocess(newShape, frame.size());
        end = std::chrono::high_resolution_clock::now();
        t_post = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        t_posts += t_post;

        // Draw
        draw(frame);
        double fps = 1e6 / (t_pre + t_infer + t_post);
        std::string fps_str = cv::format("FPS: %.2f", fps);
        cv::putText(frame, fps_str, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        count++;
        if (verbose)
            std::cout << std::fixed << std::setprecision(3) << count << " -> Preprocess: " << t_pre / 1e3
                      << " ms, Inference: " << t_infer / 1e3 << " ms, Postprocess: " << t_post / 1e3
                      << " ms, FPS: " << fps << std::endl;

        // Save the video
        if (!writer.isOpened())
            writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 20, frame.size());
        writer.write(frame);
    }
    if (cap.isOpened())
        cap.release();
    if (writer.isOpened())
        writer.release();

    std::cout << std::fixed << std::setprecision(3) << "Preprocess: " << t_pres / 1e3 / count
              << " ms, Inference: " << t_infers / 1e3 / count << " ms, Postprocess: " << t_posts / 1e3 / count
              << " ms" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "FPS: " << 1e6 * count / (t_pres + t_infers + t_posts)
              << std::endl;
}