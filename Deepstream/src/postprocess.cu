#include "nvdsinfer_custom_impl.h"
#include <iostream>
#include <string>
#include <vector>

extern "C" {

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, min, max) (MAX(MIN(a, max), min))

bool NvDsInferParseCustomPostprocess(std::vector<NvDsInferLayerInfo> const &layer_info,
                                     NvDsInferNetworkInfo const &network_info,
                                     NvDsInferParseDetectionParams const &detection_params,
                                     std::vector<NvDsInferObjectDetectionInfo> &object_detection_info) {

    auto layerFinder = [&layer_info](std::string const &name,
                                     NvDsInferDataType data_type) -> NvDsInferLayerInfo const * {
        for (auto &layer : layer_info) {
            if (layer.dataType == data_type && layer.layerName == name)
                return &layer;
        }
        return nullptr;
    };

    NvDsInferLayerInfo const *bboxes = layerFinder("bboxes", FLOAT);
    NvDsInferLayerInfo const *scores = layerFinder("scores", FLOAT);
    NvDsInferLayerInfo const *ids = layerFinder("ids", INT32); // UINT32 not supported

    // Validate presence of all required layers
    if (!bboxes || !scores || !ids) {
        std::cerr << "ERROR: some layers missing or unsupported data types "
                  << "in output tensors" << std::endl;
        return false;
    }
    // Validate shapes of the layers
    if (bboxes->inferDims.numDims != 2 || bboxes->inferDims.d[1] != 4) {
        std::cerr << "ERROR: unexpected bboxes layer shape" << std::endl;
        return false;
    }
    if (scores->inferDims.numDims != 1) {
        std::cerr << "ERROR: unexpected scores layer shape" << std::endl;
        return false;
    }
    if (ids->inferDims.numDims != 1) {
        std::cerr << "ERROR: unexpected ids layer shape" << std::endl;
        return false;
    }
    if (bboxes->inferDims.d[0] != scores->inferDims.d[0] || bboxes->inferDims.d[0] != ids->inferDims.d[0]) {
        std::cerr << "ERROR: mismatched number of detections between layers" << std::endl;
        return false;
    }
    int num_detections = bboxes->inferDims.d[0];

    // Parse detections

    for (int i = 0; i < num_detections; ++i) {
        NvDsInferObjectDetectionInfo res;
        float x1 = static_cast<float *>(bboxes->buffer)[i * 4];
        float y1 = static_cast<float *>(bboxes->buffer)[i * 4 + 1];
        float x2 = static_cast<float *>(bboxes->buffer)[i * 4 + 2];
        float y2 = static_cast<float *>(bboxes->buffer)[i * 4 + 3];
        res.left = CLIP(x1, 0, network_info.width - 1);
        res.top = CLIP(y1, 0, network_info.height - 1);
        res.width = CLIP(x2 - x1, 0, network_info.width - res.left - 1);
        res.height = CLIP(y2 - y1, 0, network_info.height - res.top - 1);
        if (res.width <= 0.0f || res.height <= 0.0f)
            continue; // skip invalid boxes

        res.detectionConfidence = static_cast<float *>(scores->buffer)[i];
        res.classId = static_cast<unsigned int>(static_cast<int32_t *>(ids->buffer)[i]);
        object_detection_info.push_back(res);
    }

    return true;
}

} // extern "C"
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomPostprocess);