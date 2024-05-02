#import "bindings.h"
#import "yolov8n.h"

#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#include <stdlib.h>

#if __cplusplus
extern "C" {
#endif

const void *load(const char *path) {
  NSString *pathStr = [[NSString alloc] initWithUTF8String:path];
  NSURL *modelURL = [NSURL fileURLWithPath:pathStr];

  const void *model =
      CFBridgingRetain([[yolov8n alloc] initWithContentsOfURL:modelURL
                                                        error:nil]);
  return model;
}

void predict(const void *model, float *input, float *ouput) {
  yolov8n *yolov8nModel = (__bridge yolov8n *)(model);

  // Convert input to CVPixelBufferRef
  CVPixelBufferRef inputBuffer;
  CVPixelBufferCreateWithBytes(
      kCFAllocatorDefault, 640, 640, kCVPixelFormatType_32BGRA, input,
      640 * sizeof(float), nil, nil, nil, &inputBuffer);

  // Perform prediction
  yolov8nOutput *pred = [yolov8nModel predictionFromImage:inputBuffer
                                                    error:nil];
  // Copy the prediction to the output array
  memcpy(ouput, pred.var_914.dataPointer, 1 * 84 * 8400 * sizeof(float));
}

#if __cplusplus
} // Extern C
#endif