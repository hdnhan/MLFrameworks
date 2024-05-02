#if __cplusplus
extern "C" {
#endif

const void* load(const char* path);
void predict(const void* model, float* input, float* output);

#if __cplusplus
}  // Extern C
#endif