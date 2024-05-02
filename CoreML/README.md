## CoreML on M2 MacBook Air
### Python
```bash
python python/main.py
```

### C++ (Objective-C++)
```bash
# Install XCode from App Store
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# Compile the model
xcrun coremlc compile ../Assets/yolov8n.mlpackage cpp
xcrun coremlc generate ../Assets/yolov8n.mlpackage cpp

# Compile and run the project
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/main
```