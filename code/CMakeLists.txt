cmake_minimum_required(VERSION 3.10)

# 项目名称
project(UAV_Tracking_System)

# 设置 C++ 标准
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 寻找 OpenCV 库 (PnP 算法需要) 
find_package(OpenCV REQUIRED)

# 2. 寻找 Eigen3 库 (滤波器矩阵运算需要) 
find_package(Eigen3 REQUIRED)

# 3. 寻找 nlohmann_json (解析 jsonl 文件需要) 
# 如果你是通过 sudo apt install nlohmann-json3-dev 安装的，使用以下方式
find_package(nlohmann_json REQUIRED)

# 添加可执行文件 (假设你的文件名是 test.cpp)
add_executable(uav_tracker  main.cpp PnPSolver.cpp DataReader.cpp KalmanFilter.cpp utils.cpp )

# 链接库和头文件
target_link_libraries(uav_tracker
    PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
target_include_directories(uav_tracker PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)
