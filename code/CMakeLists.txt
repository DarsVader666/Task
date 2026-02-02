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
find_package(nlohmann_json REQUIRED)

# 添加可执行文件 (假设你的文件名是 test.cpp)
add_executable(uav_tracker  main.cpp PnPSolver.cpp DataReader.cpp KalmanFilter.cpp utils.cpp )

# 获取项目根目录（即CMakeLists.txt所在目录）的绝对路径
get_filename_component(PROJECT_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
# 将该路径作为一个预处理器宏传递给源代码
target_compile_definitions(uav_tracker PRIVATE PROJECT_DATA_DIR="${PROJECT_DATA_DIR}")

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
