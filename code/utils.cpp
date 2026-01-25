#include "utils.h"


// 时间同步函数实现
void synchronizeData(
    double timestamp,
    const std::vector<StatusData>& statusData,
    const std::vector<PoseData>& cameraData,
    const std::vector<PoseData>& uavData,
    StatusData& outStatus,
    PoseData& outCamera,
    PoseData& outUav
) {
    // 查找最接近的状态数据
    auto statusIt = std::min_element(statusData.begin(), statusData.end(),
        [timestamp](const StatusData& a, const StatusData& b) {
            return std::abs(a.record_time - timestamp) < std::abs(b.record_time - timestamp);
        });
    if (statusIt != statusData.end()) {
        outStatus = *statusIt;
    } else {
        throw std::runtime_error("No status data found for timestamp: " + std::to_string(timestamp));
    }

    // 查找最接近的相机位姿数据
    auto cameraIt = std::min_element(cameraData.begin(), cameraData.end(),
        [timestamp](const PoseData& a, const PoseData& b) {
            return std::abs(a.record_time - timestamp) < std::abs(b.record_time - timestamp);
        });
    if (cameraIt != cameraData.end()) {
        outCamera = *cameraIt;
    } else {
        throw std::runtime_error("No camera pose data found for timestamp: " + std::to_string(timestamp));
    }

    // 查找最接近的无人机位姿数据
    auto uavIt = std::min_element(uavData.begin(), uavData.end(),
        [timestamp](const PoseData& a, const PoseData& b) {
            return std::abs(a.record_time - timestamp) < std::abs(b.record_time - timestamp);
        });
    if (uavIt != uavData.end()) {
        outUav = *uavIt;
    } else {
        throw std::runtime_error("No UAV pose data found for timestamp: " + std::to_string(timestamp));
    }
}

// 获取图像点函数实现
std::vector<cv::Point2f> getImagePoints(const StatusData& status) {
    std::vector<cv::Point2f> points;
    
    // 添加本体中心点（如果存在）
    if (status.drone_state.has_body) {
        points.push_back(cv::Point2f(
            static_cast<double>(status.drone_state.body_x), 
            static_cast<double>(status.drone_state.body_y)
        ));
    }
    
    // 添加螺旋桨点（按ID排序）
    std::vector<Propeller> sortedProps = status.drone_state.propellers;
    std::sort(sortedProps.begin(), sortedProps.end(), 
        [](const Propeller& a, const Propeller& b) {
            return a.id < b.id;
        });
    
    for (const auto& prop : sortedProps) {
        points.push_back(cv::Point2f(
            static_cast<double>(prop.x), 
            static_cast<double>(prop.y)
        ));
    }
    
    return points;
}

