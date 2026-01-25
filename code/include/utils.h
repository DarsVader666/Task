#pragma once
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "DataReader.h"

void synchronizeData(
    double timestamp,
    const std::vector<StatusData>& statusData,
    const std::vector<PoseData>& cameraData,
    const std::vector<PoseData>& uavData,
    StatusData& outStatus,
    PoseData& outCamera,
    PoseData& outUav
);

std::vector<cv::Point2f> getImagePoints(const StatusData& status);