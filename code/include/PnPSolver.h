#pragma once
#include <opencv2/opencv.hpp>
#include "DataReader.h"

class PnPSolver
{
public:
    struct EPnPResult
    {
        cv::Vec3d translation; // 位移向量
        cv::Mat rotation;      // 旋转矩阵
        double reprojectionError; // <--- 新增：重投影误差 (像素单位)
    };

    static EPnPResult solveEPnP(
        const std::vector<cv::Point3f> &objectPoints,
        const std::vector<cv::Point2f> &imagePoints,
        const cv::Mat &cameraMatrix,
        const cv::Mat &disCoeffs);

    static EPnPResult solveWithScore(
        const std::vector<cv::Point3f> &objectPoints,
        const std::vector<cv::Point2f> &imagePoints,
        const cv::Mat &cameraMatrix,
        const cv::Mat &disCoeffs);

    static cv::Mat quatToRotMat(const cv::Vec4d &q); // 四元数转化为旋转矩阵
    static cv::Vec3d transformToWorld(               // 把目标在相机坐标系的位姿转换到世界坐标系
        const cv::Mat &R_target_cam,                 // 旋转矩阵
        const cv::Vec3d &tvec,                         // 箱子在世界坐标系的坐标
        const PoseData &cameraPose                   // 相机的位姿
    );
};
