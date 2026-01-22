#include <iostream>
#include <opencv2/opencv.hpp>
#include "DataReader.h"
#include "PnPSolver.h"
#include "KalmanFilter.h"
#include "utils.h"

int main()
{

    std::cout << "和一位";
    // 1. 读取数据文件
    auto statusData = DataReader::readStatusFile("/home/d1c/Git/Task/data_status_slow.jsonl");
    std::cout << "Found " << statusData.size() << " status records" << std::endl;
    auto cameraData = DataReader::readPoseFile("/home/d1c/Git/Task/data_tf_camera_slow.jsonl");
    std::cout << "Found " << cameraData.size() << " camera records" << std::endl;
    // auto uavData = DataReader::readPoseFile("/home/d1c/Git/Task/data_tf_uav_slow.jsonl");
    // std::cout << "Found " << cameraData.size() << " uav records" << std::endl;

    // 2. 相机参数设置（需要根据实际相机标定）
    double horizontalFOV = 1.732; // 弧度
    double imageWidth = 1920.0;
    double imageHeight = 1080.0;

    // 计算焦距
    double fx = imageWidth / (2.0 * tan(horizontalFOV / 2.0));
    double fy = fx;

    double cx = imageWidth / 2.0;  // 960
    double cy = imageHeight / 2.0; // 540

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // 3. 无人机3D模型点（本体坐标系，单位：米）
    std::vector<cv::Point3f> modelPoints = {
        cv::Point3f(0, 0, 0),       // Body中心
        cv::Point3f(-0.1, 0.1, 0),  // ID1
        cv::Point3f(-0.1, -0.1, 0), // ID2
        cv::Point3f(0.1, -0.1, 0),  // ID3
        cv::Point3f(0.1, 0.1, 0)    // ID4
        // 4个
    };

    int m = 2;
    int stateDim = 3 * (m + 1) + 1;
    int measureDim = 4;
    double dt = 0.033;
    KalmanFilter kf(stateDim, measureDim, m, dt);

    // 初始状态和协方差
    cv::Mat Q = cv::Mat::eye(stateDim, stateDim, CV_64F) * 0.01;    // 过程噪声
    Q.at<double>(stateDim - 1, stateDim - 1) = 1e-16;                // 静态尺度噪声小
    cv::Mat R = cv::Mat::eye(measureDim, measureDim, CV_64F) * 0.1; // 测量噪声
    R.at<double>(3, 3) = 50;
    kf.setProcessNoiseCov(Q);
    kf.setMeasurementNoiseCov(R);

    // --- 新增：初始化状态向量和协方差矩阵 ---
    cv::Mat initialState = cv::Mat::zeros(stateDim, 1, CV_64F);
    cv::Mat initialCov = cv::Mat::eye(stateDim, stateDim, CV_64F) * 1.0;
    // 注意：此时 state_ 依然是空的，必须调用 init
    bool isInitialized = false;
    int flag = 1;
    cv::Mat estimatedState;


    for (const auto &status : statusData)
    {
        StatusData syncedStatus;
        PoseData syncedCamera, syncedUav;

        synchronizeData(
            status.record_time,
            statusData,
            cameraData,
            //  uavData,
            syncedStatus,
            syncedCamera
            // syncedUav
        );

        // 获取2D图像点
        std::vector<cv::Point2f> imagePoints = getImagePoints(syncedStatus);

        // 检查点数匹配
        if (imagePoints.size() != modelPoints.size())
        {
            std::cerr << "Warning: Point count mismatch ("
                      << imagePoints.size() << " vs "
                      << modelPoints.size() << ")" << std::endl;
            continue;
        }

        // ePnP姿态估计
        auto epnpResult = PnPSolver::solveEPnP(
            modelPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs);

        // 转换到世界坐标系
        cv::Vec3d targetPosWorld = PnPSolver::transformToWorld(
            epnpResult.rotation,
            epnpResult.translation,
            syncedCamera);

        // 计算特征尺度l
        double l = cv::norm(targetPosWorld - cv::Vec3d(
                                                 syncedCamera.translation.x,
                                                 syncedCamera.translation.y,
                                                 syncedCamera.translation.z));

        // 构建测量向量
        cv::Mat measurement = (cv::Mat_<double>(4, 1) << targetPosWorld[0],
                               targetPosWorld[1],
                               targetPosWorld[2],
                               l);

        // --- 修改：第一帧执行初始化，后续帧执行预测更新 ---
        if (!isInitialized)
        {
            // 用第一帧的测量值来初始化位置
            initialState.at<double>(0) = targetPosWorld[0];
            initialState.at<double>(1) = targetPosWorld[1];
            initialState.at<double>(2) = targetPosWorld[2];
            initialState.at<double>(stateDim - 1) = 0.2; // 尺度 l

            kf.init(initialState, initialCov);
            isInitialized = true;
            std::cout << "KF Initialized at: " << targetPosWorld << std::endl;
            continue; // 第一帧跳过 predict/update
        }

        // 卡尔曼滤波
        kf.predict();
        kf.update(measurement);
        estimatedState=kf.getState();
        if (flag % 10 == 0)
        {
            std::cout << flag << ". " << "Estimtated Position: "
                      << estimatedState.at<double>(0) << ","
                      << estimatedState.at<double>(1) << ","
                      << estimatedState.at<double>(2) << std::endl;
            std::cout << "Estimated Scale: "
                      << estimatedState.at<double>(stateDim - 1) << std::endl;
        }
        flag++;
    }

    return 0;
}