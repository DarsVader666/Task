#include <iostream>
#include <opencv2/opencv.hpp>
#include "DataReader.h"
#include "PnPSolver.h"
#include "KalmanFilter.h"
#include "utils.h"
double preTime = 0;
double preY,preX,preZCam,preZ;
int main()
{

    std::cout << "和一位";
    // 1. 读取数据文件
    auto statusData = DataReader::readStatusFile("../../data_status_slow.jsonl");
    std::cout << "Found " << statusData.size() << " status records" << std::endl;
    auto cameraData = DataReader::readPoseFile("../../data_tf_camera_slow.jsonl");
    std::cout << "Found " << cameraData.size() << " camera records" << std::endl;
    auto uavData = DataReader::readPoseFile("../../data_tf_uav_slow.jsonl");
    std::cout << "Found " << uavData.size() << " uav records" << std::endl;

    std::ofstream outfile("/home/d1c/Git/Task/error.jsonl", std::ios::out | std::ios::trunc);
    if (!outfile.is_open())
    {
        std::cerr << "无法打开文件 error.jsonl" << std::endl;
        return 1;
    }
    // 2. 相机参数设置（需要根据实际相机标定）
    double horizontalFOV = 1.732; // 弧度
    double imageWidth = 1920.0;
    double imageHeight = 1080.0;

    // 计算焦距
    // double fx = imageWidth / (2.0 * tan(horizontalFOV / 2.0));
    double fx = 815.0;
    double fy = fx;

    double cx = imageWidth / 2.0;  // 960
    double cy = imageHeight / 2.0; // 540

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);

    // 3. 无人机3D模型点（本体坐标系，单位：米）
    double x = 0.5;
    // 方案 2: ID1右上, 顺时针
    std::vector<cv::Point3f> modelPoints = {cv::Point3f(0,0,0), cv::Point3f(-x,-x,0), cv::Point3f(-x,x,0), cv::Point3f(x,x,0), cv::Point3f(x,-x,0)};
    int m = 2;
    int stateDim = 3 * (m + 1) + 1;
    int measureDim = 3; // l不测量
    double dt = 0.033;
    KalmanFilter kf(stateDim, measureDim, m);

    // 初始状态和协方差
    cv::Mat Q = cv::Mat::eye(stateDim, stateDim, CV_64F) * 1;     // 过程噪声
    Q.at<double>(stateDim - 1, stateDim - 1) = 0;               // 静态尺度没有过程噪声
    cv::Mat R = cv::Mat::eye(measureDim, measureDim, CV_64F) * 1; // 测y量噪声
    R.at<double>(0, 0) = 5*1e1;                                    // X 的测量噪声，给大一点（不信任）
    R.at<double>(1, 1) = 5*1e1;                                      // Y 的测量噪声，给小一点（信任）
    R.at<double>(2, 2) = 5*1e1;                                     // Z 的测量噪声，给最大（最不信任）

    Q.at<double>(0, 0) = 0;//1e-4; // 位置x
    Q.at<double>(1, 1) = 0;//1e-4; // 位置y
    Q.at<double>(2, 2) = 0;//1e-7; // 位置z

    Q.at<double>(3, 3) = 0;//1e-3; // 速度x
    Q.at<double>(4, 4) = 0;//1e-3; // 速度y3
    Q.at<double>(5, 5) = 0;//1e-3; // 速度z

    Q.at<double>(6, 6) = 5 * 1e-2; // 加速度x
    Q.at<double>(7, 7) = 5 * 1e-2; // 加速度y
    Q.at<double>(8, 8) = 5 * 1e-2; // 加速度z

    // for (int i = 9; i < 12; i++) Q.at<double>(i, i) = 1.0;
    kf.setProcessNoiseCov(Q);
    kf.setMeasurementNoiseCov(R);

    // --- 新增：初始化状态向量和协方差矩阵 ---
    cv::Mat initialState = cv::Mat::zeros(stateDim, 1, CV_64F) * 1;
    initialState.at<double>(stateDim - 1) = 0.33; // 尺度 l
    cv::Mat initialCov = cv::Mat::eye(stateDim, stateDim, CV_64F) * 1;
    initialCov.at<double>(2,2)=1e2;
    initialCov.at<double>(stateDim-1,stateDim-1)=2.5*1e2;
    // 注意：此时 state_ 依然是空的，必须调用 init
    bool isInitialized = false;
    int flag = 1;
    cv::Mat estimatedState;
    double distance = 0; // 相机与被追踪无人机的真实距离
    for (const auto &status : statusData)
    {
        StatusData syncedStatus;
        PoseData syncedCamera, syncedUav;

        synchronizeData(
            status.record_time,
            statusData,
            cameraData,
            uavData,
            syncedStatus,
            syncedCamera,
            syncedUav);

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

        // 2. 获取当前相机的位置得到p_O
        cv::Vec3d camPos(syncedCamera.translation.x, syncedCamera.translation.y, syncedCamera.translation.z);
        // 构建测量向量p_0
        cv::Mat measurement = (cv::Mat_<double>(3, 1) << camPos[0], camPos[1], camPos[2]);

        // ePnP姿态估计
        auto epnpResult = PnPSolver::solveEPnP(
            modelPoints,
            imagePoints,
            cameraMatrix,
            distCoeffs);

        // 转换到世界坐标系,得到p_T
        cv::Vec3d targetPosWorld = PnPSolver::transformToWorld(
            epnpResult.rotation,
            epnpResult.translation,
            syncedCamera);
        // 获得归一化的方向向量 q_ln(从相机指向目标)
        cv::Vec3d direction = targetPosWorld - camPos; // 方向向量
        double dist_pnp = cv::norm(direction);         // 计算模
        cv::Vec3d q_ln = direction;
        q_ln = direction / (x*100); // 归一化
        

        // 计算特征尺度l
        double l = cv::norm(targetPosWorld - cv::Vec3d(
                                                 syncedCamera.translation.x,
                                                 syncedCamera.translation.y,
                                                 syncedCamera.translation.z));

        // 构建测量向量
        /* cv::Mat measurement = (cv::Mat_<double>(4, 1) << targetPosWorld[0],
                                targetPosWorld[1],
                                targetPosWorld[2],
                                l);*/

        // --- 修改：第一帧执行初始化，后续帧执行预测更新 ---
        if (!isInitialized)
        {
            // 用第一帧的测量值来初始化位置
            initialState.at<double>(0) = targetPosWorld[0];
            initialState.at<double>(1) = targetPosWorld[1];
            initialState.at<double>(2) = targetPosWorld[2];

            preX=targetPosWorld[0];
            preY=targetPosWorld[1];
            preZ=targetPosWorld[2];

            kf.init(initialState, initialCov);
            isInitialized = true;
            std::cout << "KF Initialized at: " << targetPosWorld << std::endl;
            preTime = status.record_time;
            continue; // 第一帧跳过 predict/update
        }
        
        
        double current_time = status.record_time;
        double dt = current_time - preTime;



        // 2. 安全检查：防止时间戳重复或乱序导致 dt <= 0
        if (dt <= 0.001)
        {
            // 如果时间差太小，可能是重复帧，跳过预测直接用上一帧状态，或者给个极小值
            // dt = 0.001;
            continue;
        }


       

        // 重要：在当前循环结束前，更新“上一帧”的 PnP 原始值
        preX = targetPosWorld[0];
        preY = targetPosWorld[1];
        preZ = targetPosWorld[2];
        // ------------------------------------------



        // 卡尔曼滤波
        kf.predict(dt);
        kf.updateMeasurementMatrix(q_ln);
        
        kf.update(measurement);
        estimatedState = kf.getState();
        double final_l = estimatedState.at<double>(3 * (m + 1));

        if (flag)
        {
            std::cout<<preZ<<std::endl;
            std::cout << status.record_time << "s" << ": "
                      << estimatedState.at<double>(0) << ","
                      << estimatedState.at<double>(1) << ","
                      << estimatedState.at<double>(2) << std::endl;
            std::cout << flag << ". " << "Estimtated Position: "
                      << estimatedState.at<double>(0) << ","
                      << estimatedState.at<double>(1) << ","
                      << estimatedState.at<double>(2) << std::endl;
            std::cout << "Estimated Scale: "
                      << final_l << std::endl;
        }
        nlohmann::json record;
        // 添加数据
        record["record_time"] = status.record_time; // 示例时间戳
        nlohmann::json errorObj;
        errorObj["x"] = (estimatedState.at<double>(0) - syncedUav.translation.x);
        errorObj["y"] = (estimatedState.at<double>(1) - syncedUav.translation.y);
        errorObj["z"] = (estimatedState.at<double>(2) - syncedUav.translation.z);

        record["error"] = errorObj;

        nlohmann::json ll;
        record["l"] = final_l/(100*x);
        outfile << record.dump() << std::endl;
        preTime = status.record_time;
        flag++;
        // 调试用：在循环里直接打印 targetPosWorld，不经过滤波器
        // std::cout << "PnP Raw Pos: " << targetPosWorld << std::endl;
    }

    return 0;
}