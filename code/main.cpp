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
    // 2. 相机参数设置
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
    cv::Mat Q = cv::Mat::zeros(stateDim, stateDim, CV_64F) ;
    cv::Mat R = cv::Mat::eye(measureDim, measureDim, CV_64F) * 1; // 测量噪声
    R.at<double>(0, 0) = 5*1e1;                                    
    R.at<double>(1, 1) = 5*1e1;                                      
    R.at<double>(2, 2) = 5*1e1;                                     

    //仅最高阶有过程噪声
    Q.at<double>(3*m, 3*m) = 5 * pow(10,-(4-m)); // 速度x
    Q.at<double>(3*m+1, 3*m+1) = 5 * pow(10,-(4-m)); // 速度y
    Q.at<double>(3*m+2, 3*m+2) = 5 * pow(10,-(4-m)); // 速度z



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
        // 转换格式
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
        q_ln = direction / (x*100); // 降低权重
        

        // 计算特征尺度l
        double l = cv::norm(targetPosWorld - cv::Vec3d(
                                                 syncedCamera.translation.x,
                                                 syncedCamera.translation.y,
                                                 syncedCamera.translation.z));


        //第一帧执行初始化，后续帧执行预测更新
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
            continue; // 非第一帧跳过
        }
        
        
        double current_time = status.record_time;
        double dt = current_time - preTime;



        // 防止时间戳重复或乱序导致 dt <= 0
        if (dt <= 0.001)
        {
            continue;
        }


        // 卡尔曼滤波
        kf.predict(dt);
        kf.updateMeasurementMatrix(q_ln);
        
        kf.update(measurement);
        estimatedState = kf.getState();
        double final_l = estimatedState.at<double>(3 * (m + 1));

        //打印结果
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
        // 记录数据
        nlohmann::json record;
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
    }

    return 0;
}