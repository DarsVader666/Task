#pragma once
#include <opencv2/opencv.hpp>

class KalmanFilter
{
public:
    KalmanFilter(int stateDim, int measureDim, int m);

    void init(const cv::Mat& initialState, const cv::Mat& initialCov);
    void predict(double dt);
    void update(const cv::Mat& measurement);
    cv::Mat getState() const;
    cv::Mat getMeasurementMatrix();
    cv::Mat getErrorCov();
    cv::Mat getMeasurementNoiseCov();

    void setProcessNoiseCov(const cv::Mat& Q);//过程噪声协方差矩阵
    void setMeasurementNoiseCov(const cv::Mat& R);
    void updateMeasurementMatrix(const cv::Vec3d& q_ln);//动态更新H矩阵 
    

private:
    int stateDim_;//状态参量维度
    int measureDim_;//测量维度
    int m_;//阶数
    double dt_;//间隔时间

    cv::Mat state_;//状态矩阵
    cv::Mat errorCov_;//误差协方差矩阵
    cv::Mat transitionMatrix_;//转移矩阵\Phi
    cv::Mat measurementMatrix_;//测量矩阵H
    
    cv::Mat processNoiseCov_;//过程噪声协方差矩阵
    cv::Mat measurementNoiseCov_;//过程噪声协方差矩阵
    cv::Mat kalmanGain_;//卡尔曼增益矩阵

    void initTransitionMatrix();
    void initMeasurementMatrix();
    void updateTransitionMatrix(double dt);
    
};