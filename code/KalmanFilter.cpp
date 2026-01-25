#include "KalmanFilter.h"

KalmanFilter::KalmanFilter(int stateDim, int measureDim, int m) : stateDim_(stateDim), measureDim_(measureDim), m_(m)
{
    transitionMatrix_ =cv ::Mat::eye(stateDim_,stateDim,CV_64F);
    initMeasurementMatrix();
}

void KalmanFilter::init(const cv::Mat &initialState, const cv::Mat &initialCov) // 初始化卡尔曼滤波器的初始状态与协方差矩阵
{
    state_ = initialState.clone();
    errorCov_ = initialCov.clone();
}

void KalmanFilter::updateTransitionMatrix(double dt)
{
    transitionMatrix_ = cv::Mat::zeros(stateDim_, stateDim_, CV_64F);

    // 重新计算幂级数系数
    for (int k = 0; k <= m_; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = k; j <= m_; j++)
            {
                // 注意：这里要用传入的 dt
                double coeff = pow(dt, j - k) / tgamma(j - k + 1); 
                transitionMatrix_.at<double>(3 * k + i, 3 * j + i) = coeff; 
            }
        }
    }
    // 静态特征尺度 l 保持不变
    transitionMatrix_.at<double>(stateDim_ - 1, stateDim_ - 1) = 1.0;
}

/*void KalmanFilter::initTransitionMatrix()
{
    // 拿第一体的结果计算转移矩阵
    transitionMatrix_ = cv::Mat::zeros(stateDim_, stateDim_, CV_64F);

    for (int k = 0; k <= m_; k++)
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = k; j <= m_; j++)
            {
                double coeff = pow(dt_, j - k) / tgamma(j - k + 1);         // tgamma(n)=(n-1)!
                transitionMatrix_.at<double>(3 * k + i, 3 * j + i) = coeff; //+i即为构造对角线结构
            }
        }
    }
    // 静态特征尺度l，最右下角那个是1
    transitionMatrix_.at<double>(stateDim_ - 1, stateDim_ - 1) = 1.0;
}*/

void KalmanFilter::initMeasurementMatrix() // 初始化观测矩阵
{                                          // H_k=[I_3,0,...,-q_{ln}(k)]
    measurementMatrix_ = cv::Mat::zeros(measureDim_, stateDim_, CV_64F);

    for (int i = 0; i < 3; i++)
    {
        measurementMatrix_.at<double>(i, i) = 1.0;
    }
    // 特征尺度测量
    measurementMatrix_.at<double>(3, stateDim_ - 1) = 1.0;
} // 用特征尺度替代q_{ln}(k)

void KalmanFilter::predict(double dt) // 计算先验误差协方差
{
    // 状态预测1
    updateTransitionMatrix(dt);

    // 第二步：执行标准的预测步骤
    // 状态预测 x = F * x
    state_ = transitionMatrix_ * state_;
    // 协方差矩阵预测2
    errorCov_ = transitionMatrix_ * errorCov_ * transitionMatrix_.t() + processNoiseCov_;
}

void KalmanFilter::update(const cv::Mat &measurement) // 校正
{
    // 计算卡尔曼增益3
    cv::Mat S = measurementMatrix_ * errorCov_ * measurementMatrix_.t() + measurementNoiseCov_;
    kalmanGain_ = errorCov_ * measurementMatrix_.t() * S.inv();

    // 计算后验估计4
    cv::Mat innovation = measurement - measurementMatrix_ * state_;
    state_ = state_ + kalmanGain_ * innovation;

    // 更新误差协方差
    cv::Mat I = cv::Mat::eye(stateDim_, stateDim_, CV_64F);
    errorCov_ = (I - kalmanGain_ * measurementMatrix_) * errorCov_;
}

cv::Mat KalmanFilter::getState() const // 读取当前卡尔曼滤波器的各参数（状态与误差协方差）
{
    return state_.clone();
}

void KalmanFilter::setProcessNoiseCov(const cv::Mat &Q) // 设定过程噪声协方差
{
    Q.copyTo(processNoiseCov_);
}

void KalmanFilter::setMeasurementNoiseCov(const cv::Mat &R) // 设定测量噪声协方差
{
    R.copyTo(measurementNoiseCov_);
}
void KalmanFilter::updateMeasurementMatrix(const cv::Vec3d &q_ln) // 更新测量矩阵
{
    // 重新构造H矩阵[I,0,0, ... ,0,0,q_ln]
    measurementMatrix_ = cv::Mat::zeros(3, 3 * (m_ + 1) + 1, CV_64F);

    measurementMatrix_.at<double>(0, 0) = 1.0;
    measurementMatrix_.at<double>(1, 1) = 1.0;
    measurementMatrix_.at<double>(2, 2) = 1.0;

    // 最后一列
    measurementMatrix_.at<double>(0, 3 * (m_ + 1)) = -q_ln[0];
    measurementMatrix_.at<double>(1, 3 * (m_ + 1)) = -q_ln[1];
    measurementMatrix_.at<double>(2, 3 * (m_ + 1)) = -q_ln[2];
}
cv::Mat KalmanFilter::getMeasurementMatrix()
{
    return measurementMatrix_;
}

cv::Mat KalmanFilter::getErrorCov()
{
    return errorCov_;
}

cv::Mat KalmanFilter::getMeasurementNoiseCov()
{
    return measurementNoiseCov_;
}