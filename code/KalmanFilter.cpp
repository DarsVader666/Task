#include "KalmanFilter.h"

KalmanFilter::KalmanFilter(int stateDim, int measureDim, int m, double dt) : stateDim_(stateDim), measureDim_(measureDim), m_(m), dt_(dt)
{
    initTransitionMatrix();

    initMeasurementMatrix();
}

void KalmanFilter::init(const cv::Mat &initialState, const cv::Mat &initialCov)
{
    state_ = initialState.clone();
    errorCov_ = initialCov.clone();
}

void KalmanFilter::initTransitionMatrix()
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
}

void KalmanFilter::initMeasurementMatrix() // 计算观测矩阵
{                                          // H_k=[I_3,0,...,-q_{ln}(k)]
    measurementMatrix_ = cv::Mat::zeros(measureDim_, stateDim_, CV_64F);
    
    for (int i = 0; i < 3; i++)
    {
        measurementMatrix_.at<double>(i, i) = 1.0;
    }
    // 特征尺度测量
    measurementMatrix_.at<double>(3, stateDim_ - 1) = 1.0;
} // 用特征尺度替代q_{ln}(k)

void KalmanFilter::predict()
{
    // 状态预测1
    state_ = transitionMatrix_ * state_;

    // 协方差矩阵预测2
    errorCov_ = transitionMatrix_ * errorCov_ * transitionMatrix_.t() + processNoiseCov_;
}

void KalmanFilter::update(const cv::Mat &measurement) // 校正
{
    // 计算卡尔曼增益3
    cv::Mat S = measurementMatrix_ * errorCov_ * measurementMatrix_.t() + measurementNoiseCov_;
    kalmanGain_ = errorCov_ * measurementMatrix_.t() * S.inv();

    // 后验估计4
    cv::Mat innovation = measurement - measurementMatrix_ * state_;
    state_ = state_ + kalmanGain_ * innovation;

    // 协方差更新5
    cv::Mat I = cv::Mat::eye(stateDim_, stateDim_, CV_64F);
    errorCov_ = (I - kalmanGain_ * measurementMatrix_) * errorCov_;
}

cv::Mat KalmanFilter::getState() const
{
    return state_.clone();
}

void KalmanFilter::setProcessNoiseCov(const cv::Mat &Q)
{
    Q.copyTo(processNoiseCov_);
}

void KalmanFilter::setMeasurementNoiseCov(const cv::Mat &R)
{
    R.copyTo(measurementNoiseCov_);
}