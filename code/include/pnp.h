#ifndef PNP_H
#define PNP_H
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <iostream>
#include <vector>

class pnp{
    private:
    std::vector<cv::Point3f> pts3d; // 目标 3D 坐标
    std::vector<cv::Point2f> pts2d; // JSONL 里的像素坐标
    cv::Mat K, dist;                // 内参和畸变
    cv::Mat rvec, tvec;
    public:




};

#endif
