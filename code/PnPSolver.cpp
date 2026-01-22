#include "PnPSolver.h"
#include <opencv2/calib3d.hpp>

PnPSolver::EPnPResult PnPSolver::solveEPnP(
    const std::vector<cv::Point3f> &objectPoints,
    const std::vector<cv::Point2f> &imagePoints,
    const cv::Mat &cameraMatrix,
    const cv::Mat &distCoeffs)
{

    EPnPResult result;
    cv::Mat rvec;

    // 检查输入有效性
    if (objectPoints.empty() || imagePoints.empty())
    {
        throw std::runtime_error("输入点集为空");
    }
    if (objectPoints.size() != imagePoints.size())
    {
        throw std::runtime_error("3D点与2D点数量不匹配");
    }
    if (cameraMatrix.empty() || cameraMatrix.rows != 3 || cameraMatrix.cols != 3)
    {
        throw std::runtime_error("相机内参矩阵无效");
    }

    bool success = cv::solvePnP(objectPoints,
                                imagePoints,
                                cameraMatrix,
                                distCoeffs,
                                rvec,
                                result.translation,
                                false,
                                cv::SOLVEPNP_EPNP); // 把像素点求出旋转向量rvec和平移向量tvec
    /*cv::Rodrigues(rvec, result.rotation); // 旋转向量转化为旋转矩阵
    return result;*/
    if (!success)
    {
        throw std::runtime_error("ePnP求解失败");
    }

    // 将旋转向量转换为旋转矩阵
    cv::Rodrigues(rvec, result.rotation);

    // 检查结果有效性
    if (result.rotation.empty() || result.rotation.rows != 3 || result.rotation.cols != 3)
    {
        throw std::runtime_error("ePnP返回空结果");
    }

    return result;
}
//
cv::Mat PnPSolver::quatToRotMat(const cv::Vec4d &q)
{
    double x = q[0], y = q[1], z = q[2], w = q[3];
    return (cv::Mat_<double>(3, 3) << 1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w,
            2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w,
            2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y);
}

cv::Vec3d PnPSolver::transformToWorld(
    const cv::Mat &R_target_cam,
    const cv::Vec3d &tvec,
    const PoseData &cameraPose)
{
    cv::Mat R_cam_world = quatToRotMat({cameraPose.rotation.x,
                                        cameraPose.rotation.y,
                                        cameraPose.rotation.z,
                                        cameraPose.rotation.w});

    cv::Mat t_cam_world = (cv::Mat_<double>(3, 1) << // 摄像机在世界坐标系下的位移
                               cameraPose.translation.x,
                           cameraPose.translation.y,
                           cameraPose.translation.z);

    // 将 Vec3d 转换为 Mat (3x1 矩阵)
    // cv::Mat tvec_mat = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
    // 3. PnP 算出的 tvec 就是点在相机系下的坐标 P_c
    cv::Mat p_cam = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
    cv::Mat p_world = R_cam_world * p_cam+t_cam_world;
    return cv::Vec3d(
        p_world.at<double>(0),
        p_world.at<double>(1),
        p_world.at<double>(2));
}