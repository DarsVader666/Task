#include<iostream>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include<nlohmann/json.hpp>
#include<vector>

using json = nlohmann::json;
using namespace Eigen;

const int m =1;//k = m
const int dim = 3 * (m+1) + 1;
int main(){
    VectorXd x =VectorXd::Zero(dim);
    MatrixXd P = MatrixXd::Identity(dim,dim) * 1.0;//
    MatrixXd Q = MatrixXd::Identity(dim,dim) * 0.01;//过程噪声协方差矩阵
    MatrixXd R = MatrixXd::Identity(3,3) * 0.1;//观测噪声协方差矩阵



    return 0;
}