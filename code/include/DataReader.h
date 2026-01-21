#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
struct StatusData // 构建数据格式
{
    double record_time;
    double msg_time;
    struct
    {
        bool has_body;
        double body_x, body_y;
        struct Propeller//螺旋桨的位姿
        {
            int id;
            double x, y;
            int status;
            std::string status_desc;
        };
        std::vector<Propeller> propellers;
    }drone_state;
};