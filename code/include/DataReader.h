#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
struct Propeller // 螺旋桨的位姿
{
    int id;
    double x, y;
    int status;
    std::string status_desc;
};

struct DroneState
{
    bool has_body;
    double body_x, body_y;
    std::vector<Propeller> propellers;
};

struct StatusData // 构建位姿数据格式
{
    double record_time;
    double msg_time;
    DroneState drone_state;
};

struct PoseData {
    std::string frame_id;
    std::string child_frame_id;
    double record_time;
    double tf_time;
    struct {
        double x, y, z;
    } translation;
    struct {
        double x, y, z, w;
    } rotation;
};

class DataReader
{
public:
    static std::vector<StatusData> readStatusFile(const std::string &filename);
    static std::vector<PoseData> readPoseFile(const std::string &filename);
};