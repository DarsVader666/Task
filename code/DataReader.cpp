#include "DataReader.h"
#include <fstream>
#include<iostream>
#include <nlohmann/json.hpp>
#include<filesystem>

namespace fs = std::filesystem;


using json = nlohmann::json;

std::vector<StatusData> DataReader::readStatusFile(const std::string &filename)
{
    std::vector<StatusData> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        auto j = json::parse(line);
        StatusData entry;
        entry.record_time = j["record_time"];
        entry.msg_time = j["msg_time"];
        entry.drone_state.has_body = j["drone_state"]["has_body"];
        entry.drone_state.body_x = j["drone_state"]["body_x"];
        entry.drone_state.body_y = j["drone_state"]["body_y"];

        for (auto &prop : j["drone_state"]["propellers"])
        {
            Propeller p;
            p.id = prop["id"];
            p.x = prop["x"];
            p.y = prop["y"];
            p.status = prop["status"];
            p.status_desc = prop["status_desc"];
            entry.drone_state.propellers.push_back(p);
        }
        data.push_back(entry);
    }
    return data;
}

std::vector<PoseData> DataReader:: readPoseFile(const std::string &filename)
{
    std::vector<PoseData> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        auto j = json::parse(line);
        PoseData entry;
        entry.frame_id = j["frame_id"];
        entry.child_frame_id = j["child_frame_id"];
        entry.record_time = j["record_time"];
        entry.tf_time = j["tf_time"];
        entry.translation.x = j["translation"]["x"];
        entry.translation.y = j["translation"]["y"];
        entry.translation.z = j["translation"]["z"];
        entry.rotation.x = j["rotation"]["x"];
        entry.rotation.y = j["rotation"]["y"];
        entry.rotation.z = j["rotation"]["z"];
        entry.rotation.w = j["rotation"]["w"];
        data.push_back(entry);
    }
    return data;
}

