#pragma once
#include <iostream>
#include <fstream>
#include<string>
#include <map>
#include "common/real.h"
#define GET_PARAM(c,p) c.getParam(#p,p)

class Config
{
private:
    std::map<std::string,std::string> paramMap;
    bool checkParam(std::string);
public:
    Config(std::string configFile);
    void getParam(std::string, unsigned int&);
    void getParam(std::string, int&);
    void getParam(std::string, float&);
    void getParam(std::string, bool&);
    void getParam(std::string, float3&);
    ~Config();
};