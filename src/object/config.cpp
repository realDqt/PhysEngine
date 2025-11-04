#include "object/config.h"
#include <sstream>
Config::Config(std::string configFile)
{
    std::ifstream in(configFile);
    std::string str, tmp;

    if (in.is_open())
    {
        while (!in.eof())
        {
            std::getline(in, str, '=');
            
            tmp = str;
            std::getline(in, str);
            paramMap[tmp] = str;
        }

        in.close();
    }
    else
    {
        std::cout << "fail to open config file" << std::endl;
        exit(-1);
    }
}
bool Config::checkParam(std::string paramName)
{
    if (paramMap.find(paramName) == paramMap.end())
    {
        std::cout << "no such parameter in config file: " << paramName <<std::endl;
        exit(-1);
        return false;
    }
    else
        return true;
}

void Config::getParam(std::string paramName, unsigned int &param)
{
    if (!checkParam(paramName))
        return;
    param = atoi(paramMap[paramName].data());
}

void Config::getParam(std::string paramName, int &param)
{
    if (!checkParam(paramName))
        return;
    param = atoi(paramMap[paramName].data());
}

void Config::getParam(std::string paramName, float &param)
{
    if (!checkParam(paramName))
        return;
    param = atof(paramMap[paramName].data());
}

void Config::getParam(std::string paramName, bool &param)
{
    if (!checkParam(paramName))
        return;
    param = !paramMap[paramName].compare("true");
}

void Config::getParam(std::string paramName, float3 &param)
{
    if (!checkParam(paramName))
        return;
    float tempArray[3];
    std::string dataStr = paramMap[paramName];
    std::istringstream iss(dataStr);

    std::string tempStr;
    for (int i = 0; i < 3; i++) {
        getline(iss, tempStr, ',');
        tempArray[i] = atof(tempStr.data());
    }
    param = make_float3(tempArray[0], tempArray[1], tempArray[2]);
}

Config::~Config() {}