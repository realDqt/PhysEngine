#include<common/logger.h>
#include <sstream>

std::string LOG_COLORS[5] =
{
    "\033[32m",
    "\033[0m",
    "\033[33m",
    "\033[31m",
    "\033[35m",
};

std::string LOG_COLOR_RESET = "\033[0m";

#include <stdint.h>
#if defined __GNUC__ || defined LINUX
#include <sys/time.h>
#define MY_VA_LIST _G_va_list
#else
#include <windows.h>
#define MY_VA_LIST va_list
#endif

//// time
#define FORMAT_TIME_SIZE 24
char* getDateString(){
    char *pszFormatTime = new char[FORMAT_TIME_SIZE];
#if defined __GNUC__ || defined LINUX
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    struct tm *current_time = localtime(&tv.tv_sec);
    sprintf(pszFormatTime,
        "%04d-%02d-%02d %02d:%02d:%02d.%03d",
        current_time->tm_year + 1900,
        current_time->tm_mon + 1,
        current_time->tm_mday,
        current_time->tm_hour,
        current_time->tm_min,
        current_time->tm_sec,
        tv.tv_usec / 1000);
#else
    SYSTEMTIME sys_time;
    GetLocalTime(&sys_time);
#if _MSC_VER
    sprintf_s(pszFormatTime,
        FORMAT_TIME_SIZE,
        "%04d-%02d-%02d %02d:%02d:%02d.%03d",
        sys_time.wYear,
        sys_time.wMonth,
        sys_time.wDay,
        sys_time.wHour,
        sys_time.wMinute,
        sys_time.wSecond,
        sys_time.wMilliseconds);
#else
    sprintf(pszFormatTime,
        "%04d-%02d-%02d %02d:%02d:%02d.%03d",
        sys_time.wYear,
        sys_time.wMonth,
        sys_time.wDay,
        sys_time.wHour,
        sys_time.wMinute,
        sys_time.wSecond,
        sys_time.wMilliseconds);
#endif
#endif
    return pszFormatTime;
}


char* getTimeString(){
    char *pszFormatTime = new char[FORMAT_TIME_SIZE];
#if defined __GNUC__ || defined LINUX
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    struct tm *current_time = localtime(&tv.tv_sec);
    sprintf(pszFormatTime,
        "%02d:%02d:%02d.%03d",
        current_time->tm_hour,
        current_time->tm_min,
        current_time->tm_sec,
        tv.tv_usec / 1000);
#else
    SYSTEMTIME sys_time;
    GetLocalTime(&sys_time);
#if _MSC_VER
    sprintf_s(pszFormatTime,
        FORMAT_TIME_SIZE,
        "%02d:%02d:%02d.%03d",
        sys_time.wHour,
        sys_time.wMinute,
        sys_time.wSecond,
        sys_time.wMilliseconds);
#else
    sprintf(pszFormatTime,
        "%02d:%02d:%02d.%03d",
        sys_time.wHour,
        sys_time.wMinute,
        sys_time.wSecond,
        sys_time.wMilliseconds);
#endif
#endif
    return pszFormatTime;
}

Logger::Logger(const std::string& logFileName, std::size_t maxFileSize)
    : logFileName(logFileName), maxFileSize(maxFileSize * 1024), currentFileSize(0) {
    OpenLogFile();
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

void Logger::Log(const LogTypeKD& type, const std::string& message) {
    if (logFile.is_open()) {
        CheckFileSize();
        LogMessage(type, message);
    }
}

void Logger::OpenLogFile() {
    logFile.open(logFileName, std::ios::out | std::ios::app);
    if (!logFile.is_open()) {
        std::cerr << "Error: Cannot open log file " << logFileName << std::endl;
    }
    else {
        currentFileSize = GetFileSize(logFileName);
    }
}

std::size_t Logger::GetFileSize(const std::string& fileName) {
    std::ifstream file(fileName, std::ios::binary | std::ios::ate);
    return static_cast<std::size_t>(file.tellg());
}

void Logger::CheckFileSize() {
    if (currentFileSize >= maxFileSize) {
        logFile.close();
        RotateLogFile();
    }
}

void Logger::RotateLogFile() {
    std::string newFileName = GenerateNewFileName();
    std::rename(logFileName.c_str(), newFileName.c_str());
    OpenLogFile();
}

std::string Logger::GenerateNewFileName() {
    std::ostringstream newFileName;
    time_t rawTime;
    struct tm timeInfo;
    char timestamp[80];

    time(&rawTime);
    localtime_s(&timeInfo, &rawTime);
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", &timeInfo);

    newFileName << logFileName << "_" << timestamp;
    return newFileName.str();
}

void Logger::LogMessage(const LogTypeKD& type, const std::string& message) {
    time_t rawTime;
    struct tm timeInfo;
    char timestamp[80];

    time(&rawTime);
    localtime_s(&timeInfo, &rawTime);
    strftime(timestamp, sizeof(timestamp), "[%Y-%m-%d %H:%M:%S]", &timeInfo);

    char prefix[1024];
    std::sprintf(prefix, "[%s][%s][%s:%d] ", logTypeToString[type].c_str(), timestamp, __FILENAME__, __LINE__);

    logFile << prefix << message << std::endl;
    currentFileSize += message.length() + strlen(prefix);

    if (currentFileSize >= maxFileSize) {
        logFile.close();
        RotateLogFile();
    }
}