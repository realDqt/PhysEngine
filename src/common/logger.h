#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <map>

#define LOG_THRES LogTypeKD::Debug

//// filename
#ifdef __APPLE__
#define __FILENAME__ (strrchr(__FILE__, '/') + 1)
#else
#define __FILENAME__ (strrchr(__FILE__, '\\') + 1)
#endif

char* getTimeString();

enum LogTypeKD{
    Debug=0,
    Info,
    Warn,
    Error,
    Fatal
};

extern std::string LOG_COLORS[5];
extern std::string LOG_COLOR_RESET;


//// ostream  log
#define LOG_OSTREAM(lt) \
    if constexpr (lt < LOG_THRES) ; \
    else std::cout<<LOG_COLORS[(int)lt] << '[' << getTimeString() << "][" << __FILENAME__ << ":" << __LINE__ << "] " << LOG_COLOR_RESET
    
#define LOG_OSTREAM_DEBUG LOG_OSTREAM(LogTypeKD::Debug)
#define LOG_OSTREAM_INFO LOG_OSTREAM(LogTypeKD::Info)
#define LOG_OSTREAM_WARN LOG_OSTREAM(LogTypeKD::Warn)
#define LOG_OSTREAM_ERROR LOG_OSTREAM(LogTypeKD::Error)


//// printf log
#define LOG_PRINT(lt, format, ...) \
    if(lt < LOG_THRES) ; \
    else { printf("%s[%s][%s:%d]%s "  format, LOG_COLORS[(int)lt], getTimeString(), __FILENAME__, __LINE__, LOG_COLOR_RESET, ##__VA_ARGS__); }
    
#define LOG_PRINT_DEBUG(format, ...) LOG_PRINT(LogTypeKD::Debug)
#define LOG_PRINT_INFO(format, ...) LOG_PRINT(LogTypeKD::Info)
#define LOG_PRINT_WARN(format, ...) LOG_PRINT(LogTypeKD::Warn)
#define LOG_PRINT_ERROR(format, ...) LOG_PRINT(LogTypeKD::Error)

/**
 * @brief The Logger class is responsible for logging messages to a file.
 *
 * The Logger class provides functionality to log messages of different log types to a file.
 * It allows specifying the log type (such as Debug, Info, Warn, or Error) and the message to be logged.
 * The Logger class also supports setting a maximum file size for the log file.
 */
class Logger {
public:
    Logger(const std::string& logFileName, std::size_t maxFileSize);

    ~Logger();

    void Log(const LogTypeKD& type, const std::string& message);

private:
    std::string logFileName;
    std::ofstream logFile;
    std::size_t maxFileSize;
    std::size_t currentFileSize;

    std::map<LogTypeKD, std::string> logTypeToString = {
        {LogTypeKD::Error, "Error"},
        {LogTypeKD::Warn, "Waring"},
        {LogTypeKD::Info, "Info"}
    };

    void OpenLogFile();

    std::size_t GetFileSize(const std::string& fileName);

    void CheckFileSize();

    void RotateLogFile();

    std::string GenerateNewFileName();

    void LogMessage(const LogTypeKD& type, const std::string& message);
};

extern Logger logger;