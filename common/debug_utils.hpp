#ifndef __DEBUG_UTILS_HPP__
#define __DEBUG_UTILS_HPP__
#include <iostream>
#include <iomanip>
#include <string>

#ifndef VERBOSE
#define VERBOSE 1
#endif

// General logging macro
#if VERBOSE >= 1
    #define LOG_VERBOSE(level, msg) if (level <= VERBOSE) { std::cout << "[log]  " << msg << std::endl; }
#else
    #define LOG_VERBOSE(level, msg) ((void)0) // No-op
#endif

// Conditional verbose logging macro
#if VERBOSE >= 1
    #define LOG_VERBOSE_IF(level, condition, msg) \
        if (level <= VERBOSE && (condition)) { std::cout << "[log]  " << msg << std::endl; }
#else
    #define LOG_VERBOSE_IF(level, condition, msg) ((void)0) // No-op
#endif

// Conditional verbose logging macro with else
#if VERBOSE >= 1
    #define LOG_VERBOSE_IF_ELSE(level, condition, msg_true, msg_false) \
        if (level <= VERBOSE && (condition)) { std::cout << "[log]  " << msg_true << std::endl; } \
        else { std::cout << "[log]  " << msg_false << std::endl; }
#else
    #define LOG_VERBOSE_IF_ELSE(level, condition, msg_true, msg_false) ((void)0) // No-op
#endif

#define MSG_HLINE(box_width) std::cout << std::string(box_width, '-') << std::endl;
#define MSG_BONDLINE(box_width) std::cout << '+' << std::string(box_width - 2, '-') << '+' << std::endl;

#define MSG_BOX_LINE(box_width, msg)                          \
    do {                                                 \
        std::ostringstream oss;                          \
        oss << msg;                                      \
        std::cout << "| " << std::left                   \
                  << std::setw(box_width - 4) << oss.str() \
                  << " |" << std::endl;                  \
    } while (0)

#define MSG_BOX(box_width, msg)                          \
    do {                                                 \
        std::ostringstream oss;                          \
        oss << msg;                                      \
        MSG_BONDLINE(box_width);                        \
        std::cout << "| " << std::left                   \
                  << std::setw(box_width - 4) << oss.str() \
                  << " |" << std::endl;                  \
        MSG_BONDLINE(box_width);                        \
    } while (0)

#define HEADER_PRINT(header, msg) \
    do { \
        std::ostringstream oss; \
        oss << msg; \
        std::cout << '[' << header << "]  " << oss.str() << std::endl; \
    } while (0)

#endif
