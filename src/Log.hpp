#pragma once
#include <iostream>

namespace nn {

inline void Log() {
    std::cout << std::endl;
}

template <typename T, typename... Args>
inline void Log(T x, Args&&... args) {
    std::cout << x << ' ';
    Log(args...);
}

template <typename... Args>
inline void LogInfo(Args&&... args) {
    Log("Info: ", args...);
}

template <typename... Args>
inline void LogWarning(Args&&... args) {
    Log("Warning: ", args...);
}

template <typename... Args>
inline void LogError(Args&&... args) {
    Log("Error: ", args...);
}

} // namespace nn
