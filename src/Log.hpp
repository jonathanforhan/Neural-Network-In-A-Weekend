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
    Log("NN Info: ", args...);
}

template <typename... Args>
inline void LogWarning(Args&&... args) {
    Log("NN Warning: ", args...);
}

template <typename... Args>
inline void LogError(Args&&... args) {
    Log("NN Error: ", args...);
}

} // namespace nn
