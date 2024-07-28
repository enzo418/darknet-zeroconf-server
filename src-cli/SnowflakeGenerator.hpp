#pragma once

#include <time.h>

#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>  // std::setw
#include <iostream>
#include <mutex>
#include <thread>

typedef int_least64_t snowflake;

// is not inserted into the database, it is used to avoid the use of null
// pointers.
constexpr snowflake NullID = -1;

class SnowflakeGenerator {
   public:
    static snowflake generate(uint16_t threadID);

   private:
    static inline snowflake getCurrentTimestampMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
};