#include "SnowflakeGenerator.hpp"

/**
 * 64 bit long snowflake implementation
 *
 */

constexpr int MAX_THREADS = 127;  // 2^7 -1

// constexpr int timestampBits = 43;  // (2^(43-1)-1)/(ms per year) = 139
// years
constexpr int counterBits = 14;  // 2^14 = 16383
constexpr int threadIdBits = 7;  // 2^7 = 127

static std::array<int, MAX_THREADS> counter_thread = {0};

snowflake SnowflakeGenerator::generate(uint16_t threadID) {
    if (threadID > MAX_THREADS) throw std::runtime_error("invalid thread id");

    snowflake snowflake = getCurrentTimestampMs();

    snowflake <<= threadIdBits;
    snowflake <<= counterBits;

    snowflake |= ++counter_thread[threadID] << threadIdBits;
    snowflake |= threadID;

    return snowflake;
}