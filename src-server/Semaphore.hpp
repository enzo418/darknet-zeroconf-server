#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>

/**
 * @brief Basic c++20 semaphore implementation
 *
 */
class Semaphore {
   public:
    explicit Semaphore(int pCount = 0) : count(pCount) {}

    /**
     * @brief increments the internal counter and unblocks acquirers
     *
     */
    inline void release() {
        std::unique_lock<std::mutex> lock(mutex);
        count++;
        cv.notify_one();
    }

    /**
     * @brief decrements the internal counter or blocks until it can
     *
     */
    inline void acquire() {
        std::unique_lock<std::mutex> lock(mutex);

        cv.wait(lock, [this]() { return count > 0; });

        count--;
    }

    /**
     * @brief blocks the thread until release was called or the timeout in
     * milliseconds was exceeded.
     *
     * @tparam ms ms milliseconds to timeout
     * @return true if we acquired a value
     * @return false if timeout was exceeded and still there is no value
     */
    template <int ms>
    inline bool acquire_timeout() {
        std::unique_lock<std::mutex> lock(mutex);

        if (cv.wait_for(lock, std::chrono::microseconds(ms),
                        [this]() { return count > 0; })) {
            count--;
            return true;
        }

        return false;
    }

   private:
    std::mutex mutex;
    std::condition_variable cv;
    int count;
};