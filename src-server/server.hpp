#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <thread>

#include "darknet.h"
#include "list.hpp"

/* ------------------------------------------------------ */
/*                        PROFILER                        */
/* ------------------------------------------------------ */

class Run {
   public:
    Run(bool logResults)
        : start_time(std::chrono::high_resolution_clock::now()),
          logResults(logResults) {}

    ~Run() {
        if (!logResults) return;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time -
                                                                  start_time)
                .count();

        std::cout << "\n\nTotal Run Time: " << total_duration << " ms\n";
        for (const auto& entry : profiles) {
            double percentage =
                (entry.second / static_cast<double>(total_duration)) * 100.0;
            std::cout << entry.first << ": " << entry.second << " ms ("
                      << percentage << "%)\n";
        }
    }

    void addProfile(const std::string& name, long long duration) {
        profiles[name] += duration;
    }

   private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::map<std::string, long long> profiles;
    bool logResults;
};

extern Run* current_run;

class Profile {
   public:
    Profile(const std::string& name)
        : name(name),
          run(current_run),
          start_time(std::chrono::high_resolution_clock::now()) {}

    ~Profile() { Stop(); }

    void Stop() {
        if (stopped) return;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_time - start_time)
                            .count();
        if (run)
            run->addProfile(name, duration);
        else
            printf("Profile: %s, %ld ms\n", name.c_str(), duration);

        stopped = true;
    }

   private:
    bool stopped {false};
    std::string name;
    Run* run;
    std::chrono::high_resolution_clock::time_point start_time;
};

/* ------------------------------------------------------ */
/*                   MODEL/NETWORK DATA                   */
/* ------------------------------------------------------ */

struct context_t {
    network net;    // network
    list* options;  // options - from the config file
    layer last_layer;

    int classes_n;  // number of classes

    const char* name_list;
    char** names;

    float prob_threshold {0.25};  // probability threshold for detection
    bool dontdraw_bbox {false};   // don't draw bounding boxes
    bool log_results {false};     // log results
};

inline void recalculate_bbox_coordinates(box& b, int cols, int rows) {
    if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
    if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
    if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
    if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;
    b.w = (b.w < 1) ? b.w : 1;
    b.h = (b.h < 1) ? b.h : 1;
    b.x = (b.x < 1) ? b.x : 1;
    b.y = (b.y < 1) ? b.y : 1;

    float left = (b.x - b.w / 2.) * cols;
    float right = (b.x + b.w / 2.) * cols;
    float top = (b.y - b.h / 2.) * rows;
    float bot = (b.y + b.h / 2.) * rows;

    if (left < 0) left = 0;
    if (right > cols - 1) right = cols - 1;
    if (top < 0) top = 0;
    if (bot > rows - 1) bot = rows - 1;

    b.x = left;
    b.y = top;
    b.w = right - left;
    b.h = bot - top;
}
