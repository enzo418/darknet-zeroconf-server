#pragma once

#include "image.hpp"
#ifdef __cplusplus
extern "C" {
#endif
void demo(char *cfgfile, char *weightfile, float thresh, float hier_thresh, int cam_index, const char *filename, const char **names, int classes, int avgframes,
    int frame_skip, const char *prefix, const char *out_filename, int mjpeg_port, int dontdraw_bbox, int json_port, int dont_show, int ext_output, int letter_box_in, int time_limit_sec, const char *http_post_host, int benchmark, int benchmark_layers);
#ifdef __cplusplus
}
#endif
