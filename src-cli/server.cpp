/**
 * This file includes a server that listens for incoming images from clients
 * and processes them using the Darknet classifier. The server uses the
 * µSockets library to handle the network communication.
 *
 *
 */

#include <libusockets.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "assert.h"
#include "blas.hpp"
#include "classifier.hpp"
#include "dark_cuda.hpp"
#include "darknet.h"
#include "darknet_internal.hpp"
#include "data.hpp"
#include "http_stream.hpp"
#include "image.hpp"
#include "image_opencv.hpp"
#include "network.hpp"
#include "option_list.hpp"
#include "parser.hpp"
#include "utils.hpp"
const int SSL = 0;

#ifdef WIN32
#include <time.h>

#include "gettimeofday.h"
#else
#include <sys/time.h>
#endif

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

Run* current_run = nullptr;

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

struct context_t {
    network net;    // network
    list* options;  // options - from the config file
    layer last_layer;

    int classes_n;  // number of classes

    char* name_list;
    char** names;

    float prob_threshold {0.25};  // probability threshold for detection
    bool dontdraw_bbox {false};   // don't draw bounding boxes
};

void load_model(context_t& ctx, char* datacfg, char* cfgfile,
                char* weightfile) {
    ctx.net = parse_network_cfg_custom(cfgfile, 1, 0);
    if (weightfile) {
        load_weights(&ctx.net, weightfile);
    }
    set_batch_network(&ctx.net, 1);
    ctx.options = read_data_cfg(datacfg);

    fuse_conv_batchnorm(ctx.net);
    calculate_binary_weights(ctx.net);

    ctx.classes_n = option_find_int(ctx.options, "classes", 2);

    ctx.name_list = option_find_str(ctx.options, "names", 0);
    ctx.names = get_labels(ctx.name_list);

    ctx.last_layer = ctx.net.layers[ctx.net.n - 1];
}

void recalculate_bbox_coordinates(box& b, int cols, int rows) {
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

struct DetectionSingleResult {
    uint16_t class_id;  // based on coco.names
    float prob;
    float x, y, w, h;
};

/**
 * @brief Use the context network to detect objects in the image.
 *
 * @param ctx Network context
 * @param img Target - Any image - any size - any number of channels
 * @param out_results Output results
 * @param out_num_results Number of results
 */
void detect(context_t& ctx, cv::Mat* img, DetectionSingleResult** out_results,
            int& out_num_results) {
    if (!img) {
        return;
    }

    if (!ctx.dontdraw_bbox) {
        create_window_cv("Source", 0, 512, 512);
        move_window_cv("Source", 512, 0);
        create_window_cv("Classifier", 0, 512, 512);
    }

    int net_h = ctx.net.h;
    int net_w = ctx.net.w;
    int net_c = ctx.net.c;

    Profile p("pre-process image");

    cv::Mat new_img = cv::Mat(net_h, net_w, CV_8UC(net_c));

    cv::resize(*img, new_img, new_img.size(), 0, 0, cv::INTER_LINEAR);
    if (net_c > 1) cv::cvtColor(new_img, new_img, cv::COLOR_RGB2BGR);

    image in_s = mat_to_image(new_img);

    p.Stop();
    //

    clock_t time = clock();

    {
        Profile p("network_predict");
        network_predict(ctx.net, in_s.data);
    }

    if (!ctx.dontdraw_bbox) {
        printf("Predicted in %4.2f milli-seconds.\n",
               sec(clock() - time) * 1000);
    }

    free_image(in_s);

    const float nms = .45;  // 0.4F
    int nboxes = 0;

    Profile p2("boxes");

    // get network boxes
    detection* dets =
        get_network_boxes(&ctx.net, ctx.net.w, ctx.net.h, ctx.prob_threshold,
                          ctx.prob_threshold, 0, 1, &nboxes, 0);  // resized

    if (nms) {
        if (ctx.last_layer.nms_kind == DEFAULT_NMS)
            do_nms_sort(dets, nboxes, ctx.last_layer.classes, nms);
        else
            diounms_sort(dets, nboxes, ctx.last_layer.classes, nms,
                         ctx.last_layer.nms_kind, ctx.last_layer.beta_nms);
    }

    int ext_output = 0;

    if (!ctx.dontdraw_bbox) {
        draw_detections_cv_v3((mat_cv*)&new_img, dets, nboxes,
                              ctx.prob_threshold, ctx.names, ctx.classes_n,
                              ext_output);

        // draw over source image - Don't we do that below to verify
        // draw_detections_cv_v3((mat_cv*)img, dets, nboxes, ctx.prob_threshold,
        //                       ctx.names, ctx.classes_n, ext_output);
    }

    *out_results =
        (DetectionSingleResult*)xcalloc(nboxes, sizeof(DetectionSingleResult));

    for (int i = 0; i < nboxes; ++i) {
        int class_id = -1;
        float prob = 0;
        for (int j = 0; j < ctx.last_layer.classes; ++j) {
            if (dets[i].prob[j] > ctx.prob_threshold &&
                dets[i].prob[j] > prob) {
                prob = dets[i].prob[j];
                class_id = j;
            }
        }
        if (class_id >= 0) {
            out_num_results++;

            // NOTE: All positions and sizes are in relative coordinates (0..1)
            // to the input image (net_w, net_h).
            box b = dets[i].bbox;
            recalculate_bbox_coordinates(b, img->cols, img->rows);

            (*out_results)[i].class_id = class_id;
            (*out_results)[i].prob = prob;
            (*out_results)[i].x = b.x;
            (*out_results)[i].y = b.y;
            (*out_results)[i].w = b.w;
            (*out_results)[i].h = b.h;

            if (!ctx.dontdraw_bbox) {
                // format: [class_id] class_name prob x y w h
                printf("[%d] %s: %.0f%% %2.1f %2.1f %2.1f %2.1f\n", class_id,
                       ctx.names[class_id], prob * 100, b.x, b.y, b.w, b.h);

                cv::rectangle(*img, cv::Rect(b.x, b.y, b.w, b.h),
                              cv::Scalar(0, 255, 0), 2);
                cv::putText(*img, ctx.names[class_id], cv::Point(b.x, b.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    free_detections(dets, nboxes);

    p2.Stop();

    if (!ctx.dontdraw_bbox) {
        show_image_mat((mat_cv*)&new_img, "Classifier");

        int c = wait_key_cv(10);
        if (c == 27 || c == 1048603) return;

        // --
        show_image_mat((mat_cv*)img, "Source");

        int cc = wait_key_cv(10);
        if (cc == 27 || cc == 1048603) return;
        // // --
    }
}

void demo_classifier(context_t& ctx, int cam_index, const char* filename,
                     int max_fps, bool profile) {
    cap_cv* cap;

    printf("Filename: %s - Cam Index: %d\n", filename, cam_index);

    if (filename) {
        cap = get_capture_video_stream(filename);
    } else {
        cap = get_capture_webcam(cam_index);
    }

    if (!cap) {
        darknet_fatal_error(DARKNET_LOC, "failed to connect to webcam (%d, %s)",
                            cam_index, filename);
    }

    float fps = 0;

    mat_cv* in_img = nullptr;

    while (1) {
        Run run(profile);
        current_run = &run;

        {
            Profile p("get_capture_frame_cv");
            in_img = get_capture_frame_cv(cap);
        }

        double time = get_time_point();

        DetectionSingleResult* results = nullptr;
        int num_results;

        detect(ctx, (cv::Mat*)in_img, &results, num_results);

        free_ptrs((void**)results, num_results);

        double frame_time_ms = (get_time_point() - time) / 1000;

        float curr = 1000.f / frame_time_ms;
        if (fps == 0)
            fps = curr;
        else
            fps = .9 * fps + .1 * curr;

        // printf("FPS: %.2f\n", fps);

        //
        if (in_img) release_mat(&in_img);

        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / max_fps));
    }

    free_ptrs((void**)ctx.names, ctx.net.layers[ctx.net.n - 1].classes);
    free_network(ctx.net);
}

namespace {
    // Atomic flag to signal the detection thread to run
    static volatile int run_image_detect_in_thread = 0;

    static volatile int flag_exit;

    struct us_loop_t* loop = nullptr;

    // Define the packet header structure
    enum ImageType { RAW_BGR, RAW_RGB, ENCODED };  // Assumed with 3 channels

#pragma pack(push, 1)  // 16 bytes + 4 padding
    struct PacketHeader {
        uint8_t version;        // only version 1 is supported
        uint16_t image_number;  // Image number in the group
        uint32_t group_number;  // Group number
        uint32_t data_length;   // Length of the image data
        uint8_t image_type;     // New field for image type
        uint16_t width;         // New field for image width
        uint16_t height;        // New field for image height
        uint32_t padding {0};
    };
#pragma pack(pop)

    // socket extension
    struct detect_socket {
        char* backpressure;
        int length;
    };

    // socket context extension
    struct socket_detect_context {
        uint8_t* buffer;
        size_t buffer_size;

        PacketHeader header;
        bool header_received = false;
        size_t bytes_received = 0;
    };

    // Images received
    struct ImageReceived {
        cv::Mat img;
        uint16_t image_number;
        uint32_t group_number;

        us_socket_t* s;
    };

    // Queue to store images received from the client
    static std::queue<ImageReceived> image_queue;
    static std::mutex image_queue_mutex;

#pragma pack(1)
    struct DetectionBoxData {
        uint16_t class_id;  // based on coco.names
        float prob;
        float x, y, w, h;

        uint16_t padding;
    };

    struct DetectionResultHeader {
        uint32_t group_number;
        uint16_t image_number;
        uint16_t num_boxes;
    };
#pragma pack()

    struct ResponsePac {
        struct us_socket_t* res_s;
        DetectionResultHeader res;
        DetectionBoxData* data;
    };

    // Queue to store the detection results
    static std::queue<ResponsePac> detection_result_queue;
    static std::mutex detection_result_queue_mutex;

    static ResponsePac g_res;
    static int volatile g_res_ready = 0;
    static int volatile g_sender_busy = 0;
}  // namespace

/* After an image is processed by the detect function, saves and defers the
 * detection results to be sent to the client in the server loop. */
void on_wakeup(struct us_loop_t* loop) {
    printf("Wakeup\n");

    // Consume detection results and send them to the client
    while (1) {
        ResponsePac result;

        {
            // std::lock_guard<std::mutex> lock(detection_result_queue_mutex);
            // if (detection_result_queue.empty()) break;

            // result = detection_result_queue.front();
            // detection_result_queue.pop();

            if (!custom_atomic_load_int(&g_res_ready)) break;

            result = g_res;
            custom_atomic_store_int(&g_sender_busy, 1);
        }

        // us_socket_t* s = result.res_s;

        // if (!s) continue;

        // struct detect_socket* ds = (struct detect_socket*)us_socket_ext(SSL,
        // s);

        // // Send the detection results to the client
        // size_t length = sizeof(DetectionResultHeader) +
        //                 result.res.num_boxes * sizeof(DetectionBoxData);
        // char* data = (char*)malloc(length);

        // memcpy(data, &result.res, sizeof(DetectionResultHeader));
        // memcpy(data + sizeof(DetectionResultHeader), result.data,
        //        result.res.num_boxes * sizeof(DetectionBoxData));

        // printf("Sending header (%lu) bytes + %d boxes\n",
        //        sizeof(DetectionResultHeader), result.res.num_boxes);

        // int written = us_socket_write(SSL, s, data, length, 0);

        // if (written != length) {
        //     char* new_buffer = (char*)malloc(ds->length + length - written);
        //     memcpy(new_buffer, ds->backpressure, ds->length);
        //     memcpy(new_buffer + ds->length, data + written, length -
        //     written); free(ds->backpressure); ds->backpressure = new_buffer;
        //     ds->length += length - written;
        // }

        // // free buffer
        // free(data);

        // free the boxes
        free(result.data);

        custom_atomic_store_int(&g_res_ready, 0);
        custom_atomic_store_int(&g_sender_busy, 0);
    }

    printf("Wakeup done [Finished sending results]\n");
}

void* image_detect_in_thread(void* ptr) {
    printf("Detect Thread started\n");

    context_t* ctx = (context_t*)ptr;

    while (!custom_atomic_load_int(&flag_exit)) {
        while (!custom_atomic_load_int(&run_image_detect_in_thread)) {
            if (custom_atomic_load_int(&flag_exit)) return 0;
            this_thread_yield();
        }

        ImageReceived img_received;
        {
            std::lock_guard<std::mutex> lock(image_queue_mutex);
            if (image_queue.empty()) {
                custom_atomic_store_int(&run_image_detect_in_thread, 0);
                continue;
            }

            img_received = std::move(image_queue.front());
            image_queue.pop();
        }

        DetectionSingleResult* results = nullptr;
        int num_results = 0;

        detect(*ctx, &img_received.img, &results, num_results);

        ResponsePac res = {
            .res_s = img_received.s,
            .res = {.group_number = img_received.group_number,
                    .image_number = img_received.image_number,
                    .num_boxes = (uint16_t)num_results},
        };

        {
            Profile p("copy results");

            if (num_results > 0) {
                res.data = (DetectionBoxData*)xcalloc(num_results,
                                                      sizeof(DetectionBoxData));

                for (int i = 0; i < num_results; ++i) {
                    res.data[i].class_id = results[i].class_id;
                    res.data[i].prob = results[i].prob;
                    res.data[i].x = results[i].x;
                    res.data[i].y = results[i].y;
                    res.data[i].w = results[i].w;
                    res.data[i].h = results[i].h;
                }
            }
        }

        custom_atomic_store_int(&run_image_detect_in_thread, 0);

        {
            // std::lock_guard<std::mutex> lock(detection_result_queue_mutex);
            // detection_result_queue.push(res);
            g_res = res;

            // custom_atomic_store_int(&g_res_ready, 1);
        }

        us_wakeup_loop(loop);

        free(results);
    }

    printf("Detect Thread stopped\n");

    return 0;
}

void process_image(us_socket_t* s, uint8_t* image_data, uint32_t image_data_len,
                   uint8_t image_type, uint16_t image_number,
                   uint32_t group_number, uint16_t width, uint16_t height) {
    // printf("Decoding received image: %dx%d   n=%d   g=%d\n", width, height,
    //        image_number, group_number);

    cv::Mat img;
    if (image_type == RAW_BGR) {
        img = cv::Mat(height, width, CV_8UC3, (void*)image_data);
    } else if (image_type == RAW_RGB) {
        img = cv::Mat(height, width, CV_8UC3, (void*)image_data);
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    } else if (image_type == ENCODED) {
        img = cv::imdecode(cv::Mat(1, image_data_len, CV_8UC1, image_data),
                           cv::IMREAD_COLOR);
    }

    if (!img.empty()) {
        // printf("Image decoded\n");

        ImageReceived img_received;
        img_received.img = std::move(img);
        img_received.image_number = image_number;
        img_received.group_number = group_number;
        img_received.s = s;

        {
            std::lock_guard<std::mutex> lock(image_queue_mutex);
            image_queue.push(std::move(img_received));
        }

        custom_atomic_store_int(&run_image_detect_in_thread, 1);
    }
}

/* ------------------------------------------------------ */
/*                    SOCKET FUNCTIONS                    */
/* ------------------------------------------------------ */

struct us_socket_t* on_data(us_socket_t* s, char* data, int length) {
    /* Do not accept any data while in shutdown state */
    if (us_socket_is_shut_down(SSL, (us_socket_t*)s)) {
        return s;
    }

    struct socket_detect_context* ds_context =
        (struct socket_detect_context*)us_socket_context_ext(
            SSL, us_socket_context(SSL, s));

    uint8_t* buffer = ds_context->buffer;
    PacketHeader& header = ds_context->header;
    bool& header_received = ds_context->header_received;
    size_t& bytes_received = ds_context->bytes_received;

    // printf("Received %d bytes\n", length);

    while (length > 0) {
        if (!header_received) {
            size_t header_size = sizeof(PacketHeader);
            size_t bytes_to_copy = std::min(header_size - bytes_received,
                                            static_cast<size_t>(length));
            std::memcpy(reinterpret_cast<uint8_t*>(&header) + bytes_received,
                        data, bytes_to_copy);
            bytes_received += bytes_to_copy;
            data += bytes_to_copy;
            length -= bytes_to_copy;

            if (bytes_received == header_size) {
                header_received = true;
                bytes_received = 0;

                if (ds_context->buffer_size < header.data_length) {
                    free(buffer);
                    buffer = (uint8_t*)malloc(header.data_length);
                    ds_context->buffer = buffer;
                    ds_context->buffer_size = header.data_length;
                }

                // printf("Received header: version=%d, image_number=%d, "
                // "group_number=%d, data_length=%d, image_type=%d, "
                // "width=%d, height=%d\n",
                // header.version, header.image_number, header.group_number,
                // header.data_length, header.image_type, header.width,
                // header.height);
            } else {
                // printf("Need %d more bytes for header\n",
                //        (uint32_t)header_size - (uint32_t)bytes_received);
            }
        } else {
            size_t bytes_to_copy = std::min(header.data_length - bytes_received,
                                            static_cast<size_t>(length));
            std::memcpy(buffer + bytes_received, data, bytes_to_copy);
            bytes_received += bytes_to_copy;
            data += bytes_to_copy;
            length -= bytes_to_copy;

            if (bytes_received == header.data_length) {
                // printf("Received complete image\n");

                // Process the complete image
                process_image(s, buffer, header.data_length, header.image_type,
                              header.image_number, header.group_number,
                              header.width, header.height);
                header_received = false;
                bytes_received = 0;
            } else {
                // printf("Need %d more bytes for image\n",
                //        header.data_length - (uint32_t)bytes_received);
            }
        }
    }

    return s;
}

struct us_socket_t* on_detect_socket_writable(struct us_socket_t* s) {
    struct detect_socket* ds = (struct detect_socket*)us_socket_ext(SSL, s);

    printf("Socket writable\n");

    // Continue writing out our backpressure
    int written = us_socket_write(SSL, s, ds->backpressure, ds->length, 0);
    if (written != ds->length) {
        char* new_buffer = (char*)malloc(ds->length - written);
        memcpy(new_buffer, ds->backpressure, ds->length - written);
        free(ds->backpressure);
        ds->backpressure = new_buffer;
        ds->length -= written;
    } else {
        free(ds->backpressure);
        ds->length = 0;
    }

    return s;
}

/* Socket opened handler */
struct us_socket_t* on_socket_open(struct us_socket_t* s, int is_client,
                                   char* ip, int ip_length) {
    struct detect_socket* ds = (struct detect_socket*)us_socket_ext(SSL, s);
    struct socket_detect_context* socket_context =
        (struct socket_detect_context*)us_socket_context_ext(
            SSL, us_socket_context(SSL, s));

    socket_context->buffer = nullptr;
    socket_context->buffer_size = 0;
    socket_context->header_received = false;
    socket_context->bytes_received = 0;

    ds->backpressure = 0;
    ds->length = 0;

    printf("Client connected\n");

    return s;
}

/* Socket closed handler */
struct us_socket_t* on_socket_close(struct us_socket_t* s, int code,
                                    void* reason) {
    struct detect_socket* ds = (struct detect_socket*)us_socket_ext(SSL, s);
    struct socket_detect_context* socket_context =
        (struct socket_detect_context*)us_socket_context_ext(
            SSL, us_socket_context(SSL, s));

    printf("Client disconnected\n");

    if (socket_context->buffer) free(socket_context->buffer);
    free(ds->backpressure);

    return s;
}

/* Socket half-closed handler */
struct us_socket_t* on_socket_end(struct us_socket_t* s) {
    printf("Socket end\n");
    us_socket_shutdown(SSL, s);
    return us_socket_close(SSL, s, 0, NULL);
}

struct us_socket_t* on_socket_timeout(struct us_socket_t* s) {
    printf("Socket timeout\n");

    /* Close idle HTTP sockets */
    return us_socket_close(SSL, s, 0, NULL);
}

/* Loop pre iteration handler */
void on_pre(struct us_loop_t* loop) {}

/* Loop post iteration handler */
void on_post(struct us_loop_t* loop) {}

/* ----------- Entry point to start the server ---------- */
void run_server(context_t& ctx, int max_fps, bool profile) {
    assert(sizeof(PacketHeader) == 20);

    loop = us_create_loop(0, on_wakeup, on_pre, on_post, 0);

    // Initialize uSockets
    struct us_socket_context_t* context = us_create_socket_context(
        SSL, loop, sizeof(struct socket_detect_context), {});

    // Set up the server
    us_socket_context_on_open(SSL, context, on_socket_open);
    us_socket_context_on_data(SSL, context, on_data);
    us_socket_context_on_close(SSL, context, on_socket_close);
    us_socket_context_on_end(SSL, context, on_socket_end);
    us_socket_context_on_timeout(SSL, context, on_socket_timeout);

    us_socket_context_on_writable(SSL, context, on_detect_socket_writable);

    /* Start accepting echo sockets */
    struct us_listen_socket_t* listen_socket =
        us_socket_context_listen(SSL, context, 0, 3000, 0, 0);

    if (listen_socket) {
        printf("Listening on port 3000...\n");

        Run run(profile);
        current_run = &run;

        // run detection thread
        custom_thread_t detect_thread = NULL;
        if (custom_create_thread(&detect_thread, 0, image_detect_in_thread,
                                 &ctx))
            darknet_fatal_error(DARKNET_LOC, "Detect Thread creation failed");

        // start server
        us_loop_run(loop);

        // signal the detection thread to exit
        custom_atomic_store_int(&flag_exit, 1);

        custom_join(detect_thread, 0);

        free_ptrs((void**)ctx.names, ctx.net.layers[ctx.net.n - 1].classes);
        free_network(ctx.net);
    } else {
        printf("Failed to listen!\n");
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s [demo/run] [cfg] [weights]\n", argv[0]);
        return 0;
    }

    srand(2222222);

    char* gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int* gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    } else {
        gpu = Darknet::CfgAndState::get().gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int dont_show = find_arg(argc, argv, "-dont_show");
    int profile = find_arg(argc, argv, "-profile");
    int max_fps = find_int_arg(argc, argv, "-max_fps", 60);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    char* data = argv[2];
    char* cfg = argv[3];
    char* weights = argv[4];
    char* filename = (argc > 5) ? argv[5] : 0;

    context_t ctx;
    load_model(ctx, data, cfg, weights);

    ctx.dontdraw_bbox = dont_show;

    if (strcmp(argv[1], "demo") == 0) {
        demo_classifier(ctx, cam_index, filename, max_fps, (bool)profile);
    } else {
        run_server(ctx, max_fps, (bool)profile);
    }

    if (gpus && gpu_list && ngpus > 1) free(gpus);

    return 0;
}
