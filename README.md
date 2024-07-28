# Darknet TCP Server

This project provides a TCP server for the Darknet object detection framework. The server is designed to handle image processing requests over a TCP connection, allowing for remote object detection.

## Building
Refer to the original [README](README_darknet.md#building) for building instructions.

In case cmake does not find CUDA check that your cuda version supports the HOST compiler (gcc) 
https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version/46380601#46380601
Install the supported version by your CUDA version and tell cmake to use it:
```bash
CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 cmake -DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda-11.7/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-10 .. -DCMAKE_BUILD_TYPE=Release
```

## Communication Protocol

### Request

Clients should send a request with the following header format:

```c++
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
```

Followed by the image data.

`image_type` can be one of the following values:
- `0`: GRB image
- `1`: RGB image
- `2`: Encoded image (Supports any format that OpenCV can read)

### Response

The server will respond with the following header format:

```c++
struct DetectionResultHeader {
    uint32_t group_number;
    uint16_t image_number;
    uint16_t num_boxes;
};
```

Followed by the detection results:

```c++
struct DetectionBoxData {
    uint16_t class_id;
    float prob;
    float x, y, w, h;

    uint16_t padding;
};
```

`num_boxes` is the number of `DetectionBoxData` entries that follow.
`class_id` is the class ID of the detected object. Refer to the `coco.names` file for the class names.


## Example
First download YOLOv4-tiny weights:
```bash
cd darknet
wget --no-clobber https://github.com/hank-ai/darknet/releases/download/v2.0/yolov4-tiny.weights
```

Run the server with the following command:

```bash
# -dont_show flag is used to disable the display of the video feed
build/src-server/darknet_server run cfg/coco.data cfg/yolov4-tiny.cfg yolov4-tiny.weights -dont_show
```

The server will start and listen on port 3000. You can now start the client with the following command:

```bash
# only requires cv2 (opencv-python) to be installed
python3 src-python/demo_client_to_server.py
```

<hr>


Reefer to [`README_darknet.md`](README_darknet.md) for more information on the darknet framework.