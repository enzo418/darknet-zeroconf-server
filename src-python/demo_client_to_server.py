import cv2
import socket
import struct

# Constants
SERVER_IP = '127.0.0.1'
SERVER_PORT = 3042
IMAGE_TYPE = 2  # Assuming 0 for RAW_BGR, adjust as needed

# Header
# "=" with standard packing (no alignment)
# uint8_t version, uint16_t image_number, uint32_t group_number, uint32_t data_length, uint8_t image_type, uint16_t width, uint16_t height, uint32_t padding
PACKET_HEADER_FORMAT = '=B H I I B H H I'
PACKET_HEADER_SIZE = struct.calcsize(PACKET_HEADER_FORMAT)

assert PACKET_HEADER_SIZE == 20, f"PacketHeader size is not 20 bytes {PACKET_HEADER_SIZE}"

# response
DETECTION_BOX_FORMAT = '=H f f f f f H'
DETECTION_BOX_SIZE = struct.calcsize(DETECTION_BOX_FORMAT)
DETECTION_RESULT_HEADER_FORMAT = '=I H H'
DETECTION_RESULT_HEADER_SIZE = struct.calcsize(DETECTION_RESULT_HEADER_FORMAT)

assert DETECTION_RESULT_HEADER_SIZE == 8, f"DetectionResultHeader size is not 8 bytes {DETECTION_RESULT_HEADER_SIZE}"
assert DETECTION_BOX_SIZE == 24, f"DetectionBox size is not 24 bytes {DETECTION_BOX_SIZE}"

def capture_and_send_images():
    # Initialize the camera
    import sys
    src = 1
    if len(sys.argv) > 1:
        try:
            src = int(sys.argv[1])
        except:
            src = sys.argv[1]
            
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Initialize the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_IP, SERVER_PORT))

    image_number = 0
    group_number = 0

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Resize image to 640x480
            frame = cv2.resize(frame, (640, 480))

            # Get image dimensions
            height, width, _ = frame.shape

            # Convert the image to bytes
            image_data = frame.tobytes()
            data_length = len(image_data)

            if IMAGE_TYPE == 2: # encode
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                result, image_data = cv2.imencode('.jpg', frame, encode_param)
                if not result:
                    print("Error: Could not encode image.")
                    break
                data_length = len(image_data)

            # Construct the PacketHeader
            header = struct.pack(PACKET_HEADER_FORMAT, 1, image_number, group_number, data_length, IMAGE_TYPE, width, height, 0)

            # Send the PacketHeader
            print(f"Sending header with length {PACKET_HEADER_SIZE} and data = {header}")
            sock.sendall(header)

            # Send the image data
            print(f"Sending image data with length {data_length}")
            sock.sendall(image_data)

            # Increment image number
            image_number += 1
            
            if True:
                print("Waiting for detection results...")
                # Receive the DetectionResult header
                detection_result_header = sock.recv(DETECTION_RESULT_HEADER_SIZE)
                group_number, image_number, num_boxes = struct.unpack(DETECTION_RESULT_HEADER_FORMAT, detection_result_header)

                print(f"Received detection results for image {image_number} with {num_boxes} boxes")

                if num_boxes != 0:
                    # Receive the DetectionBoxes
                    detection_boxes = []
                    for _ in range(num_boxes):
                        box_data = sock.recv(DETECTION_BOX_SIZE)

                        detection_box = struct.unpack(DETECTION_BOX_FORMAT, box_data)
                        detection_boxes.append(detection_box)

                        print(f"Received box data with length {len(box_data)} and data = {detection_box}")

                    # Process the detection results (e.g., draw boxes on the frame)
                    for box in detection_boxes:
                        class_id, prob, x, y, w, h, _ = box
                        # Draw the detection box on the frame
                        start_point = (int(x), int(y))
                        end_point = (int(x + w), int(y + h))
                        color = (0, 255, 0)  # Green color for the box
                        thickness = 2
                        cv2.rectangle(frame, start_point, end_point, color, thickness)
                        cv2.putText(frame, f'ID: {class_id}, Prob: {prob:.2f}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    print(f"Received detection results for image {image_number} with {num_boxes} boxes")

            # Display the frame (optional)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Wait input
            #input("Press Enter to send the next image")
    finally:
        # Release the camera and close the socket
        cap.release()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_send_images()
