std_msgs/Header header       # timestamp + frame_id (e.g. "camera_color_optical_frame")
string color_label           # "red" | "green" | "blue"
geometry_msgs/Point position # 3D location on camera frame
float32 area                 # Contour area based on pixels — To filter out little noisy detections