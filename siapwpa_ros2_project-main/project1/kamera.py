import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


# ros2 run ros_gz_bridge parameter_bridge \
#   /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
#   /world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image@sensor_msgs/msg/Image@gz.msgs.Image




class CameraViewer(Node):
    def __init__(self):
        super().__init__("camera_viewer")
        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image",
            self.image_callback,
            10
        )
        self.get_logger().info("Camera viewer started.")

    def image_callback(self, msg):
        # Konwersja obrazu ROS -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        cv2.imshow("Camera", frame)
        cv2.imshow("Edges", edges)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = CameraViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
