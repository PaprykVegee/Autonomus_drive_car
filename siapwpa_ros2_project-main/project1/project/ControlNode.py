#to create bridge
#ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist

# ros2 run ros_gz_bridge parameter_bridge \
#   /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
#   /world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image@sensor_msgs/msg/Image@gz.msgs.Image

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline

# --- Klasa Regulatora PID ---
class regulatorPID():
    def __init__(self, P, I, D, dt, outlim):
        self.P = P
        self.I = I
        self.D = D
        self.dt = dt
        self.outlim = outlim
        self.x_prev = 0 
        self.x_sum = 0

    def output(self, x):
        
        P = self.P * x
        if np.abs(x) < self.outlim: 
            self.x_sum += self.I * x * self.dt
        D = self.D * (x - self.x_prev) / self.dt
        out = P + self.x_sum + D
        self.x_prev = x
        if out > self.outlim: out = self.outlim
        elif out < -self.outlim: out = -self.outlim
        
        return out
    
# --- Funkcje Przetwarzania Obrazu ---
def perspectiveWarp(frame):
    # Funkcja do transformacji perspektywy (Widok z Góry - BEV)
    height, width = frame.shape[:2]
    y_sc = 0.6 
    x_sc = 0.3399 
    H2 = int(height * y_sc)
    W2_L = int(width * x_sc)
    W2_R = int(width * (1 - x_sc))
    
    src = np.float32([
        [W2_L, H2], 
        [W2_R, H2], 
        [width, height],
        [0, height]
    ])

    dst = np.float32([
        [0, 0],             
        [width, 0],         
        [width, height],    
        [0, height]         
    ])
    img_size = (width, height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(frame, matrix, img_size)
    return birdseye

def get_yellow_centroids(frame, visu=True):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([10, 80, 80])   
    upper_yellow = np.array([35, 255, 255])
    
    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    height, width = frame.shape[:2]
  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    centroids_list = []

    for i in range(1, num_labels):  
        cx, cy = centroids[i]
        if 0 <= cx < width and 0 <= cy < height:
            centroids_list.append((int(cx), int(cy)))
            if visu:
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), 2)
    
    return centroids_list, frame


class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node")
        
        self.x = 0 
        self.vx = 0.0 
        
        self.PID = regulatorPID(0.1, 0, 0.02, 0.05, 1.0) 
        
        self.send_msg_timer = 0.05 

        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image",
            self.image_callback,
            10
        )
        self.get_logger().info("Camera viewer and controller started.")

        self.timer = self.create_timer(self.send_msg_timer, self.set_speed)

    def set_speed(self): 
        msg_out = Twist()
        out = self.PID.output(self.x) 
        print(f'Output PID [deg]: {out:.4f}')

        msg_out.linear.x = self.vx  # dynamiczna prędkość obliczona w image_callback
        msg_out.linear.y = 0.0 
        msg_out.angular.z = float(out)
        
        self.cmd_vel_publisher.publish(msg_out)
        self.get_logger().info(f"Msg sent: v_x={msg_out.linear.x:.2f}, a_z={msg_out.angular.z:.4f}")



    def image_callback(self, msg): 
        try:
            frame_orginal = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge conversion error: {e}")
            return

        frame = frame_orginal.copy()
        frame_bev = perspectiveWarp(frame)

        centroids_list, frame = get_yellow_centroids(frame_bev)

        height, width = frame_bev.shape[:2]

        if len(centroids_list) > 0:
            avg_x = np.mean([c[0] for c in centroids_list])

            center_x = width / 2
            error = 5 * (center_x - avg_x) / center_x  # błąd kierunku

            self.x = error

            # --- Dynamiczna prędkość liniowa ---
            v_max = 5.0
            v_min = 1
            k = 0.7

            v = v_max * (1 - k * abs(error))
            v = np.clip(v, v_min, v_max)
            self.vx = v

            # --- Wizualizacja ---
            cv2.line(frame, (int(center_x), 0), (int(center_x), height), (255, 0, 0), 2)
            cv2.line(frame, (int(avg_x), 0), (int(avg_x), height), (0, 0, 255), 2)
            cv2.putText(frame, f"Error: {error:.3f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Speed: {v:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Gdy nie wykryto linii — zatrzymujemy skręt i zwalniamy
            self.x = 0.0
            self.vx = 0.0

        cv2.imshow("Bird eye", frame)
        cv2.waitKey(1)




def main():
    rclpy.init()
    node = ControllerNode()
    try:
        rclpy.spin(node) 
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()