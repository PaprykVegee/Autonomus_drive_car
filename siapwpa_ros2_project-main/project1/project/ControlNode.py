import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import os 
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

#to create bridge
#ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist

# ros2 run ros_gz_bridge parameter_bridge \
#   /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
#   /world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image@sensor_msgs/msg/Image@gz.msgs.Image

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
        out =   P + self.x_sum + D
        self.x_prev = x
        if out > self.outlim: out = self.outlim
        elif out < -self.outlim: out = -self.outlim
        return out
    
def perspectiveWarp(frame):
    # 640x420
    height, width = frame.shape[:2]
    y_sc = 0.6 # y_sc = 0.6
    x_sc = 0.3399 # x_sc = 0.33
    H2 = int(height*y_sc)
    # W2_L = int(width//2 - 220)
    # W2_R = int(width//2 + 200)
    W2_L = int(width * x_sc)
    W2_R = int(width * (1 - x_sc))
    # --- 1. Punkty źródłowe (SRC) - Trapez na obrazie ---
    src = np.float32([
        [W2_L, H2],  # Lewy-górny punkt trapezu (daleko)
        [W2_R, H2],  # Prawy-górny punkt trapezu (daleko)
        [width, height],  # Prawy-dolny punkt trapezu (blisko)
        [0, height]    # Lewy-dolny punkt trapezu (blisko)
    ])

    # --- 2. Punkty docelowe (DST) - Prostokąt w widoku "z góry" (BEV) ---
    dst = np.float32([
        [0, 0],             # Mapuj lewy-górny (src) na lewy-górny (dst)
        [width, 0],         # Mapuj prawy-górny (src) na prawy-górny (dst)
        [width, height],    # Mapuj prawy-dolny (src) na prawy-dolny (dst)
        [0, height]         # Mapuj lewy-dolny (src) na lewy-dolny (dst)
    ])
    img_size = (width, height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    birdseye = cv2.warpPerspective(frame, matrix, img_size)
    return birdseye

def get_spline(frame, visu = True):
    # Convert to HSV, threshold on yellow object
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 80, 80])   
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    height, width = frame.shape[:2]
  
    # Get centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    x_pts = []
    y_pts = []
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        if cy < height and cx < width:
            x_pts.append(cx)
            y_pts.append(cy)
            cv2.circle(frame, (int(cx),int(cy)), 5, color=(0,255,0), thickness=2)
    if len(y_pts) >= 4:
        # convert to np array
        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        # sort with y falling to keep proper order
        sort_idx = np.argsort(y_pts)
        x_pts = x_pts[sort_idx]
        y_pts = y_pts[sort_idx]
        # create spline
        spline = UnivariateSpline(y_pts, x_pts, s=0.7)  # s=0 dokładnie przez punkty, większe s → gładziej
    
        if visu:
            y_smooth = np.linspace(y_pts.min(),y_pts.max(), 30)
            x_smooth = spline(y_smooth) 
            spline_points = np.vstack([x_smooth, y_smooth]).T.astype(np.int32)
            cv2.polylines(frame, [spline_points], isClosed=False, color=(0,255,0), thickness=2)
        # returns spline (math object), start point, frame with spline drawed
        return spline, (int(x_pts[-1]), int(y_pts[-1])), frame
    else: return None, (0, 0), frame

# def get_dir(spline, p1, frame = None, dy = 50):
#     x1, y1 = p1
#     x1 = spline(y1)
#     if x1 is None: return 0., frame
#     y2 = y1 - dy
#     x2 = spline(y2)
#     if x2 is None: return 0., frame
#     p2 = (int(x2), int(y2))
#     if not frame is None:
#         cv2.arrowedLine(frame, p1, p2, (0,0,255), 3)
#     # dS = (y2 - y1) / (x2 - x1)
#     return y2 - y1, x2 - x1, frame

def get_dir(spline, p1, frame=None, dy=50):
    x1, y1 = p1
    try:
        y_min = spline.get_knots().min()
    except Exception:
        # Jeśli spline jest z jakiegoś powodu puste lub uszkodzone, zwracamy 0
        return 0., 0., frame
    x1 = spline(y1) 
    y2_target = y1 - dy
    y2 = max(y2_target, y_min)
    if y2 == y1: 
        return 0., 0., frame
    x2 = spline(y2)
    if np.isnan(x1) or np.isnan(x2):
        print("Błąd NaN w spline. Zwracam neutralny wektor.")
        return 0., 0., frame 
    # Bezpieczne rzutowanie na int po walidacji
    p1_int = (int(x1), int(y1))
    p2_int = (int(x2), int(y2)) 
    # Rysowanie strzałki kierunku
    if frame is not None:
        cv2.arrowedLine(frame, p1_int, p2_int, (0, 0, 255), 3) 
    return y1 - y2, x1 - x2, frame

class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node")
        # Timer
        self.v = 0.3
        self.x = 0 
        
        self.PID = regulatorPID(0.01, 0.01, 0.01, 0.5, 1)
        self.send_msg_timer = 0.5

        # Publisher na /cmd_vel -> control
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Create subscriber -> Camera 
        self.bridge = CvBridge()

        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image",
            # self.set_speed,
            # 10
            self.image_callback,
            10
        )
        self.get_logger().info("Camera viewer started.")


        # Timer for msg sending
        self.timer = self.create_timer(self.send_msg_timer, self.set_speed)

    def set_speed(self):
        msg_out=Twist()
        out = self.PID.output(self.x)
        print(f'out: {out}')

        # --- speed mapping ---
        msg_out.linear.x = 1.0
        msg_out.linear.y = 0.
        msg_out.angular.z = float(out)
        self.cmd_vel_publisher.publish(msg_out)
        self.get_logger().info(f"Msg sent: x={msg_out.linear.x}, y={msg_out.linear.y}, a_z={msg_out.angular.z}")


    # --- image processing fcn ----

    def image_callback(self, msg):

        # Konwersja obrazu ROS -> OpenCV
        frame_orginal = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #480x640
        frame = frame_orginal.copy()
        frame = perspectiveWarp(frame)
        spline, p0, frame = get_spline(frame)
        if not spline is None: 
            dy, dx, frame = get_dir(spline, p0, frame)
            self.x = np.arctan2(dx, dy) * 180 / np.pi
            print(f'x: {self.x}')
        else:
            print('Control unavailable.')

        
        cv2.imshow("Camera", frame_orginal)
        cv2.imshow("Bird eye", frame)
        cv2.waitKey(1)



def main():
    rclpy.init()
    node = ControllerNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
