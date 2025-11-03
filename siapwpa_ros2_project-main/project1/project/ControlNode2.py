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

def get_spline(frame, visu = True):
    # Znajdowanie linii (żółtej) i dopasowanie splajnu
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([10, 80, 80])   
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    height, width = frame.shape[:2]
  
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    x_pts = []
    y_pts = []
    
    # Zbieranie centroidów
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        if cy < height and cx < width:
            x_pts.append(cx)
            y_pts.append(cy)
            if visu:
                cv2.circle(frame, (int(cx),int(cy)), 5, color=(0,255,0), thickness=2)
                
    if len(y_pts) >= 4:
        x_pts = np.array(x_pts)
        y_pts = np.array(y_pts)
        sort_idx = np.argsort(y_pts)
        sort_idx = sort_idx[-12:]
        x_pts = x_pts[sort_idx]
        y_pts = y_pts[sort_idx]

        spline = UnivariateSpline(y_pts, x_pts, s=0.7)  
    
        if visu:
            y_smooth = np.linspace(y_pts.min(),y_pts.max(), 30)
            x_smooth = spline(y_smooth) 
            spline_points = np.vstack([x_smooth, y_smooth]).T.astype(np.int32)
            cv2.polylines(frame, [spline_points], isClosed=False, color=(0,255,0), thickness=2)
            
        return spline, (int(x_pts[-1]), int(y_pts[-1])), frame
    else: 
        return None, (0, 0), frame

def get_dir(spline, p1, frame=None, dy=50):
    # Obliczanie wektora kierunkowego (błędu kątowego)
    x1, y1 = p1
    try:
        y_min = spline.get_knots().min()
    except Exception:
        return 0., 0., frame
        
    x1 = spline(y1) 
    y2_target = y1 - dy 
    y2 = max(y2_target, y_min)
    if y2 == y1: 
        return 0., 0., frame
    x2 = spline(y2)

    if np.isnan(x1) or np.isnan(x2):
        print("Błąd None w spline.")
        return 0., 0., frame 

    p1_int = (int(x1), int(y1))
    p2_int = (int(x2), int(y2)) 
    if frame is not None:
        cv2.arrowedLine(frame, p1_int, p2_int, (0, 0, 255), 3) 
    return y1 - y2, x1 - x2, frame 

# --- Główna Klasa Węzła ROS 2 ---

class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node")
        
        # Zmienna przechowująca błąd kątowy
        self.x = 0 
        
        # P: 0.05, I: 0.001, D: 0.1, dt: 0.05 (czas cyklu timera), outlim: 1.0 (max. skręt)
        self.PID = regulatorPID(0.07, 0.02, 0.02, 0.05, 1.0) 
        
        # Częstotliwość wysyłania komend (20 Hz)
        self.send_msg_timer = 0.05 

        # Publisher dla prędkości 
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscriber dla Obrazu
        self.bridge = CvBridge()
        self.create_subscription(
            Image,
            "/world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image",
            self.image_callback,
            10
        )
        self.get_logger().info("Camera viewer and controller started.")

        # Timer do cyklicznego wysyłania komend sterujących
        self.timer = self.create_timer(self.send_msg_timer, self.set_speed)

    # --- Ustawianie wartości wyjściowych ---
    def set_speed(self): 
        msg_out = Twist()
        # Obliczanie wyjścia regulatora na podstawie błędu 'self.x'
        out = self.PID.output(self.x) 
        print(f'Output PID [deg]: {out:.4f}')

        msg_out.linear.x = 1.0 # Stała prędkość do przodu
        msg_out.linear.y = 0.0 # Brak ruchu bocznego
        
        msg_out.angular.z = float(out)
        
        self.cmd_vel_publisher.publish(msg_out)
        self.get_logger().info(f"Msg sent: v_x={msg_out.linear.x:.1f}, a_z={msg_out.angular.z:.4f}")


    # --- Funkcja wywoływana przez Subskrypcję Obrazu (PERCEPCJA) ---
    def image_callback(self, msg): 
        try:
            frame_orginal = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge conversion error: {e}")
            return
        frame = frame_orginal.copy()
        frame_bev = perspectiveWarp(frame)

        spline, p0, frame_vis = get_spline(frame)
        
        if spline is not None: 
            
            dy, dx, frame_vis = get_dir(spline, p0, frame_vis)
            self.x = np.arctan2(dx, dy) * 180 / np.pi
            print(f'Err:  [deg]: {self.x:.2f}')
        else:
            print('Control unavailable. Using last known error.')
        
        # Wyświetlanie obrazów
        cv2.imshow("Camera", frame_orginal)
        cv2.imshow("Bird eye", frame_vis)
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