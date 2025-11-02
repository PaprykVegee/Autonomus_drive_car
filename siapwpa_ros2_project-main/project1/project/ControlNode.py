import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import os 
import cv2
import numpy as np
from scipy.interpolate import splprep, splev

#to create bridge
#ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist

# ros2 run ros_gz_bridge parameter_bridge \
#   /cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
#   /world/mecanum_drive/model/vehicle_blue/link/front_camera_link/sensor/front_camera/image@sensor_msgs/msg/Image@gz.msgs.Image

class ControllerNode(Node):

    def __init__(self):
        super().__init__("controller_node")
        # Timer
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

      # --- control algo ---


      # --------------------

      # --- speed mapping ---
      msg_out.linear.x = 0.3
      msg_out.linear.y = 0.
      msg_out.angular.z = 0.
      self.cmd_vel_publisher.publish(msg_out)
      self.get_logger().info(f"Msg sent: x={msg_out.linear.x}, y={msg_out.linear.y}, a_z={msg_out.angular.z}")



    # --- image processing fcn ----

    def perspectiveWarp(self, frame):
        # Obszar z obrazu wejściowego, który ma być rzutowany
        width = 640 
        height = 480
        offset = 100
        src = np.float32([
            [200, 300],  # lewy górny róg pasa
            [440, 300],  # prawy górny róg pasa
            [0, 480],    # lewy dolny róg obrazu
            [width, 480] # prawy dolny róg obrazu
        ])

        # --- 2. Punkty docelowe (prostokąt bird's eye) ---
        dst = np.float32([
            [0, 0],          # lewy górny
            [width, 0],      # prawy górny
            [0, height],     # lewy dolny
            [width, height]  # prawy dolny
        ])


        # Wyciągnięcie rozmiaru obrazu
        img_size = (frame.shape[1], frame.shape[0])

        # Rzutowanie obszaru z src na obszar dst
        matrix = cv2.getPerspectiveTransform(src, dst)
        birdseye = cv2.warpPerspective(frame, matrix, img_size)

        return birdseye

    def image_callback(self, msg):



        # 1) uporzadkowac kod
        # 2) zmienic transformate hougha -> szukanie centroidow 


        # Konwersja obrazu ROS -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') #480x640
        frame = self.perspectiveWarp(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 100, 200)

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([10, 80, 80])   
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)


        # thresh_lvl = 50
        # _, thresh_frame = cv2.threshold(frame,  thresh_lvl, 255, cv2.THRESH_BINARY_INV)
        edges = cv2.Canny(mask, 100, 200)

        lines = cv2.HoughLinesP(edges,
                        rho=1,
                        theta=np.pi/180,
                        threshold=50,
                        minLineLength=50,
                        maxLineGap=20)
        x = []
        y = []
        if not lines is None:                
            for l in lines:                
                x1, y1, x2, y2 = l[0]
                x.append(x1)
                x.append(x2)
                y.append(y1)
                y.append(y2)

                # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        
        # Tworzymy spline parametryczny przechodzący dokładnie przez punkty
        spline_avail = False
        if len(x) >= 4:
            k = 3  # domyślny cubic spline
            spline_avail = True
        elif len(x) == 3:
            k = 2
            spline_avail = True
        elif len(x) == 2:
            k = 1
            spline_avail = True

        if spline_avail: 
            tck, u = splprep([x, y], s=0, k=k)

            # Generujemy 100 punktów na spline, aby linia była gładka
            u_fine = np.linspace(0, 1, 100)
            x_fine, y_fine = splev(u_fine, tck)

            # Rysujemy spline na obrazie
            for i in range(1, len(x_fine)):
                pt1 = (int(x_fine[i-1]), int(y_fine[i-1]))
                pt2 = (int(x_fine[i]), int(y_fine[i]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)  # zielona linia


        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        # edges = cv2.Canny(blur, threshold1=50, threshold2=150)

        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_frame)

        # thresh_frame = np.uint8(labels == 2) * 255
        # jeśli thresh_frame jest 1-kanałowy, konwertujemy na BGR
        # output = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)




        # for i in range(2, num_labels):  # pomijamy tło i pierwszy największy komponent
        #     cx, cy = centroids[i]
        #     cx, cy = int(cx), int(cy)  # zamiana na int
        #     cv2.circle(output, (cx, cy), 5, (255, 0, 0), 3)



        # idx = 2 # secong biggest area

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        # num_labels, labels = cv2.connectedComponents(thresh_frame)

        # thresh_frame = (labels == 1)
        # thresh_frame = cv2.erode(thresh_frame, np.ones((30,30)))
        # thresh_frame = cv2.dilate(thresh_frame, np.ones((5,5)), iterations = 1)

        cv2.imshow("Camera", frame)
        cv2.imshow("Edges", mask)
        cv2.waitKey(1)



def main():
    rclpy.init()
    node = ControllerNode()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
