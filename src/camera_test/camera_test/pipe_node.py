import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import math


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(Image, '/vision/image_annotated', 10)
        
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.count = 0
        

    def image_callback(self, msg):
        # Convert ROS2 image to OpenCV image
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CVBridge conversion failed: {e}')
            return
        img = cv2.resize(img, (720, 720))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)
        wrist = []
        middle_finger_base = []

        if results.multi_hand_landmarks:
            coordenadas = []
        
            wrist = results.multi_hand_landmarks[0].landmark[0]
            middle_finger_base = results.multi_hand_landmarks[0].landmark[9]
      
            origin_point = self.origin_ref(wrist, middle_finger_base)
            parpendicular_vector = self.perpendicular_vector(origin_point, middle_finger_base)
            
            h, w, _ = img.shape
                # Convert normalized coordinates (0–1) to pixel coordinates
            ox, oy = int(origin_point[0] * w), int(origin_point[1] * h)
            mx, my = int(middle_finger_base.x * w), int(middle_finger_base.y * h)
            print(h,w)
            # Draw origin point
            cv2.circle(img, (ox, oy), 8, (0, 255, 0), cv2.FILLED)

            # Draw vector (line) from origin to middle finger base
            cv2.line(img, (ox, oy), (mx, my), (0, 0, 255), 2)


            # Convert both to pixel coordinates
            px, py = int(parpendicular_vector[0] * w), int(parpendicular_vector[1] * h)
            cv2.circle(img, (px, py), 8, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (mx, my), 8, (0, 255, 0), cv2.FILLED)
            # Draw perpendicular line
            cv2.line(img, (ox, oy), (px, py), (255, 0, 0), 2)  # Blue line

            #print(origin_point)
            #print(parpendicular_vector)
            #print(middle_finger_base)
        

        # Publish annotated image
        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            annotated_msg.header = msg.header  # Keep timestamp and frame_id consistent
            self.publisher.publish(annotated_msg)
        except Exception as e:
            self.get_logger().error(f'Error publishing annotated image: {e}')

        cv2.imshow("MediaPipe Hands", img)
        cv2.waitKey(1)

    def origin_ref(self, wrist, middle_finger_base):
        origin_point = [(wrist.x + middle_finger_base.x)/2, (wrist.y + middle_finger_base.y)/2]
        return origin_point

    def perpendicular_vector(self, origin, upper_point):
        # Vector from origin to upper_point (middle finger base)
        dx = upper_point.x - origin[0]
        dy = upper_point.y - origin[1]

        # Rotate 90 degrees (counterclockwise)
        perp_dx = -dy
        perp_dy = dx

        # Add rotated vector to origin to get the new point
        perp_x = origin[0] + perp_dx
        perp_y = origin[1] + perp_dy

        return [perp_x, perp_y]

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
