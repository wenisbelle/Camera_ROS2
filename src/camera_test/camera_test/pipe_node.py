import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import math

from camera_interface.msg import CameraInterface


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(Image, '/vision/image_annotated', 10)
        self.publisher_pose = self.create_publisher(Pose, '/camera_target_pose', 10)
        
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.count = 0
        self.FINGERS_DOWN = [False, False, False, False]
        

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
        circle_radius=40
        img_width=720

        if results.multi_hand_landmarks:
      
            wrist = results.multi_hand_landmarks[0].landmark[0]
            middle_finger_base = results.multi_hand_landmarks[0].landmark[9]
            hand_point = self.origin_ref(wrist, middle_finger_base)
            perpendicular_vector = self.perpendicular_vector(hand_point, middle_finger_base)
            
            self.draw_point(img, hand_point)
            self.draw_point(img, [middle_finger_base.x,  middle_finger_base.y])
            self.draw_point(img, perpendicular_vector)
            self.draw_middle_point(img)            
            self.draw_vector(img, hand_point, perpendicular_vector)
            self.draw_vector(img, hand_point, [middle_finger_base.x,  middle_finger_base.y])
       
            h, w, _ = img.shape
            origin = [w/2, h/2]
            distance = self.calculate_distance_from_center(hand_point)
            yaw_angle = self.calculate_angle(hand_point, perpendicular_vector)
            
            pose_msg = Pose()
            
            info_fingers = self.fingers_up(results.multi_hand_landmarks[0])
            if info_fingers == self.FINGERS_DOWN:
                pose_msg.position.x = 0.0
                pose_msg.position.y = 0.0
                pose_msg.position.z = 0.0
                pose_msg.orientation.x = 0.0
                pose_msg.orientation.y = 0.0
                pose_msg.orientation.z = 0.0
                pose_msg.orientation.w = 0.0
            else:
                pose_msg.position.x = distance[0]
                pose_msg.position.y = distance[1]
                pose_msg.position.z = 0.0
                if distance[2] <= circle_radius/img_width:
                    pose_msg.orientation.x = 0.0
                    pose_msg.orientation.y = 0.0
                    pose_msg.orientation.z = 0.0
                    pose_msg.orientation.w = 0.0
                else:
                    pose_msg.orientation.x = 0.0
                    pose_msg.orientation.y = 0.0
                    pose_msg.orientation.z = math.cos(yaw_angle/2)
                    pose_msg.orientation.w = math.sin(yaw_angle/2)
            
            self.publisher_pose.publish(pose_msg)
                    

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
        hand_point = [(wrist.x + middle_finger_base.x)/2, (wrist.y + middle_finger_base.y)/2]
        return hand_point

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

    def draw_point(self, img, point):
        h, w, _ = img.shape
        px, py = int(point[0] * w), int(point[1] * h)
        cv2.circle(img, (px, py), 8, (0, 255, 0), cv2.FILLED)

    def draw_vector(self, img, origin_point, end_point):
        h, w, _ = img.shape
        ox, oy = int(origin_point[0] * w), int(origin_point[1] * h)
        px, py = int(end_point[0] * w), int(end_point[1] * h)
        cv2.line(img, (ox, oy), (px, py), (0, 0, 255), 2)

    def draw_middle_point(self, img):
        h, w, _ = img.shape
        px, py = w//2, h//2
        cv2.circle(img, (px, py), 40, (0, 255, 0), cv2.FILLED)

    def calculate_distance_from_center(self, hand_point, circle_radius=40, img_width=720):
        euclidian_distance =  math.sqrt((hand_point[0]-0.5)**2 + (hand_point[1]-0.5)**2)

        distance_x = (hand_point[0] - 0.5)
        distance_y = (hand_point[1] - 0.5)
      
        if euclidian_distance <= circle_radius/img_width:
            distance_x = 0.0
            distance_y = 0.0

        return [distance_x, distance_y, euclidian_distance]

    def calculate_angle(self, point1, point2):
        angle =  math.atan((point2[1] - point1[1]) / (point2[0] - point1[0]))
        return angle

    def fingers_up(self, hand):
        fingers = []
        for finger_upper_point in [8,12, 16, 20]:
            if hand.landmark[finger_upper_point].y <  hand.landmark[finger_upper_point-2].y: # no opencv o eixo y é invertido
                fingers.append(True)
            else:
                fingers.append(False)
        return fingers

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
