import rclpy
from rclpy.node import Node
import cv2
import mediapipe as mp

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.mp_hands = mp.solutions.hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def timer_callback(self):
        cap = cv2.VideoCapture(0)  
        success, img = cap.read()
        if not success:
            self.get_logger().warning("Failed to capture image.")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, handLms, mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow("MediaPipe Hands", img)
        cv2.waitKey(1)

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