import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node

import numpy as np

from social_controller.controller_config import *

from geometry_msgs.msg import TwistStamped
from skeleton_interfaces.msg import Predictions

class ControllerNode(Node):
    def __init__(self):
        super().__init__(CONTROLLER_NODE_NAME)
        self._publisher = self.create_publisher(
            TwistStamped, CONTROLLER_COMMAND_TOPIC, 5
        )

        if NEED_PREDICTIONS:
            self._subscriber = self.create_subscription(
                Predictions, PREDICTION_TOPIC, self._prediction_callback, 5
            )

    def _prediction_callback(self, msg):
        predictions = np.array(msg.predictions.data)
        logits = np.array(msg.logits.data)
        predictions = np.reshape(predictions, (12, 19, 6, 2))
        print(predictions)

def main(args=None):
    
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()