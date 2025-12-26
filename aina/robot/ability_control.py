# Copyright (c) Meta Platforms, Inc. and affiliates.

import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from aina.utils.constants import ABILITY_MOTOR_LIMITS

np.set_printoptions(precision=2, suppress=True)


class AbilityControl(Node):
    # This is a wrapper class for the ability hand that will listen to
    # zmq ports to listen to the commands and send them to the ability hand
    # Since ability hand is a serial device, this is better
    def __init__(self):
        super().__init__("ability_control")

        self.command_publisher = self.create_publisher(
            Float32MultiArray,
            "ability_hand/commanded_position",
            50,
        )
        self.state_subscriber = self.create_subscription(
            Float32MultiArray,
            "ability_hand/position",
            self.state_callback,
            50,
        )
        self.home()
        self.position = [0, 0, 0, 0, 0, 0]

    def publish_command(self):
        msg = Float32MultiArray()
        msg.data = self.commanded_position
        self.command_publisher.publish(msg)

    def set_position(self, position):
        position = [float(x) for x in position]
        self.commanded_position = position
        self.publish_command()

    def state_callback(self, msg):
        # print(f"State callback: {msg.data}")
        self.position = list(msg.data)

    def get_position(self):
        return self.position

    def home(self):
        print(f"Homing hand")
        self.set_position([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

    def close_hand(self):
        print(f"Closing hand")
        self.set_position([60.0, 60.0, 60.0, 60.0, 50.0, -100.0])

    def open_hand(self):
        print(f"Opening hand")
        self.set_position([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def move_hand(ability_control, ability_pos, pos_range, finger_id, pos_id):
    if pos_id == len(pos_range):
        pos_id = 0
        ability_pos[finger_id] = 0
        finger_id += 1
        if finger_id < len(ability_pos):
            pos_range = np.linspace(
                ABILITY_MOTOR_LIMITS[finger_id][0],
                ABILITY_MOTOR_LIMITS[finger_id][1],
                20,
            )

    if finger_id == len(ability_pos):
        finger_id = 0
        pos_range = np.linspace(
            ABILITY_MOTOR_LIMITS[finger_id][0],
            ABILITY_MOTOR_LIMITS[finger_id][1],
            20,
        )

    ability_pos[finger_id] = pos_range[pos_id]
    ability_control.set_position(ability_pos)

    return ability_pos, pos_range, finger_id, pos_id


def main():

    rclpy.init(args=None)
    ability_control = AbilityControl()
    finger_id = 0
    pos_range = np.linspace(
        ABILITY_MOTOR_LIMITS[finger_id][0], ABILITY_MOTOR_LIMITS[finger_id][1], 20
    )
    pos_id = 0
    ability_pos = [0, 0, 0, 0, 0, 0]
    while True:
        try:
            # Let ROS process incoming messages (including state_callback)
            rclpy.spin_once(ability_control, timeout_sec=0.01)

            ability_pos, pos_range, finger_id, pos_id = move_hand(
                ability_control, ability_pos, pos_range, finger_id, pos_id
            )

            # Get the joint angles
            ability_joints = ability_control.get_position()
            print(f"Ability joints: {ability_joints}")

            pos_id += 1

            time.sleep(0.1)

        except KeyboardInterrupt:
            break

    rclpy.shutdown()


def init_ability_control():

    rclpy.init(args=None)
    ability_control = AbilityControl()
    # rclpy.spin(ability_control)

    ability_control.home()
    time.sleep(1)
    ability_control.close_hand()
    time.sleep(1)
    ability_control.open_hand()
    time.sleep(1)
    ability_control.home()
    rclpy.shutdown()


if __name__ == "__main__":

    main()
