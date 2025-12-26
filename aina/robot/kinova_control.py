# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections.abc
import os
import sys
import threading
import time

import numpy as np

collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

from aina.utils.constants import KINOVA_IP, KINOVA_PATH, WRIST_TO_EEF
from aina.utils.file_ops import suppress

np.set_printoptions(precision=3, suppress=True)
from scipy.spatial.transform import Rotation as R

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """

    def check(notification, e=e):
        print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
        if (
            notification.action_event == Base_pb2.ACTION_END
            or notification.action_event == Base_pb2.ACTION_ABORT
        ):
            e.set()

    return check


class ConnectionArgs:
    def __init__(self, ip, username="admin", password="admin"):
        self.ip = ip
        self.username = username
        self.password = password


class KinovaControl:
    def __init__(self):
        sys.path.insert(0, KINOVA_PATH)
        import utilities

        # Initialize the connection to the Kinova arm
        args = ConnectionArgs(ip=KINOVA_IP)
        self.device_connection = utilities.DeviceConnection.createTcpConnection(args)
        self.router = (
            self.device_connection.__enter__()
        )  # This is the object that returns when we use "with ... as router"

        # Create required services
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

    def __del__(self):
        if hasattr(self, "device_connection"):
            self.device_connection.__exit__(None, None, None)

    def home(self):
        # Will move the arm to the home position
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move arm to ready position
        print("Homing the arm")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Safe position reached")
        else:
            print("Timeout on action notification wait")
        return finished

    def get_cartesian_pose(self):
        feedback = self.base_cyclic.RefreshFeedback()
        cartesian_pose = np.array(
            [
                feedback.base.tool_pose_x,
                feedback.base.tool_pose_y,
                feedback.base.tool_pose_z,
                feedback.base.tool_pose_theta_x,
                feedback.base.tool_pose_theta_y,
                feedback.base.tool_pose_theta_z,
            ]
        )
        return cartesian_pose

    def get_wrist_to_base(self):
        """
        Returns the transform from the wrist to the base frame in homogeneous matrix.
        This is at the tip of the ability mount
        """

        eef_to_base = self.get_eef_to_base()
        wrist_to_base = eef_to_base @ WRIST_TO_EEF
        return wrist_to_base

    def get_eef_to_base(self):
        """
        Returns the transform from the end effector to the base frame in homogeneous matrix.
        """
        endeffector_pose = self.get_cartesian_pose()
        eef_to_base = np.eye(4)
        eef_to_base[:3, :3] = R.from_euler(
            "xyz", endeffector_pose[3:], degrees=True
        ).as_matrix()
        eef_to_base[:3, 3] = endeffector_pose[:3]
        return eef_to_base

    def get_joint_angles(self):
        """
        Method to get the current angular pose of the arm

        Returns:
        --------
        angles : np.array
            The angular pose of the arm.
        """
        feedback = self.base_cyclic.RefreshFeedback()
        joint_angles = np.array(
            [
                feedback.actuators[0].position,
                feedback.actuators[1].position,
                feedback.actuators[2].position,
                feedback.actuators[3].position,
                feedback.actuators[4].position,
                feedback.actuators[5].position,
                feedback.actuators[6].position,
            ]
        )
        return joint_angles

    def move_eef(self, eef_to_base):
        # Pose is eef in homogenous matrix

        cartesian_pose = np.zeros(6)
        cartesian_rotation = R.from_matrix(eef_to_base[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        cartesian_translation = eef_to_base[:3, 3]
        cartesian_pose[:3] = cartesian_translation
        cartesian_pose[3:] = cartesian_rotation

        self.move_cartesian(cartesian_pose)

        # endeffector_pose = self.get_cartesian_pose()
        # eef_to_base = np.eye(4)
        # eef_to_base[:3, :3] = R.from_euler(
        #     "xyz", endeffector_pose[3:], degrees=True
        # ).as_matrix()
        # eef_to_base[:3, 3] = endeffector_pose[:3]
        # return eef_to_base

    def move_cartesian(self, pose):
        """
        Method to move the arm to a given cartesian pose

        Parameters:
        ----------
        pose : np.array
            The cartesian pose to move to.
            Format: [x, y, z, theta_x, theta_y, theta_z]
            Units: meters, degrees.
        """
        with suppress(stdout=True):
            print("Starting Cartesian action movement ...")
            action = Base_pb2.Action()
            action.name = "Example Cartesian action movement"
            action.application_data = ""

            # feedback = base_cyclic.RefreshFeedback()

            cartesian_pose = action.reach_pose.target_pose
            cartesian_pose.x = pose[0]  # (meters)
            cartesian_pose.y = pose[1]  # (meters)
            cartesian_pose.z = pose[2]  # (meters)
            cartesian_pose.theta_x = pose[3]  # (degrees)
            cartesian_pose.theta_y = pose[4]  # (degrees)
            cartesian_pose.theta_z = pose[5]  # (degrees)

            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                check_for_end_or_abort(e), Base_pb2.NotificationOptions()
            )

            print("Executing action")
            self.base.ExecuteAction(action)

            print("Waiting for movement to finish ...")
            finished = e.wait(TIMEOUT_DURATION)
            self.base.Unsubscribe(notification_handle)

            if finished:
                print("Cartesian movement completed")
            else:
                print("Timeout on action notification wait")
            return finished

    def move_angular(self, angles):
        """
        Method to move the arm to a given angular pose

        Parameters:
        ----------
        angles : np.array
            The angular pose to move to.
            Format: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
            Units: degrees.
        """
        print("Starting angular action movement ...")
        action = Base_pb2.Action()
        action.name = "Example angular action movement"
        action.application_data = ""

        actuator_count = self.base.GetActuatorCount()

        # Place arm straight up
        for joint_id in range(actuator_count.count):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = angles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e), Base_pb2.NotificationOptions()
        )

        print("Executing action")
        self.base.ExecuteAction(action)

        print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Angular movement comPpleted")
        else:
            print("Timeout on action notification wait")
        return finished


if __name__ == "__main__":
    kinova_control = KinovaControl()
    cart_pose = kinova_control.get_eef_to_base()
    print(f"Current eef pose: {cart_pose}")
