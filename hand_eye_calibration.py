from aina.calibration.extrinsics import ExtrinsicsCalibration

if __name__ == "__main__":

    calibration = ExtrinsicsCalibration(camera_names=["left", "right"])
    calibration.get_images_and_arm_pose(visualize=True)
    input("Press Enter to continue to the left camera calibration...")
    calibration.get_calibration_error_in_2d(camera_name="left", visualize=True)
    input("Press Enter to continue to the right camera calibration...")
    calibration.get_calibration_error_in_2d(camera_name="right", visualize=True)

    # NOTE: These errors should be around less than 10 pixels. If not, you should re-run the calibration.
