# DroneVision
MCCC 111B Final Project utilizing OpenCV and Ardupilot

### Project Goal
This project utilizes the [pyardrone](https://github.com/afg984/pyardrone) and openCV libraries to create a simple drone control system that tracks a and follows a single human face, positioning the drone to keep the tracked object in the center of its field of view.

### Files
**drone_face_tracking** Runs an autonomous drone functionality that immediately launches the drone and begins reacting to a recognised face.  The displayed viewport can be closed by pressing 'q' and will land the drone.

**drone_ui** Provides a very simple UI for manual drone control, primarily for testing and debugging purposes.  This file does not use the drone's camera at all.
