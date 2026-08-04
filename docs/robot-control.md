# Robot control inputs

Every input source produces the same simulator-independent command types:
base twist, named joint position/velocity/effort, gripper fraction, or stop.
Commands are arbitrated per robot and are never numerically mixed.

Priority is emergency stop, keyboard/gamepad, GUI, localhost command, then
autonomous control. Manual commands have a short TTL. Releasing a device or
losing a command connection therefore sends zero velocity instead of leaving
the last command active.

## Keyboard and gamepad

- W/S: forward and reverse
- A/D: turn left and right
- Q/E: lateral motion for holonomic bases
- Space: latched emergency stop
- Enter: clear emergency stop
- Left gamepad stick: translation
- Right gamepad stick: yaw

The first configured gamepad is used. Missing gamepad hardware does not prevent
keyboard, command, or autonomous operation.

## Command line

The Isaac adapter listens only on localhost by default:

~~~bash
uv run radcounter-robot-command measurement_robot twist 0.5 0.0 0.2 --ttl 2
uv run radcounter-robot-command measurement_robot joint position arm_1=0.5 arm_2=-0.2
uv run radcounter-robot-command measurement_robot gripper tool_gripper 1.0
uv run radcounter-robot-command measurement_robot stop
~~~

Messages use one bounded JSON line over TCP. They are validated and queued;
the Isaac main/physics thread alone applies commands to the robot.

## Autonomous control

Register a callback with IsaacRobotInputRouter.register_auto_controller. It is
called every physics update and may return any common command. A live manual
source preempts it automatically. Disabling auto removes its pending command.

Robot-specific policies, ROS 2 controllers, planners, and learned policies can
all be adapted through the same callback or RobotControllerPlugin interfaces.
