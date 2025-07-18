from zaber_motion import Units
from zaber_motion.ascii import Connection

with Connection.open_serial_port("/dev/ttyACM1") as connection:
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    device = device_list[0]

    axis = device.get_axis(3)
    print(axis.get_position(Units.ANGLE_DEGREES))

    # if not axis.is_homed():
    #   axis.home()

    # Move to 10 deg
    # axis.move_absolute(10, Units.ANGLE_DEGREES)
    axis.move_absolute(10, Units.ANGLE_DEGREES)
    print(axis.get_position(Units.ANGLE_DEGREES))

    # Move by an additional 5mm
    # axis.move_relative(5, Units.LENGTH_MILLIMETRES)
    # print(axis.get_position(Units.LENGTH_MILLIMETRES))
