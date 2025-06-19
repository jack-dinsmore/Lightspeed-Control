from PyZWOEFW import EFW, SingleMiniEFW
import ctypes
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)

wheel = EFW('/home/lightspeed/Documents/Lightspeed-Control/PyZWOEFW-main/examples/libEFWFilter.so.1.7')
wheel.GetPosition(0)
wheel.SetPosition(0, 4)
print(wheel.GetPosition(0))
for i in range(7):
    wheel.SetPosition(0, i)
    print(wheel.GetPosition(0))