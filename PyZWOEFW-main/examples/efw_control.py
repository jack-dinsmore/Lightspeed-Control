from PyZWOEFW import EFW, SingleMiniEFW
import ctypes
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)

wheel = EFW(verbose=False)
wheel.GetPosition(0)
print(wheel.GetPosition(0))
for i in range(7):
    wheel.SetPosition(0, i)
    print(wheel.GetPosition(0))