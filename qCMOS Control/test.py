import time
from dcam import Dcamapi, Dcam, DCAMERR
# import GPS_time
from camera_params import CAMERA_PARAMS

# Initialize the DCAM API
def initialize_camera():
    if Dcamapi.init() is not False:
        dcam = Dcam(0)  # Assuming first device (index 0)
        if dcam.dev_open() is not False:
            print("Camera opened successfully.")
            
            return dcam
        else:
            print(f"Error opening device: {dcam.lasterr()}")
            Dcamapi.uninit()
    else:
        print(f"Error initializing DCAM API: {Dcamapi.lasterr()}")
    return None

# Capture frames and print timestamps and framestamps
def capture_frames(dcam, num_frames=10):
    print(f"Capturing {num_frames} frames...")
    for i in range(num_frames):
        timeout_milisec = 10000  # Adjust timeout if necessary
        if dcam.wait_capevent_frameready(timeout_milisec) is not False:
            result = dcam.buf_getframe_with_timestamp_and_framestamp(i)
            if result is not False:
                frame, npBuf, timestamp, framestamp = result
                print(f"Frame: {i}, Timestamp: {timestamp.sec + timestamp.microsec / 1e6}, Framestamp: {framestamp}")
            else:
                print(f"Failed to get frame {i}: {dcam.lasterr()}")
        else:
            print(f"Timeout waiting for frame {i}: {dcam.lasterr()}")

# Start capturing frames
def start_capture(dcam):
    if dcam.cap_start() is False:
        print(f"Error starting capture: {dcam.lasterr()}")
    else:
        print("Capture started successfully.")

# Stop capturing frames
def stop_capture(dcam):
    if dcam.cap_stop() is not False:
        print("Capture stopped successfully.")
    else:
        print(f"Error stopping capture: {dcam.lasterr()}")

# Release the buffer
def release_buffer(dcam):
    if dcam.buf_release() is not False:
        print("Buffer released successfully.")
    else:
        print(f"Error releasing buffer: {dcam.lasterr()}")

# Allocate the buffer
def allocate_buffer(dcam, size=100):
    if dcam.buf_alloc(size) is not False:
        print("Buffer allocated successfully.")
    else:
        print(f"Error allocating buffer: {dcam.lasterr()}")

def main():
    # Initialize the camera
    dcam = initialize_camera()
    if dcam is None:
        return
    #dcam.prop_setvalue(CAMERA_PARAMS['READOUT_SPEED'],2.0)
    # Allocate buffer for image capture
    allocate_buffer(dcam, size=100)  # Adjust buffer size as necessary
        
    # Set the time stamp producer property


        
    # Start capturing frames
    start_capture(dcam)
        
    # Capture and print timestamps and framestamps for some frames
    capture_frames(dcam, num_frames=10)
        
    # Stop capturing
    stop_capture(dcam)
    

    # Release the buffer
    release_buffer(dcam)
    
    
    # Allocate a new buffer
    allocate_buffer(dcam, size=100)
        
    # Start capture again after reallocating the buffer
    print("Starting capture again after reallocating the buffer...")
    start_capture(dcam)
        
    # Capture and print timestamps and framestamps for some more frames
    capture_frames(dcam, num_frames=10)
        
    # Stop capturing finally
    stop_capture(dcam)
    
    # Release the buffer
    release_buffer(dcam)

    # Close the camera and uninitialize the API
    dcam.dev_close()
    Dcamapi.uninit()
    print("Camera closed and DCAM API uninitialized.")

if __name__ == "__main__":
    main()

