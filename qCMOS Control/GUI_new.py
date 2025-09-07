import tkinter as tk
from tkinter import StringVar, OptionMenu, Checkbutton, Label, Entry, Button, Scale, Frame, LabelFrame, messagebox
from dcam import Dcamapi, Dcam, DCAMERR
from camera_params import CAMERA_PARAMS, DISPLAY_PARAMS
import GPS_time
import threading
import time
import numpy as np
import cv2
import queue
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import os
import warnings
from PyZWOEFW import EFW
import asyncio
from cyberpower_pdu import CyberPowerPDU, OutletCommand
from labjack import ljm
from zaber_motion import Units
from zaber_motion.ascii import Connection
import ctypes
import logging
import traceback
import functools
import gc
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('camera_debug.log'),
        logging.StreamHandler()
    ]
)

def log_dcam_call(func):
    """Decorator to log all DCAM API calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.debug(f"Calling {func.__name__} with args={args[:2]}...")
        try:
            result = func(*args, **kwargs)
            logging.debug(f"{func.__name__} returned successfully")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} failed: {e}")
            logging.error(traceback.format_exc())
            raise
    return wrapper


class DCamLock:
    """Global lock for DCAM API calls to ensure thread safety"""
    _lock = threading.RLock()  # Use RLock to allow reentrant locking
    
    @classmethod
    def __enter__(cls):
        acquired = cls._lock.acquire(timeout=10)  # 10 second timeout
        if not acquired:
            logging.error("Failed to acquire DCamLock - possible deadlock")
            raise TimeoutError("Failed to acquire DCamLock")
        return cls
    
    @classmethod
    def __exit__(cls, *args):
        cls._lock.release()


class SafeFrameBuffer:
    """Thread-safe frame buffer with bounds checking"""
    def __init__(self, size):
        self.size = size
        self.frames = {}
        self.lock = threading.RLock()
        
    def add_frame(self, index, frame):
        with self.lock:
            safe_index = index % self.size
            # Make a copy to ensure frame lifetime
            if frame is not None:
                self.frames[safe_index] = np.copy(frame)
            else:
                self.frames[safe_index] = None
            logging.debug(f"Added frame at index {safe_index}")
    
    def get_frame(self, index):
        with self.lock:
            safe_index = index % self.size
            if safe_index in self.frames and self.frames[safe_index] is not None:
                return np.copy(self.frames[safe_index])
            return None
    
    def clear(self):
        with self.lock:
            self.frames.clear()
            gc.collect()  # Force garbage collection


class SharedData:
    def __init__(self):
        self.camera_params = {}
        self.lock = threading.RLock()  # Use RLock for reentrant locking


class CameraThread(threading.Thread):
    def __init__(self, shared_data, frame_queue, timestamp_queue, gui_ref):
        super().__init__()
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.gui_ref = gui_ref  # Reference to the GUI
        self.dcam = None
        self.running = True
        self.capturing = False
        self.frame_index = 0  # Start at the 0th frame
        self.modified_params = {}  # Dictionary to keep track of modified parameters
        self.start_time = None  # Variable to store the start time
        self.paused = threading.Event()  # Event to manage pausing
        self.paused.set()  # Start in unpaused state
        self.first_run = True  # Flag to check if it's the first run
        self.first_frame = True  # Flag to capture GPS timestamp only once per capture sequence
        self.buffer_size = 1000  # Increased buffer size
        self.safe_buffer = SafeFrameBuffer(self.buffer_size)
        self.capture_lock = threading.RLock()  # Use RLock for capture operations
        self.save_queue = None  # Initialize save_queue
        
        # Command queue for thread-safe property changes
        self.command_queue = queue.Queue()
        self.command_response_queue = queue.Queue()
        
        # Frame processing lock
        self.frame_processing_lock = threading.Lock()

    def run(self):
        try:
            # Run the connection process in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.connect_camera, daemon=True).start()
        except Exception as e:
            logging.error(f"Fatal error in camera thread: {e}")
            logging.error(traceback.format_exc())

    @log_dcam_call
    def connect_camera(self):
        # Retry loop until camera connects
        while self.running:
            logging.info("Attempting to initialize DCAM API...")
            # Initialize the API
            init_success = True
            try:
                with DCamLock():
                    ret = Dcamapi.init()
                    if ret is False:
                        err = Dcamapi.lasterr()
                        logging.error(f"DCAM API init error: {err}")
                        init_success = False
            except Exception as e:
                logging.error(f"Exception during Dcamapi.init(): {e}")
                init_success = False

            if not init_success:
                # Ensure uninitialized state
                try:
                    with DCamLock():
                        Dcamapi.uninit()
                except Exception as e:
                    pass
                logging.info("Initialization failed, retrying in 5 seconds...")
                if self.gui_ref is not None:
                    self.gui_ref.status_message.config(text="Camera not connected.", fg="red")
                time.sleep(5)
                continue

            # Attempt to open the camera device
            try:
                logging.info("Opening camera device...")
                with DCamLock():
                    cam = Dcam(0)
                    if cam.dev_open() is False:
                        err = cam.lasterr()
                        logging.error(f"Error opening device: {err}")
                        cam = None
                        raise RuntimeError(f"Device open failed: {err}")
                    logging.info("Camera device opened successfully.")
                    self.dcam = cam
            except Exception as e:
                logging.error(f"Exception opening camera: {e}")
                if self.gui_ref is not None:
                    self.gui_ref.status_message.config(text="Camera not connected.", fg="red")
                # Clean up DCAM API before retrying
                try:
                    with DCamLock():
                        Dcamapi.uninit()
                except Exception:
                    pass
                logging.info("Camera not connected. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            # At this point, camera connected
            break

        if not self.running or self.dcam is None:
            return

        # Configure camera
        if self.first_run:
            self.set_defaults()
            self.first_run = False
        else:
            self.restore_modified_params()

        self.update_camera_params()
        logging.info("Camera connected. Entering main loop.")
        
        while self.running:
            try:
                # Process commands from the queue
                self.process_commands()
                
                self.paused.wait()
                if self.capturing:
                    self.capture_frame()
                else:
                    time.sleep(0.01)
            except Exception as e:
                logging.error(f"Error in main camera loop: {e}")
                logging.error(traceback.format_exc())
                time.sleep(0.1)  # Prevent tight error loop

    def process_commands(self):
        """Process commands from the command queue"""
        try:
            while not self.command_queue.empty():
                try:
                    command = self.command_queue.get_nowait()
                    cmd_type = command.get('type')
                    
                    with DCamLock():
                        if cmd_type == 'set_property':
                            prop_name = command['prop_name']
                            value = command['value']
                            self._set_property_internal(prop_name, value)
                            self.command_response_queue.put({'success': True})
                        elif cmd_type == 'update_params':
                            self._update_camera_params_internal()
                            self.command_response_queue.put({'success': True})
                        elif cmd_type == 'start_capture':
                            self._start_capture_internal()
                            self.command_response_queue.put({'success': True})
                        elif cmd_type == 'stop_capture':
                            self._stop_capture_internal()
                            self.command_response_queue.put({'success': True})
                        else:
                            logging.warning(f"Unknown command type: {cmd_type}")
                            self.command_response_queue.put({'success': False, 'error': 'Unknown command'})
                except queue.Empty:
                    break
                except Exception as e:
                    logging.error(f"Error processing command: {e}")
                    self.command_response_queue.put({'success': False, 'error': str(e)})
        except Exception as e:
            logging.error(f"Error processing commands: {e}")

    @log_dcam_call
    def disconnect_camera(self):
        logging.info("Disconnecting camera...")
        with self.capture_lock:
            self.stop_capture()
            with DCamLock():
                if self.dcam is not None:
                    self.dcam.dev_close()
                    self.dcam = None
                Dcamapi.uninit()
        logging.info("Camera disconnected.")

    def reset_camera(self):
        logging.info("Resetting camera...")
        self.pause_thread()  # Pause the thread before resetting
        self.disconnect_camera()
        # Run the camera connection in a separate thread to avoid blocking the GUI
        threading.Thread(target=self.connect_camera, daemon=True).start()
        self.resume_thread()  # Resume the thread after resetting
        logging.info("Camera has been reset.")

    def pause_thread(self):
        logging.info("Pausing camera thread...")
        self.paused.clear()  # Pauses the thread

    def resume_thread(self):
        logging.info("Resuming camera thread...")
        self.paused.set()  # Resumes the thread

    @log_dcam_call
    def set_defaults(self):
        logging.info("Setting default camera parameters...")
        with DCamLock():
            self._set_property_internal('READOUT_SPEED', 1.0)
            self._set_property_internal('EXPOSURE_TIME', 0.1)
            self._set_property_internal('TRIGGER_SOURCE', 1.0)  # 1.0 corresponds to INTERNAL
            self._set_property_internal('TRIGGER_MODE', 6.0)  # 6.0 corresponds to START
            self._set_property_internal('OUTPUT_TRIG_KIND_0', 2.0) # 1.0 corresonds to LOW
            self._set_property_internal('OUTPUT_TRIG_ACTIVE_0', 1.0)
            self._set_property_internal('OUTPUT_TRIG_POLARITY_0', 1.0)
            self._set_property_internal('OUTPUT_TRIG_PERIOD_0', 1.0)
            self._set_property_internal('OUTPUT_TRIG_KIND_1', 1.0) # 1.0 corresonds to LOW
            self._set_property_internal('OUTPUT_TRIG_ACTIVE_1', 1.0)
            self._set_property_internal('OUTPUT_TRIG_POLARITY_1', 2.0)
            self._set_property_internal('OUTPUT_TRIG_PERIOD_1', 1.0)
            self._set_property_internal('SENSOR_MODE', 18.0)
            self._set_property_internal('IMAGE_PIXEL_TYPE', 1.0)

    def _update_camera_params_internal(self):
        """Internal method to update camera params - must be called with DCamLock held"""
        logging.info("Updating camera parameters...")
        if self.dcam is None:
            logging.error("Camera not initialized")
            return
            
        with self.shared_data.lock:
            self.shared_data.camera_params.clear()
            idprop = self.dcam.prop_getnextid(0)
            while idprop is not False:
                propname = self.dcam.prop_getname(idprop)
                if propname is not False:
                    propvalue = self.dcam.prop_getvalue(idprop)
                    if propvalue is not False:
                        valuetext = self.dcam.prop_getvaluetext(idprop, propvalue)
                        if valuetext is not False:
                            self.shared_data.camera_params[propname] = valuetext
                        else:
                            self.shared_data.camera_params[propname] = propvalue
                idprop = self.dcam.prop_getnextid(idprop)

    @log_dcam_call
    def update_camera_params(self):
        """Public method to update camera params - uses command queue"""
        if threading.current_thread() == self:
            # If called from camera thread, execute directly
            with DCamLock():
                self._update_camera_params_internal()
        else:
            # If called from another thread, use command queue
            self.command_queue.put({'type': 'update_params'})
            try:
                response = self.command_response_queue.get(timeout=5)
                if not response.get('success'):
                    logging.error(f"Failed to update params: {response.get('error')}")
            except queue.Empty:
                logging.error("Timeout waiting for update_params response")

    def stop(self):
        logging.info("Stopping camera thread...")
        self.running = False
        self.stop_capture()
        self.paused.set()  # Ensure thread isn't paused
        with DCamLock():
            if self.dcam is not None:
                self.dcam.dev_close()
                self.dcam = None
            Dcamapi.uninit()

    def _set_property_internal(self, prop_name, value):
        """Internal method to set property - must be called with DCamLock held"""
        if prop_name in CAMERA_PARAMS:
            logging.info(f"Setting property: {prop_name} = {value}")
            if self.dcam is not None:
                set_success = self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
                if set_success is False:
                    raise Exception(f"Failed to set property {prop_name}: {self.dcam.lasterr()}")
                self._update_camera_params_internal()
                # Track the modified parameter in the dictionary
                self.modified_params[prop_name] = value

    @log_dcam_call
    def set_property(self, prop_name, value):
        """Public method to set property - uses command queue for thread safety"""
        if threading.current_thread() == self:
            # If called from camera thread, execute directly
            with DCamLock():
                self._set_property_internal(prop_name, value)
        else:
            # If called from another thread, use command queue
            self.command_queue.put({
                'type': 'set_property',
                'prop_name': prop_name,
                'value': value
            })
            try:
                response = self.command_response_queue.get(timeout=5)
                if not response.get('success'):
                    logging.error(f"Failed to set property: {response.get('error')}")
            except queue.Empty:
                logging.error("Timeout waiting for set_property response")

    @log_dcam_call
    def restore_modified_params(self):
        logging.info("Restoring modified camera parameters...")
        # Restore all modified parameters to their original values
        with DCamLock():
            for prop_name, value in self.modified_params.items():
                if self.dcam is not None:
                    self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
        logging.info("Restored all modified camera parameters.")

    def _start_capture_internal(self):
        """Internal method to start capture - must be called from camera thread"""
        logging.info("Starting capture (internal)...")
        with self.capture_lock:
            # Clear the GPS timestamp buffer before starting a new capture
            try:
                GPS_time.clear_buffer()
            except Exception as e:
                logging.error(f"Error clearing GPS buffer: {e}")

            # Ensure the camera is not capturing before starting a new capture session
            if self.capturing:
                logging.warning("Capture is already running. Stopping previous capture.")
                self._stop_capture_internal()

            if self.dcam is None:
                logging.error("Camera not initialized")
                return
            
            # Clear the safe buffer
            self.safe_buffer.clear()

            # Allocate buffer for image capture before triggering
            if self.dcam.buf_alloc(self.buffer_size) is not False:  # Adjust buffer size as necessary
                self.capturing = True
                self.frame_index = 0  # Reset the frame index when starting capture
                self.first_frame = True  # Initialize the first_frame flag

                # Set the time stamp producer property
                self.dcam.prop_setgetvalue(CAMERA_PARAMS['TIME_STAMP_PRODUCER'], 1)

                # Start capturing
                if self.dcam.cap_start() is False:
                    logging.error(f"Error starting capture: {self.dcam.lasterr()}")
                    self.capturing = False
                    self.dcam.buf_release()
            else:
                logging.error("Buffer allocation failed. Capture not started.")

    @log_dcam_call
    def start_capture(self):
        """Public method to start capture - uses command queue for thread safety"""
        logging.info("Starting capture...")
        if threading.current_thread() == self:
            # If called from camera thread, execute directly
            with DCamLock():
                self._start_capture_internal()
        else:
            # If called from another thread, use command queue
            self.command_queue.put({'type': 'start_capture'})
            try:
                response = self.command_response_queue.get(timeout=5)
                if not response.get('success'):
                    logging.error(f"Failed to start capture: {response.get('error')}")
            except queue.Empty:
                logging.error("Timeout waiting for start_capture response")

    def _stop_capture_internal(self):
        """Internal method to stop capture - must be called from camera thread"""
        logging.info("Stopping capture (internal)...")
        self.capturing = False
        if self.dcam is not None:
            if self.dcam.cap_stop() is not False:
                self.dcam.buf_release()
            logging.info("Capture stopped and buffer released.")
        # Restore modified parameters after stopping capture
        self.restore_modified_params()

    @log_dcam_call
    def stop_capture(self):
        """Public method to stop capture - uses command queue for thread safety"""
        logging.info("Stopping capture...")
        if threading.current_thread() == self:
            # If called from camera thread, execute directly
            with self.capture_lock:
                with DCamLock():
                    self._stop_capture_internal()
        else:
            # If called from another thread, use command queue
            self.command_queue.put({'type': 'stop_capture'})
            try:
                response = self.command_response_queue.get(timeout=5)
                if not response.get('success'):
                    logging.error(f"Failed to stop capture: {response.get('error')}")
            except queue.Empty:
                logging.error("Timeout waiting for stop_capture response")

    def capture_frame(self):
        timeout_milisec = 1000  # Increased timeout
        try:
            with self.frame_processing_lock:
                with DCamLock():
                    if self.dcam is None:
                        logging.error("Camera not initialized in capture_frame")
                        return
                    
                    if not self.capturing:
                        return
                        
                    if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                        # Capture the frame at the current index
                        frame_index_safe = self.frame_index % self.buffer_size
                        result = self.dcam.buf_getframe_with_timestamp_and_framestamp(frame_index_safe)
                        
                        if result is not False:
                            frame, npBuf, timestamp, framestamp = result
                            logging.debug(f"Frame: {self.frame_index}, Timestamp: {timestamp.sec+timestamp.microsec/1e6}, Framestamp: {framestamp}")

                            # Always copy the frame data to ensure safety
                            frame_copy = np.copy(npBuf)
                            
                            # Store in safe buffer
                            self.safe_buffer.add_frame(self.frame_index, frame_copy)
                            
                            # Insert frame and timestamp into queues with proper thread safety
                            try:
                                # For display queue, drop old frames if full
                                if self.frame_queue.full():
                                    try:
                                        self.frame_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                self.frame_queue.put_nowait(frame_copy)
                            except queue.Full:
                                logging.debug("Frame queue full, dropping frame")

                            # Timestamp queue
                            try:
                                self.timestamp_queue.put_nowait((timestamp.sec + timestamp.microsec / 1e6, framestamp))
                            except queue.Full:
                                logging.debug("Timestamp queue full")
                            
                            # Save queue if saving is enabled
                            if self.save_queue is not None:
                                try:
                                    self.save_queue.put_nowait((frame_copy, timestamp.sec + timestamp.microsec / 1e6, framestamp))
                                except queue.Full:
                                    logging.warning("Save queue full, frame may be lost")

                            # Fetch the GPS timestamp only once at the start of the capture sequence
                            if self.first_frame:
                                try:
                                    self.start_time = GPS_time.get_first_timestamp()
                                except Exception as e:
                                    logging.error(f"Error getting GPS timestamp: {e}")
                                    self.start_time = None
                                self.first_frame = False  # Prevent further updates

                            # Increment the frame index
                            self.frame_index += 1
                        else:
                            dcamerr = self.dcam.lasterr()
                            if dcamerr.is_timeout():
                                logging.debug('Timeout waiting for frame')
                            else:
                                logging.error(f'Dcam.wait_event() fails with error {dcamerr}')
                    else:
                        time.sleep(0.001)  # Very short sleep if wait fails
        except Exception as e:
            logging.error(f"Error in capture_frame: {e}")
            logging.error(traceback.format_exc())
            time.sleep(0.1)  # Prevent tight error loop


class SaveThread(threading.Thread):
    def __init__(self, save_queue, timestamp_queue, camera_thread, object_name, shared_data):
        super().__init__()
        self.save_queue = save_queue
        self.timestamp_queue = timestamp_queue
        self.running = True
        self.camera_thread = camera_thread  # Reference to the camera thread to access properties
        self.object_name = object_name
        self.batch_size = 100  # Number of frames per cube; adjust based on available memory
        self.frame_buffer = []  # Buffer to accumulate frames
        self.timestamp_buffer = []  # Buffer to accumulate timestamps
        self.framestamp_buffer = []  # Buffer to accumulate framestamps
        self.cube_index = 0  # Index to keep track of cube number
        self.shared_data = shared_data

    def run(self):
        try:
            logging.info(f"SaveThread running: {self.running}")
            if not self.running:
                return

            # Get start_time and format it
            start_time = self.camera_thread.start_time

            # For the filename, remove characters not suitable for filenames
            current_time = time.localtime()
            start_time_filename_str = time.strftime('%Y%m%d_%H%M%S', current_time)

            os.makedirs("captures", exist_ok=True)

            # Start processing frames
            while self.running or not self.save_queue.empty():
                try:
                    frame, timestamp, framestamp = self.save_queue.get(timeout=1)
                    logging.debug(f"SaveThread: Got frame, timestamp {timestamp}, framestamp {framestamp}")
                    self.frame_buffer.append(frame)
                    self.timestamp_buffer.append(timestamp)
                    self.framestamp_buffer.append(framestamp)

                    # If we have enough frames, write them to disk
                    if len(self.frame_buffer) >= self.batch_size:
                        self.write_cube_to_disk(start_time_filename_str)
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error in SaveThread main loop: {e}")
                    logging.error(traceback.format_exc())

            # After loop ends, write any remaining frames
            if self.frame_buffer:
                self.write_cube_to_disk(start_time_filename_str)
                
        except Exception as e:
            logging.error(f"Fatal error in SaveThread: {e}")
            logging.error(traceback.format_exc())

    @log_dcam_call
    def write_cube_to_disk(self, start_time_filename_str):
        try:
            # Increment cube index
            self.cube_index += 1

            # Generate filename with cube index
            filename = f"{self.object_name}_{start_time_filename_str}_cube{self.cube_index:03d}.fits"
            filepath = os.path.join("captures", filename)
            logging.info(f"SaveThread: Writing cube to {filepath}")

            # Create Primary HDU
            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['OBJECT'] = (self.object_name, 'Object name')
            primary_hdu.header['CUBEIDX'] = (self.cube_index, 'Cube index number')

            # Stack frames into a 3D numpy array
            data_cube = np.stack(self.frame_buffer, axis=0)

            # Create ImageHDU for the data cube
            image_hdu = fits.ImageHDU(data=data_cube)
            image_hdu.header['EXTNAME'] = 'DATA_CUBE'
            
            with self.shared_data.lock:
                for key, value in self.shared_data.camera_params.items():
                    with warnings.catch_warnings():
                        # Ignore warnings for keys being too long
                        warnings.filterwarnings("ignore")
                        try:
                            image_hdu.header[key] = value
                        except Exception as e:
                            logging.warning(f"Could not add header {key}: {e}")

            # Create Binary Table HDU for timestamps
            col1 = fits.Column(name='TIMESTAMP', format='D', array=self.timestamp_buffer)
            col2 = fits.Column(name='FRAMESTAMP', format='K', array=self.framestamp_buffer)
            cols = fits.ColDefs([col1, col2])
            timestamp_hdu = fits.BinTableHDU.from_columns(cols)
            timestamp_hdu.header['EXTNAME'] = 'TIMESTAMPS'

            # Create HDUList and write to FITS file
            hdulist = fits.HDUList([primary_hdu, image_hdu, timestamp_hdu])
            hdulist.writeto(filepath, overwrite=True)
            hdulist.close()

            logging.info(f"SaveThread: Saved cube to {filepath}")

            # Clear the buffers
            self.frame_buffer.clear()
            self.timestamp_buffer.clear()
            self.framestamp_buffer.clear()
            
        except Exception as e:
            logging.error(f"Error writing cube to disk: {e}")
            logging.error(traceback.format_exc())

    def stop(self):
        self.running = False
        # Do not call self.join() here; let the calling code handle it


class PeripheralsThread(threading.Thread):
    def __init__(self, shared_data, frame_queue, timestamp_queue, pdu_ip,
                 xmcc1_port, xmcc2_port, gui_ref):
        super().__init__()
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.gui_ref = gui_ref  # Reference to the GUI
        self.efw = None # ZWO 7-position filter wheel
        self.pdu_ip = pdu_ip  # IP address for the CyberPower PDU
        self.pdu = None
        self.ljm_handle = None
        self.xmcc1_port = xmcc1_port
        self.xmcc2_port = xmcc2_port
        self.ax_a_1 = None # Zaber X-MCC1, 1st axis (slit)
        self.ax_a_2 = None # Zaber X-MCC1, 2nd axis (zoom stepper)
        self.ax_b_1 = None # Zaber X-MCC2, 1st axis (focus stepper)
        self.ax_b_2 = None # Zaber X-MCC2, 2nd axis (polarization)
        self.ax_b_3 = None # Zaber X-MCC2, 3rd axis (Halpha/QWP stage)
        self.peripherals_lock = threading.RLock()  # Lock for peripheral operations

    def run(self):
        try:
            # Run the connection process in a separate thread to avoid blocking the GUI
            threading.Thread(target=self.thread_target, daemon=True).start()
        except Exception as e:
            logging.error(f"Error in PeripheralsThread run: {e}")

    def thread_target(self):
        asyncio.run(self.connect_peripherals())

    async def connect_peripherals(self):
        with self.peripherals_lock:
            self.connect_efw()
            self.connect_zaber_axes()
            logging.info("Connecting to CyberPower PDU...")
            try:
                self.pdu = CyberPowerPDU(ip_address=self.pdu_ip, simulate=False)
                await self.pdu.initialize()
            except Exception as e:
                logging.error(f"Failed to connect to CyberPower PDU: {e}")
            self.connect_labjack()

    def connect_labjack(self):
        try:
            with self.peripherals_lock:
                self.ljm_handle = ljm.openS("T4", "ANY", "ANY")
                logging.info("LabJack connected successfully.")
        except Exception as e:
            logging.error(f"Failed to connect to LabJack: {e}")

    def connect_efw(self):
        logging.info("Connecting to ZWO filter wheel...")
        try:
            with self.peripherals_lock:
                self.efw = EFW(verbose=False)
                # Need to read the position before setting position will work
                self.efw.GetPosition(0)
                logging.info("ZWO filter wheel connected successfully.")
        except Exception as e:
            logging.error(f"Failed to connect to ZWO filter wheel: {e}")
            self.efw = None

    # def connect_zaber_axes(self):
    #     logging.info("Connecting to linear stages and stepper motors...")
    #     try:
    #         with self.peripherals_lock:
    #             self.connection_a = Connection.open_serial_port(self.xmcc1_port)
    #             self.connection_a.enable_alerts()
    #             self.xmcc_a = self.connection_a.detect_devices()[0]
    #     except Exception as e:
    #         self.connection_a = None
    #         self.xmcc_a = None
    #         self.ax_a_1 = None
    #         self.ax_a_2 = None
    #         logging.error(f"Failed to connect to Zaber X-MCC1: {e}")
    #         logging.info("No control available for slit stage or zoom stepper.")
        
    #     try:
    #         with self.peripherals_lock:
    #             self.connection_b = Connection.open_serial_port(self.xmcc2_port)
    #             self.connection_b.enable_alerts()
    #             self.xmcc_b = self.connection_b.detect_devices()[0]
    #     except Exception as e:
    #         self.connection_b = None
    #         self.xmcc_b = None
    #         self.ax_b_1 = None
    #         self.ax_b_2 = None
    #         self.ax_b_3 = None
    #         logging.error(f"Failed to connect to Zaber X-MCC2: {e}")
    #         logging.info("No control available for Halpha/QWP stage, polarization stage, or focus stepper.")
        
    #     with self.peripherals_lock:
    #         if self.xmcc_a is not None:
    #             self.ax_a_1 = self.xmcc_a.get_axis(1)  # Slit stage
    #             self.ax_a_2 = self.xmcc_a.get_axis(2)  # Zoom stepper
    #             try:
    #                 self.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
    #                 logging.info("Slit stage connected successfully.")
    #             except Exception as e:
    #                 logging.error(f"Failed to get position for slit stage: {e}")
    #                 self.ax_a_1 = None
    #             try:
    #                 self.ax_a_2.get_position(Units.ANGLE_DEGREES)
    #                 logging.info("Zoom stepper connected successfully.")
    #             except Exception as e:
    #                 logging.error(f"Failed to get position for zoom stepper: {e}")
    #                 self.ax_a_2 = None
            
    #         if self.xmcc_b is not None:
    #             self.ax_b_1 = self.xmcc_b.get_axis(1)  # Focus stepper
    #             self.ax_b_2 = self.xmcc_b.get_axis(2)  # Polarization stage
    #             self.ax_b_3 = self.xmcc_b.get_axis(3)  # Halpha/QWP stage
    #             try:
    #                 self.ax_b_1.get_position(Units.ANGLE_DEGREES)
    #                 logging.info("Focus stepper connected successfully.")
    #             except Exception as e:
    #                 logging.error(f"Failed to get position for focus stepper: {e}")
    #                 self.ax_b_1 = None
    #             try:
    #                 self.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
    #                 logging.info("Polarization stage connected successfully.")
    #             except Exception as e:
    #                 logging.error(f"Failed to get position for polarization stage: {e}")
    #                 self.ax_b_2 = None
    #             try:
    #                 self.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
    #                 logging.info("Halpha/QWP stage connected successfully.")
    #             except Exception as e:
    #                 logging.error(f"Failed to get position for Halpha/QWP stage: {e}")
    #                 self.ax_b_3 = None

    def connect_zaber_axes(self):
        logging.info("Connecting to linear stages and stepper motors...")
        
        # Define expected serial numbers for each device
        XMCC1_SERIAL = 137816  # Serial number for X-MCC1
        XMCC2_SERIAL = 137819  # Serial number for X-MCC2
        
        # Try both ports and match by serial number
        ports_to_try = [self.xmcc1_port, self.xmcc2_port]
        devices_found = {}
        
        for port in ports_to_try:
            try:
                with self.peripherals_lock:
                    # Try to open connection
                    connection = Connection.open_serial_port(port)
                    connection.enable_alerts()
                    
                    # Detect devices on this port
                    devices = connection.detect_devices()
                    
                    if devices:
                        device = devices[0]  # X-MCC should be the only device on the port
                        
                        # Get the serial number
                        serial_number = device.serial_number
                        logging.info(f"Found device with serial number {serial_number} on {port}")
                        
                        # Store the device based on its serial number
                        if serial_number == XMCC1_SERIAL:
                            devices_found['xmcc1'] = {
                                'connection': connection,
                                'device': device,
                                'port': port
                            }
                            logging.info(f"X-MCC1 (SN: {XMCC1_SERIAL}) connected on {port}")
                        elif serial_number == XMCC2_SERIAL:
                            devices_found['xmcc2'] = {
                                'connection': connection,
                                'device': device,
                                'port': port
                            }
                            logging.info(f"X-MCC2 (SN: {XMCC2_SERIAL}) connected on {port}")
                        else:
                            logging.warning(f"Unknown device with serial number {serial_number} on {port}")
                            connection.close()
                    else:
                        logging.warning(f"No devices found on {port}")
                        connection.close()
                        
            except Exception as e:
                logging.error(f"Error connecting to {port}: {e}")
        
        # Now assign the connections and devices based on serial numbers
        with self.peripherals_lock:
            # Handle X-MCC1 (serial 137816)
            if 'xmcc1' in devices_found:
                self.connection_a = devices_found['xmcc1']['connection']
                self.xmcc_a = devices_found['xmcc1']['device']
                
                # Initialize axes for X-MCC1
                try:
                    self.ax_a_1 = self.xmcc_a.get_axis(1)  # Slit stage
                    self.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
                    logging.info("Slit stage connected successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize slit stage: {e}")
                    self.ax_a_1 = None
                
                try:
                    self.ax_a_2 = self.xmcc_a.get_axis(2)  # Zoom stepper
                    self.ax_a_2.get_position(Units.ANGLE_DEGREES)
                    logging.info("Zoom stepper connected successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize zoom stepper: {e}")
                    self.ax_a_2 = None
            else:
                # X-MCC1 not found
                self.connection_a = None
                self.xmcc_a = None
                self.ax_a_1 = None
                self.ax_a_2 = None
                logging.error(f"Failed to connect to X-MCC1 (SN: {XMCC1_SERIAL})")
                logging.info("No control available for slit stage or zoom stepper.")
            
            # Handle X-MCC2 (serial 137819)
            if 'xmcc2' in devices_found:
                self.connection_b = devices_found['xmcc2']['connection']
                self.xmcc_b = devices_found['xmcc2']['device']
                
                # Initialize axes for X-MCC2
                try:
                    self.ax_b_1 = self.xmcc_b.get_axis(1)  # Focus stepper
                    self.ax_b_1.get_position(Units.ANGLE_DEGREES)
                    logging.info("Focus stepper connected successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize focus stepper: {e}")
                    self.ax_b_1 = None
                
                try:
                    self.ax_b_2 = self.xmcc_b.get_axis(2)  # Polarization stage
                    self.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
                    logging.info("Polarization stage connected successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize polarization stage: {e}")
                    self.ax_b_2 = None
                
                try:
                    self.ax_b_3 = self.xmcc_b.get_axis(3)  # Halpha/QWP stage
                    self.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
                    logging.info("Halpha/QWP stage connected successfully.")
                except Exception as e:
                    logging.error(f"Failed to initialize Halpha/QWP stage: {e}")
                    self.ax_b_3 = None
            else:
                # X-MCC2 not found
                self.connection_b = None
                self.xmcc_b = None
                self.ax_b_1 = None
                self.ax_b_2 = None
                self.ax_b_3 = None
                logging.error(f"Failed to connect to X-MCC2 (SN: {XMCC2_SERIAL})")
                logging.info("No control available for Halpha/QWP stage, polarization stage, or focus stepper.")
        
        # Log summary of connections
        if 'xmcc1' in devices_found and 'xmcc2' in devices_found:
            logging.info("All Zaber devices connected successfully!")
        elif devices_found:
            logging.warning("Some Zaber devices connected, but not all found.")
        else:
            logging.error("No Zaber devices could be connected.")

    def disconnect_peripherals(self):
        logging.info("Disconnecting peripherals...")
        with self.peripherals_lock:
            if self.efw is not None:
                try:
                    self.efw.Close(0)
                    logging.info("ZWO filter wheel disconnected successfully.")
                except Exception as e:
                    logging.error(f"Failed to disconnect ZWO filter wheel: {e}")
            else:
                logging.info("ZWO filter wheel was not connected.")

            if self.pdu is not None:
                try:
                    asyncio.run(self.pdu.close())
                    logging.info("CyberPower PDU disconnected successfully.")
                except Exception as e:
                    logging.error(f"Failed to disconnect CyberPower PDU: {e}")
            else:
                logging.info("CyberPower PDU was not connected.")

            if self.ljm_handle is not None:
                ljm.close(self.ljm_handle)
                logging.info("LabJack disconnected successfully.")

            if self.connection_a is not None:
                try:
                    self.connection_a.close()
                    logging.info("Zaber X-MCC1 disconnected successfully.")
                except Exception as e:
                    logging.error(f"Failed to disconnect Zaber X-MCC1: {e}")
            
            if self.connection_b is not None:
                try:
                    self.connection_b.close()
                    logging.info("Zaber X-MCC2 disconnected successfully.")
                except Exception as e:
                    logging.error(f"Failed to disconnect Zaber X-MCC2: {e}")

    def command_outlet(self, outlet, command):
        with self.peripherals_lock:
            if self.pdu is not None:
                try:
                    asyncio.run(self.pdu.send_outlet_command(outlet, command))
                    logging.info(f"Sent command {command} to outlet {outlet}.")
                except Exception as e:
                    logging.error(f"Failed to send command to outlet {outlet}: {e}")
            else:
                logging.info("PDU is not connected.")

    async def get_all_outlet_states(self):
        with self.peripherals_lock:
            if self.pdu is not None:
                try:
                    states = await self.pdu.get_all_outlet_states()
                    return states
                except Exception as e:
                    logging.error(f"Failed to get outlet states: {e}")
                    return {}
            else:
                logging.info("PDU is not connected.")
                return {}


class CameraGUI(tk.Tk):
    def __init__(self, shared_data, camera_thread, peripherals_thread,
                 frame_queue, timestamp_queue):
        super().__init__()
        self.shared_data = shared_data
        self.camera_thread = camera_thread
        self.peripherals_thread = peripherals_thread
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.updating_camera_status = True  # Flag for camera status update
        self.updating_frame_display = True  # Flag for frame display update
        self.updating_peripherals_status = True # Flag for peripherals update
        self.display_lock = threading.Lock()  # Lock for display operations
        self.save_thread = None
        self.last_frame = None

        # Initialize percentile variables within the Tkinter root window context
        self.min_val = tk.StringVar(value="0")  # Initial min percentile set to 0%
        self.max_val = tk.StringVar(value="200")  # Initial max percentile set to 200%

        self.title("Lightspeed Prototype Control GUI")
        self.geometry("1000x1000")  # Adjust window size to tighten the layout

        # GUI code snippet
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # LabelFrame for Camera Parameters
        camera_params_frame = LabelFrame(self.main_frame, text="Camera Parameters", padx=5, pady=5)
        camera_params_frame.grid(row=0, column=0, rowspan=10, sticky='nsew')
        self.camera_status = tk.Label(camera_params_frame, text="", justify=tk.LEFT, anchor="w", width=60, height=80, wraplength=400)
        self.camera_status.pack(fill='both', expand=True)

        # Label for status messages
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="w", width=40, wraplength=400, fg="blue")
        self.status_message.grid(row=5, column=0, columnspan=2, sticky='nsew')

        # Camera Controls
        camera_controls_frame = LabelFrame(self.main_frame, text="Camera Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1, sticky='n')

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=0, column=0)
        self.exposure_time_var = tk.DoubleVar()
        self.exposure_time_var.set(100)
        self.exposure_time_var.trace_add("write", self.update_exposure_time)
        self.exposure_time_entry = Entry(camera_controls_frame, textvariable=self.exposure_time_var)
        self.exposure_time_entry.grid(row=0, column=1)

        self.save_data_var = tk.BooleanVar()
        self.save_data_checkbox = Checkbutton(camera_controls_frame, text="Save Data to Disk", variable=self.save_data_var)
        self.save_data_checkbox.grid(row=1, column=0, columnspan=2)

        self.start_button = Button(camera_controls_frame, text="Start", command=self.start_capture)
        self.start_button.grid(row=2, column=0)

        self.stop_button = Button(camera_controls_frame, text="Stop", command=self.stop_capture)
        self.stop_button.grid(row=2, column=1)

        self.reset_button = Button(camera_controls_frame, text="Reset Camera", command=self.reset_camera)
        self.reset_button.grid(row=3, column=0, columnspan=2)

        Label(camera_controls_frame, text="Object Name:").grid(row=4, column=0)
        self.object_name_entry = Entry(camera_controls_frame)
        self.object_name_entry.insert(0, "")
        self.object_name_entry.grid(row=4, column=1)

        Label(camera_controls_frame, text="Frames per Datacube").grid(row=5, column=0)
        self.cube_size_var = tk.IntVar()
        self.batch_size = 100
        self.cube_size_var.set(self.batch_size)
        self.cube_size_var.trace_add("write", self.update_batch_size)
        self.cube_size_entry = Entry(camera_controls_frame, textvariable=self.cube_size_var)
        self.cube_size_entry.grid(row=5, column=1)

        # Make button to power cycle camera by turning off and on PDU outlet 1
        self.power_cycle_button = Button(camera_controls_frame, text="Power Cycle Camera",
                                         command=self.power_cycle_camera)
        self.power_cycle_button.grid(row=6, column=0, columnspan=2)

        # Camera Settings
        camera_settings_frame = LabelFrame(self.main_frame, text="Camera Settings", padx=5, pady=5)
        camera_settings_frame.grid(row=1, column=1, sticky='n')

        Label(camera_settings_frame, text="Binning:").grid(row=0, column=0)
        self.binning_var = StringVar(self)
        self.binning_var.set("1x1")
        self.binning_menu = OptionMenu(camera_settings_frame, self.binning_var, "1x1", "2x2", "4x4", command=self.change_binning)
        self.binning_menu.grid(row=0, column=1)

        Label(camera_settings_frame, text="Bit Depth:").grid(row=1, column=0)
        self.bit_depth_var = StringVar(self)
        self.bit_depth_var.set("8-bit")  # Set default to 8-bit
        self.bit_depth_menu = OptionMenu(camera_settings_frame, self.bit_depth_var, "8-bit", "16-bit", command=self.change_bit_depth)
        self.bit_depth_menu.grid(row=1, column=1)

        Label(camera_settings_frame, text="Readout Speed:").grid(row=2, column=0)
        self.readout_speed_var = StringVar(self)
        self.readout_speed_var.set("Ultra Quiet Mode")
        self.readout_speed_menu = OptionMenu(camera_settings_frame, self.readout_speed_var, "Ultra Quiet Mode", "Standard Mode", command=self.change_readout_speed)
        self.readout_speed_menu.grid(row=2, column=1)

        # Add the Sensor Mode dropdown
        Label(camera_settings_frame, text="Sensor Mode:").grid(row=3, column=0)
        self.sensor_mode_var = StringVar(self)
        self.sensor_mode_var.set("Photon Number Resolving")  # Set default to "Photon Number Resolving"
        self.sensor_mode_menu = OptionMenu(camera_settings_frame, self.sensor_mode_var, "Photon Number Resolving", "Standard", command=self.change_sensor_mode)
        self.sensor_mode_menu.grid(row=3, column=1)

        # Subarray Controls
        subarray_controls_frame = LabelFrame(self.main_frame, text="Subarray Controls", padx=5, pady=5)
        subarray_controls_frame.grid(row=2, column=1, sticky='n')

        Label(subarray_controls_frame, text="Subarray Mode:").grid(row=0, column=0)
        self.subarray_mode_var = StringVar(self)
        self.subarray_mode_var.set("Off")
        self.subarray_mode_menu = OptionMenu(subarray_controls_frame, self.subarray_mode_var, "Off", "On", command=self.change_subarray_mode)
        self.subarray_mode_menu.grid(row=0, column=1)

        Label(subarray_controls_frame, text="Subarray HPOS:").grid(row=1, column=0)
        self.subarray_hpos_var = tk.IntVar()
        self.subarray_hpos_var.set(0)
        self.subarray_hpos_var.trace_add("write", self.update_subarray)
        self.subarray_hpos_entry = Entry(subarray_controls_frame, textvariable=self.subarray_hpos_var)
        self.subarray_hpos_entry.grid(row=1, column=1)
        self.subarray_hpos_entry.config(state='disabled')  # Disable after inserting value

        Label(subarray_controls_frame, text="Subarray HSIZE:").grid(row=2, column=0)
        self.subarray_hsize_var = tk.IntVar()
        self.subarray_hsize_var.set(4096)
        self.subarray_hsize_var.trace_add("write", self.update_subarray)
        self.subarray_hsize_entry = Entry(subarray_controls_frame, textvariable=self.subarray_hsize_var)
        self.subarray_hsize_entry.grid(row=2, column=1)
        self.subarray_hsize_entry.config(state='disabled')  # Disable after inserting value

        Label(subarray_controls_frame, text="Subarray VPOS:").grid(row=3, column=0)
        self.subarray_vpos_var = tk.IntVar()
        self.subarray_vpos_var.set(0)
        self.subarray_vpos_var.trace_add("write", self.update_subarray)
        self.subarray_vpos_entry = Entry(subarray_controls_frame, textvariable=self.subarray_vpos_var)
        self.subarray_vpos_entry.grid(row=3, column=1)
        self.subarray_vpos_entry.config(state='disabled')  # Disable after inserting value

        Label(subarray_controls_frame, text="Subarray VSIZE:").grid(row=4, column=0)
        self.subarray_vsize_var = tk.IntVar()
        self.subarray_vsize_var.set(2304)
        self.subarray_vsize_var.trace_add("write", self.update_subarray)
        self.subarray_vsize_entry = Entry(subarray_controls_frame, textvariable=self.subarray_vsize_var)
        self.subarray_vsize_entry.grid(row=4, column=1)
        self.subarray_vsize_entry.config(state='disabled')  # Disable after inserting value

        # Add text noting that values will be rounded to nearest factor of 4
        subarray_note = "Note: Values will be rounded to nearest factor of 4."
        Label(subarray_controls_frame, text=subarray_note).grid(row=5, column=0, columnspan=2)

        # Advanced Controls
        advanced_controls_frame = LabelFrame(self.main_frame, text="Advanced Controls", padx=5, pady=5)
        advanced_controls_frame.grid(row=3, column=1, sticky='n')

        self.framebundle_var = tk.BooleanVar()
        self.framebundle_checkbox = Checkbutton(advanced_controls_frame, text="Enable Frame Bundle", variable=self.framebundle_var, command=self.update_framebundle)
        self.framebundle_checkbox.grid(row=0, column=0, columnspan=2)

        Label(advanced_controls_frame, text="Frames Per Bundle:").grid(row=1, column=0)
        self.frames_per_bundle_var = tk.IntVar()
        self.frames_per_bundle_var.set(100)
        self.frames_per_bundle_var.trace_add("write", self.update_frames_per_bundle)
        self.frames_per_bundle_entry = Entry(advanced_controls_frame, textvariable=self.frames_per_bundle_var)
        self.frames_per_bundle_entry.grid(row=1, column=1)
        # Add text noting what frame bundling means
        subarray_note = "When enabled, this many frames will \nbe concatenated into one image."
        Label(advanced_controls_frame, text=subarray_note).grid(row=2, column=0, columnspan=2)

        # Display Controls with side-by-side layout for Min and Max Percentile
        display_controls_frame = LabelFrame(self.main_frame, text="Display Controls", padx=5, pady=5)
        display_controls_frame.grid(row=4, column=1, sticky='n')

        Label(display_controls_frame, text="Min Count:").grid(row=0, column=0)
        self.min_entry = Entry(display_controls_frame, textvariable=self.min_val)
        self.min_entry.grid(row=0, column=1)

        Label(display_controls_frame, text="Max Count:").grid(row=0, column=2)
        self.max_entry = Entry(display_controls_frame, textvariable=self.max_val)
        self.max_val.trace_add("write", self.refresh_frame_display)
        self.max_entry.grid(row=0, column=3)

        # Peripherals Controls
        self.peripherals_controls_frame = LabelFrame(self.main_frame, text="Peripherals Controls", padx=5, pady=5)
        self.peripherals_controls_frame.grid(row=5, column=1, sticky='n')

        # Make menu to select filter position
        Label(self.peripherals_controls_frame, text="Filter:").grid(row=0, column=0)
        self.filter_position_var = tk.StringVar()
        self.filter_options = {'0 (Open)': 0, '1 (u\')': 1, '2 (g\')': 2, '3 (r\')': 3,
                                        '4 (i\')': 4, '5 (z\')': 5}
        self.filter_position_menu = OptionMenu(self.peripherals_controls_frame, self.filter_position_var,
                                               *self.filter_options.keys(),
                                               command=self.update_filter_position)
        self.filter_position_menu.grid(row=0, column=1)

        # Make menu to open or close shutter
        Label(self.peripherals_controls_frame, text="Shutter:").grid(row=0, column=2)
        self.shutter_var = tk.StringVar()
        self.shutter_var.set('Open')
        self.shutter_options = ['Open', 'Closed']
        self.shutter_menu = OptionMenu(self.peripherals_controls_frame, self.shutter_var,
                                       *self.shutter_options, command=self.update_shutter)
        self.shutter_menu.grid(row=0, column=3)

        # Make menu to select whether slit should be in or out of the beam
        Label(self.peripherals_controls_frame, text="Slit:").grid(row=1, column=0)
        self.slit_position_var = tk.StringVar()
        self.slit_position_var.set('Out of beam')
        self.slit_options = ['In beam', 'Out of beam']
        self.slit_position_menu = OptionMenu(self.peripherals_controls_frame, self.slit_position_var,
                                             *self.slit_options, command=self.update_slit_position)
        self.slit_position_menu.grid(row=1, column=1)

        # Make menu to select whether Halpha filter, QWP, or neither should be in the beam
        Label(self.peripherals_controls_frame, text="Halpha/QWP:").grid(row=1, column=2)
        self.halpha_qwp_var = tk.StringVar()
        self.halpha_qwp_var.set('Neither')
        self.halpha_qwp_options = ['Halpha', 'QWP', 'Neither']
        self.halpha_qwp_menu = OptionMenu(self.peripherals_controls_frame, self.halpha_qwp_var,
                                          *self.halpha_qwp_options, command=self.update_halpha_qwp)
        self.halpha_qwp_menu.grid(row=1, column=3)

        # Make menu to select whether WeDoWo, Wire Grid, or neither should be in the beam
        Label(self.peripherals_controls_frame, text="Pol. Stage:").grid(row=2, column=0)
        self.wire_grid_var = tk.StringVar()
        self.wire_grid_var.set('Neither')
        self.wire_grid_options = ['WeDoWo', 'Wire Grid', 'Neither']
        self.wire_grid_menu = OptionMenu(self.peripherals_controls_frame, self.wire_grid_var,
                                         *self.wire_grid_options, command=self.update_pol_stage)
        self.wire_grid_menu.grid(row=2, column=1)

        # Make menu for zoom stepper motor options
        Label(self.peripherals_controls_frame, text="Zoom-out:").grid(row=2, column=2)
        self.zoom_stepper_var = tk.StringVar()
        self.zoom_stepper_var.set('4x')
        self.zoom_stepper_options = ['1x', '2x', '3x', '4x']
        self.zoom_stepper_menu = OptionMenu(self.peripherals_controls_frame, self.zoom_stepper_var,
                                            *self.zoom_stepper_options, command=self.update_zoom_stepper)
        self.zoom_stepper_menu.grid(row=2, column=3)

        # Make box to specify focus position. Only set focus when a button is pressed.
        Label(self.peripherals_controls_frame, text="Focus Position (um):").grid(row=3, column=0, columnspan=2)
        self.focus_position_var = tk.StringVar()
        self.focus_position_var.set('0')  # Default focus position
        self.focus_conversion_factor = 1  # Microns of focus per degree of stepper motor turn ZZZ need to measure this
        self.focus_position_entry = Entry(self.peripherals_controls_frame, textvariable=self.focus_position_var)
        self.focus_position_entry.grid(row=3, column=2)
        self.set_focus_button = Button(self.peripherals_controls_frame, text="Set Focus",
                                       command=self.update_focus_position)
        self.set_focus_button.grid(row=3, column=3)

        Label(self.peripherals_controls_frame, text="PDU Outlet States").grid(row=4, column=0, columnspan=4)
        # Make set of 16 switches to control the CyberPower PDU outlets
        self.pdu_outlet_dict = {1: 'Rotator', 2: 'Switch', 3: 'Shutter', 4: 'Empty',
                                5: 'Empty', 6: 'Empty', 7: 'Empty', 8: 'Empty',
                                9: 'Ctrl PC', 10: 'X-MCC4 A', 11: 'X-MCC B', 12: 'qCMOS',
                                13: 'Empty', 14: 'Empty', 15: 'Empty', 16: 'Empty'}
        self.pdu_outlet_vars = {}
        self.pdu_outlet_buttons = {}
        for idx, name in self.pdu_outlet_dict.items():
            # eight outlets per column, two columns
            row = (idx - 1) % 8 + 8
            col = (idx - 1) // 8 * 2
            name = str(idx) + ': ' + name  # Add outlet number to the name
            # label on left
            tk.Label(self.peripherals_controls_frame, text=name, width=12, anchor='w')\
              .grid(row=row, column=col, padx=2, pady=2)
            # on/off variable & button
            var = tk.BooleanVar(value=True)
            self.pdu_outlet_vars[idx] = var
            btn = tk.Checkbutton(self.peripherals_controls_frame, text='ON', relief='sunken',
                                 fg='green', variable=var, indicatoron=False, width=3,
                                 command=lambda i=idx: self.toggle_outlet(i))
            btn.grid(row=row, column=col + 1, padx=2, pady=2)
            self.pdu_outlet_buttons[idx] = btn

        # Start the update loops
        self.update_camera_status()
        self.update_frame_display()
        self.update_peripherals_status()

    def update_camera_status(self):
        if self.updating_camera_status:
            try:
                self.refresh_camera_status()
            except Exception as e:
                logging.error(f"Error updating camera status: {e}")
        self.after(1000, self.update_camera_status)  # Update camera status every second

    def update_frame_display(self):
        if self.updating_frame_display:
            try:
                self.refresh_frame_display()
            except Exception as e:
                logging.error(f"Error updating frame display: {e}")
        self.after(17, self.update_frame_display)  # Update frame display every ~17 ms (about 60 FPS)

    def update_peripherals_status(self):
        if self.updating_peripherals_status and self.peripherals_thread is not None:
            try:
                self.refresh_peripherals_status()
            except Exception as e:
                logging.error(f"Error updating peripherals status: {e}")
        self.after(3000, self.update_peripherals_status)  # Update peripherals status every 3 seconds

    def refresh_camera_status(self):
        status_text = ""
        with self.shared_data.lock:
            for key, value in self.shared_data.camera_params.items():
                if key in DISPLAY_PARAMS.keys():
                    status_text += f"{DISPLAY_PARAMS[key]}: {value}\n"
        self.camera_status.config(text=status_text)

    def refresh_frame_display(self, *_):
        try:
            frame = self.frame_queue.get_nowait()
            with self.display_lock:
                self.process_frame(frame)  # Display the frame using OpenCV
        except queue.Empty:
            cv2.waitKey(1)  # Keep the window interactive when no frames are available

    def refresh_peripherals_status(self):
        try:
            self.update_filter_position()
            self.update_outlet_states()
            self.update_shutter()
            self.update_slit_position()
            self.update_halpha_qwp()
            self.update_pol_stage()
            self.update_zoom_stepper()
        except Exception as e:
            logging.error(f"Error in refresh_peripherals_status: {e}")

    def process_frame(self, data):
        try:
            # Handle both 8-bit and 16-bit data
            if data.dtype == np.uint16 or data.dtype == np.uint8:
                try:
                    min_val = int(self.min_val.get())
                except:
                    min_val = 0
                try:
                    max_val = int(self.max_val.get())
                except:
                    max_val = 200

                # Make a copy to avoid modifying original data
                data_copy = np.copy(data)

                # Scale data appropriately
                if data_copy.dtype == np.uint16:
                    scaled_data = np.clip((data_copy - min_val) / (max_val - min_val) * 65535, 0, 65535).astype(np.uint16)
                elif data_copy.dtype == np.uint8:
                    scaled_data = np.clip((data_copy - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

                # Flip left-right to match sky orientation
                scaled_data = cv2.flip(scaled_data, 1)

                # Convert to BGR for consistent color display
                scaled_data_bgr = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2BGR)

                # Draw the circle if it exists
                if hasattr(self, 'circle_center'):
                    cv2.circle(scaled_data_bgr, self.circle_center, 2, (255, 0, 0), 2)

                # Resizable OpenCV window with right-click context menu
                if not hasattr(self, 'opencv_window_created'):
                    cv2.namedWindow('Captured Frame', cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback('Captured Frame', self.on_right_click)
                    self.opencv_window_created = True

                cv2.imshow('Captured Frame', scaled_data_bgr)
                cv2.waitKey(1)
                self.last_frame = data_copy  # Store a copy of the last frame
            else:
                logging.error(f"Unsupported data type: {data.dtype}")
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            logging.error(traceback.format_exc())

    def on_right_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Draw Circle", command=lambda: self.draw_circle(x, y))
            menu.add_command(label="Other Option", command=lambda: print("Other Option Selected"))
            try:
                menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
            except Exception as e:
                logging.error(f"Error showing popup menu: {e}")

    def draw_circle(self, x, y):
        # Store the circle's position
        self.circle_center = (x, y)

        # Redraw the current frame with the circle
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            with self.display_lock:
                self.process_frame(self.last_frame)

    def update_exposure_time(self, *_):
        try:
            exposure_time = float(self.exposure_time_entry.get()) / 1000
            if self.camera_thread.capturing:
                logging.warning("Cannot change exposure time during active capture.")
                self.status_message.config(text="Cannot change exposure time during active capture.")
            else:
                self.camera_thread.set_property('EXPOSURE_TIME', exposure_time)
                self.status_message.config(text=f"Exposure time set to {exposure_time*1000:.1f}ms")
        except ValueError:
            logging.error("Invalid input for exposure time")
            self.status_message.config(text="Invalid exposure time value")

    def update_batch_size(self, *_):
        try:
            self.batch_size = int(self.cube_size_entry.get())
            logging.info(f"Batch size set to {self.batch_size} frames per cube.")
        except:
            logging.error("Invalid number of frames per cube. Setting to 100.")
            self.batch_size = 100

    def update_shutter(self, *_):
        try:
            # Check that LabJack is connected.
            if self.peripherals_thread.ljm_handle is None:
                self.peripherals_thread.connect_labjack()
                if self.peripherals_thread.ljm_handle is None:
                    return
            # Status of the DIO4 port controlling the shutter
            with self.peripherals_thread.peripherals_lock:
                mask = ljm.eReadName(self.peripherals_thread.ljm_handle, "FIO_STATE")
                fio4_state = (int(mask) >> 4) & 1
                if fio4_state == 0:
                    shutter_state = 'Open'
                elif fio4_state == 1:
                    shutter_state = 'Closed'
                else:
                    logging.error(f'LabJack malfunction. DIO4 state {fio4_state}')
                if self.shutter_var.get() == shutter_state:
                    return
                if self.shutter_var.get() == 'Open':
                    ljm.eWriteName(self.peripherals_thread.ljm_handle, "DIO4", 0)
                elif self.shutter_var.get() == 'Closed':
                    ljm.eWriteName(self.peripherals_thread.ljm_handle, "DIO4", 1)
        except Exception as e:
            logging.error(f"Error updating shutter: {e}")

    def update_filter_position(self, *_):
        try:
            selected_filter = self.filter_position_var.get()
            # Check if EFW has gotten disconnected. If it has, try to connect.
            with self.peripherals_thread.peripherals_lock:
                try:
                    if self.peripherals_thread.efw is None:
                        raise Exception("EFW not connected")
                    current_position = self.peripherals_thread.efw.GetPosition(0)
                except:
                    self.peripherals_thread.efw = None
                    self.peripherals_thread.connect_efw()
                    if self.peripherals_thread.efw is None:
                        self.filter_position_menu.config(state='disabled')
                        self.status_message.config(text="Filter wheel not connected.", fg="red")
                    else:
                        self.filter_position_menu.config(state='normal')
                    return
                # If EFW is connected, update the filter position if necessary.
                if selected_filter == '':
                    current_position = self.peripherals_thread.efw.GetPosition(0)
                    self.filter_position_var.set(list(self.filter_options.keys())[current_position])
                else:
                    selected_position = self.filter_options[self.filter_position_var.get()]
                    if selected_position == self.peripherals_thread.efw.GetPosition(0):
                        return
                    try:
                        self.peripherals_thread.efw.SetPosition(0, selected_position)
                        logging.info(f"Filter position set to {self.filter_position_var.get()}")
                    except Exception as e:
                        logging.error(f"Error setting filter position: {e}")
        except Exception as e:
            logging.error(f"Error in update_filter_position: {e}")

    def update_outlet_states(self):
        try:
            if self.peripherals_thread.pdu is None:
                self.status_message.config(text="PDU not connected.", fg="red")
                return
            outlet_states = asyncio.run(self.peripherals_thread.get_all_outlet_states())
            for i in range(1, 17):
                if outlet_states and len(outlet_states) >= i:
                    if self.pdu_outlet_vars[i].get() != outlet_states[i - 1]:
                        # Update the variable and button state
                        self.pdu_outlet_vars[i].set(outlet_states[i - 1])
                        self.toggle_outlet(i, override=True)  # Update button appearance
        except Exception as e:
            logging.error(f"Error updating outlet states: {e}")

    def power_cycle_camera(self):
        try:
            # Power cycle the camera by turning off and on the PDU outlet    
            self.peripherals_thread.command_outlet(2, OutletCommand.IMMEDIATE_OFF)
            time.sleep(1)  # Wait for 1 second before turning it back on
            self.peripherals_thread.command_outlet(2, OutletCommand.IMMEDIATE_ON)
            logging.info("Camera power cycled. Will try to reconnect.")
            self.camera_thread.stop()
            self.camera_thread = CameraThread(self.shared_data, self.frame_queue,
                                              self.timestamp_queue, self)
            self.camera_thread.daemon = True
            self.camera_thread.start()
        except Exception as e:
            logging.error(f"Error power cycling camera: {e}")

    def update_slit_position(self, *_):
        try:
            if self.peripherals_thread.ax_a_1 is None:
                self.status_message.config(text="Slit stage not connected.", fg="red")
                # Make slit menu disabled
                self.slit_position_menu.config(state='disabled')
                return
            slit_option = self.slit_position_var.get()
            with self.peripherals_thread.peripherals_lock:
                curr_slit_pos = self.peripherals_thread.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
                if slit_option == 'In beam' and abs(curr_slit_pos - 70) > 0.01:
                    logging.info("Moving slit in beam.")
                    self.peripherals_thread.ax_a_1.move_absolute(70, Units.LENGTH_MILLIMETRES)
                elif slit_option == 'Out of beam' and abs(curr_slit_pos) > 0.01:
                    logging.info("Moving slit out of beam.")
                    self.peripherals_thread.ax_a_1.move_absolute(0, Units.LENGTH_MILLIMETRES)
        except Exception as e:
            logging.error(f"Error updating slit position: {e}")
    
    def update_halpha_qwp(self, *_):
        try:
            if self.peripherals_thread.ax_b_3 is None:
                self.status_message.config(text="Halpha/QWP stage not connected.", fg="red")
                # Make Halpha/QWP menu disabled and blank
                self.halpha_qwp_var.set('')
                self.halpha_qwp_menu.config(state='disabled')
                return
            halpha_qwp_option = self.halpha_qwp_var.get()
            with self.peripherals_thread.peripherals_lock:
                curr_halpha_qwp_pos = self.peripherals_thread.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
                if halpha_qwp_option == 'Halpha' and abs(curr_halpha_qwp_pos - 138) > 0.01:
                    logging.info("Moving Halpha filter in beam.")
                    self.peripherals_thread.ax_b_3.move_absolute(138, Units.LENGTH_MILLIMETRES)
                elif halpha_qwp_option == 'QWP' and abs(curr_halpha_qwp_pos - 13) > 0.01:
                    logging.info("Moving QWP in beam.")
                    self.peripherals_thread.ax_b_3.move_absolute(13, Units.LENGTH_MILLIMETRES)
                elif halpha_qwp_option == 'Neither' and abs(curr_halpha_qwp_pos - 75.5) > 0.01:
                    logging.info("Moving Halpha filter and QWP out of beam.")
                    self.peripherals_thread.ax_b_3.move_absolute(75.5, Units.LENGTH_MILLIMETRES)
        except Exception as e:
            logging.error(f"Error updating Halpha/QWP: {e}")

    def update_pol_stage(self, *_):
        try:
            if self.peripherals_thread.ax_b_2 is None:
                self.status_message.config(text="Polarization stage not connected.", fg="red")
                # Make wire grid menu disabled and blank
                self.wire_grid_var.set('')
                self.wire_grid_menu.config(state='disabled')
                return
            wire_grid_option = self.wire_grid_var.get()
            with self.peripherals_thread.peripherals_lock:
                curr_wire_grid_pos = self.peripherals_thread.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
                if wire_grid_option == 'WeDoWo' and abs(curr_wire_grid_pos - 15.5) > 0.01:
                    logging.info("Moving WeDoWo in beam.")
                    self.peripherals_thread.ax_b_2.move_absolute(15.5, Units.LENGTH_MILLIMETRES)
                elif wire_grid_option == 'Wire Grid' and abs(curr_wire_grid_pos - 128.5) > 0.01:
                    logging.info("Moving Wire Grid in beam.")
                    self.peripherals_thread.ax_b_2.move_absolute(128.5, Units.LENGTH_MILLIMETRES)
                elif wire_grid_option == 'Neither' and abs(curr_wire_grid_pos - 72) > 0.01:
                    logging.info("Moving WeDoWo and Wire Grid out of beam.")
                    self.peripherals_thread.ax_b_2.move_absolute(72, Units.LENGTH_MILLIMETRES)
        except Exception as e:
            logging.error(f"Error updating polarization stage: {e}")

    def update_zoom_stepper(self, *_):
        try:
            if self.peripherals_thread.ax_a_2 is None:
                self.status_message.config(text="Zoom stepper not connected.", fg="red")
                # Make zoom stepper menu disabled and blank
                self.zoom_stepper_var.set('')
                self.zoom_stepper_menu.config(state='disabled')
                return
            zoom_option = self.zoom_stepper_var.get()
            zoom_positions = {'1x': 0, '2x': 90, '3x': 180, '4x': 270}
            desired_position = zoom_positions[zoom_option]
            with self.peripherals_thread.peripherals_lock:
                curr_zoom_pos = self.peripherals_thread.ax_a_2.get_position(Units.ANGLE_DEGREES)
                if abs(curr_zoom_pos - desired_position) < 0.01:
                    return
                else:
                    logging.info(f"Moving zoom stepper to {zoom_option}.")
                    self.peripherals_thread.ax_a_2.move_absolute(desired_position, Units.ANGLE_DEGREES)
        except Exception as e:
            logging.error(f"Error updating zoom stepper: {e}")
    
    def update_focus_position(self, *_):
        try:
            if self.peripherals_thread.ax_b_1 is None:
                self.status_message.config(text="Focus stage not connected.", fg="red")
                # disable focus position entry and button
                self.focus_position_entry.config(state='disabled')
                self.set_focus_button.config(state='disabled')
                return
            focus_position = float(self.focus_position_var.get()) # in um
            with self.peripherals_thread.peripherals_lock:
                curr_stepper_pos = self.peripherals_thread.ax_b_1.get_position(Units.ANGLE_DEGREES)
                desired_stepper_pos = focus_position / self.focus_conversion_factor
                if abs(curr_stepper_pos - desired_stepper_pos) < 0.01:
                    return
                else:
                    logging.info(f"Moving focus stage to {focus_position} um.")
                    self.peripherals_thread.ax_b_1.move_absolute(desired_stepper_pos, Units.ANGLE_DEGREES)
        except Exception as e:
            logging.error(f"Error updating focus position: {e}")

    def toggle_outlet(self, idx, override=False):
        """Send on/off and update button color/text."""
        try:
            time.sleep(0.1)  # Allow some time for the button state to update
            state = self.pdu_outlet_vars[idx].get()
            if not override:
                # If not overriding, check that user actually wants to toggle
                confirm_button = messagebox.askyesno("Confirm Outlet Toggle",
                                                     f"Are you sure you want to turn {'ON' if state else 'OFF'} outlet {idx}?")
                if not confirm_button:
                    # If user cancels, reset the button state
                    self.pdu_outlet_vars[idx].set(not state)
                    return
            cmd = OutletCommand.IMMEDIATE_ON if state else OutletCommand.IMMEDIATE_OFF
            # fire the PDU command
            self.peripherals_thread.command_outlet(idx, cmd)
            time.sleep(1)  # Allow some time for the command to take effect
            # update the button widget (find by its grid position)
            btn = self.pdu_outlet_buttons[idx]
            if state:
                btn.config(text='ON', fg='green', relief='sunken')
            else:
                btn.config(text='OFF', fg='red', relief='raised')
        except Exception as e:
            logging.error(f"Error toggling outlet {idx}: {e}")

    def change_binning(self, selected_binning):
        if self.camera_thread.capturing:
            logging.warning("Cannot change binning during active capture.")
            self.status_message.config(text="Cannot change binning during active capture.")
            # Reset to current value
            current_binning = self.binning_var.get()
            self.binning_var.set(current_binning)
        else:
            binning_value = {"1x1": 1, "2x2": 2, "4x4": 4}[selected_binning]
            self.camera_thread.set_property('BINNING', binning_value)

    def change_bit_depth(self, selected_bit_depth):
        if self.camera_thread.capturing:
            logging.warning("Cannot change bit depth during active capture.")
            self.status_message.config(text="Cannot change bit depth during active capture.")
            # Reset to current value
            current_bit_depth = self.bit_depth_var.get()
            self.bit_depth_var.set(current_bit_depth)
        else:
            bit_depth_value = {"8-bit": 1, "16-bit": 2}[selected_bit_depth]
            self.camera_thread.set_property('IMAGE_PIXEL_TYPE', bit_depth_value)

    def change_readout_speed(self, selected_mode):
        if self.camera_thread.capturing:
            logging.warning("Cannot change readout speed during active capture.")
            self.status_message.config(text="Cannot change readout speed during active capture.")
            # Reset to current value
            current_readout_speed = self.readout_speed_var.get()
            self.readout_speed_var.set(current_readout_speed)
        else:
            readout_speed_value = {"Ultra Quiet Mode": 1.0, "Standard Mode": 2.0}[selected_mode]
            if selected_mode == "Standard Mode":
                self.sensor_mode_var.set("Standard")
                self.change_sensor_mode("Standard")
            self.camera_thread.set_property('READOUT_SPEED', readout_speed_value)

    def change_sensor_mode(self, selected_mode):
        if self.camera_thread.capturing:
            logging.warning("Cannot change sensor mode during active capture.")
            self.status_message.config(text="Cannot change sensor mode during active capture.")
            # Reset to current value
            current_sensor_mode = self.sensor_mode_var.get()
            self.sensor_mode_var.set(current_sensor_mode)
        else:
            sensor_mode_value = {"Photon Number Resolving": 18.0, "Standard": 1.0}[selected_mode]
            self.camera_thread.set_property('SENSOR_MODE', sensor_mode_value)
            if selected_mode == "Photon Number Resolving":
                self.readout_speed_var.set("Ultra Quiet Mode")
                self.change_readout_speed("Ultra Quiet Mode")

    def change_subarray_mode(self, selected_mode):
        if self.camera_thread.capturing:
            logging.warning("Cannot change subarray mode during active capture.")
            self.status_message.config(text="Cannot change subarray mode during active capture.")
            # Reset to current value
            current_subarray_mode = self.subarray_mode_var.get()
            self.subarray_mode_var.set(current_subarray_mode)
        else:
            subarray_mode_value = {"Off": 1.0, "On": 2.0}[selected_mode]
            self.camera_thread.set_property('SUBARRAY_MODE', subarray_mode_value)
            if selected_mode == "On":
                self.subarray_hpos_entry.config(state='normal')
                self.subarray_hsize_entry.config(state='normal')
                self.subarray_vpos_entry.config(state='normal')
                self.subarray_vsize_entry.config(state='normal')
            else:
                self.subarray_hpos_entry.config(state='disabled')
                self.subarray_hsize_entry.config(state='disabled')
                self.subarray_vpos_entry.config(state='disabled')
                self.subarray_vsize_entry.config(state='disabled')

    def update_subarray(self, *_):
        if self.camera_thread.capturing:
            logging.warning("Cannot change subarray parameters during active capture.")
            self.status_message.config(text="Cannot change subarray parameters during active capture.")
        else:
            try:
                # Need to round values to nearest factor of 4
                hpos = float(round(float(self.subarray_hpos_entry.get()) / 4) * 4)
                hsize = float(round(float(self.subarray_hsize_entry.get()) / 4) * 4)
                vpos = float(round(float(self.subarray_vpos_entry.get()) / 4) * 4)
                vsize = float(round(float(self.subarray_vsize_entry.get()) / 4) * 4)

                self.camera_thread.set_property('SUBARRAY_HPOS', hpos)
                self.camera_thread.set_property('SUBARRAY_HSIZE', hsize)
                self.camera_thread.set_property('SUBARRAY_VPOS', vpos)
                self.camera_thread.set_property('SUBARRAY_VSIZE', vsize)
            except ValueError:
                logging.error("Invalid input for subarray parameters")

    def update_framebundle(self):
        if self.camera_thread.capturing:
            logging.warning("Cannot change frame bundle during active capture.")
            self.status_message.config(text="Cannot change frame bundle during active capture.")
            # Reset to current value
            current_framebundle = self.framebundle_var.get()
            self.framebundle_var.set(current_framebundle)
        else:
            framebundle_enabled = self.framebundle_var.get()
            self.camera_thread.set_property('FRAMEBUNDLE_MODE', 2.0 if framebundle_enabled else 1.0)

    def update_frames_per_bundle(self, *_):
        if self.camera_thread.capturing:
            logging.warning("Cannot change frames per bundle during active capture.")
            self.status_message.config(text="Cannot change frames per bundle during active capture.")
        else:
            try:
                frames_per_bundle = int(self.frames_per_bundle_entry.get())
                self.camera_thread.set_property('FRAMEBUNDLE_NUMBER', frames_per_bundle)
            except ValueError:
                logging.error("Invalid input for frames per bundle")

    def start_capture(self):
        try:
            self.status_message.config(text="Capture started...")

            if self.save_data_var.get():
                self.save_queue = queue.Queue(maxsize=10000)  # Limit queue size
                self.camera_thread.save_queue = self.save_queue
                object_name = self.object_name_entry.get()
                self.save_thread = SaveThread(self.save_queue, self.timestamp_queue, self.camera_thread, object_name, self.shared_data)
                self.save_thread.batch_size = self.batch_size
                self.save_thread.start()
            else:
                self.camera_thread.save_queue = None

            self.camera_thread.start_capture()

            # Disable controls during capture
            self.exposure_time_entry.config(state='disabled')
            self.save_data_checkbox.config(state='disabled')
            self.start_button.config(state='disabled')
            self.reset_button.config(state='disabled')
            self.binning_menu.config(state='disabled')
            self.bit_depth_menu.config(state='disabled')
            self.readout_speed_menu.config(state='disabled')
            self.sensor_mode_menu.config(state='disabled')
            self.subarray_mode_menu.config(state='disabled')
            self.subarray_hpos_entry.config(state='disabled')
            self.subarray_hsize_entry.config(state='disabled')
            self.subarray_vpos_entry.config(state='disabled')
            self.subarray_vsize_entry.config(state='disabled')
            self.framebundle_checkbox.config(state='disabled')
            self.frames_per_bundle_entry.config(state='disabled')
        except Exception as e:
            logging.error(f"Error starting capture: {e}")
            logging.error(traceback.format_exc())
            self.status_message.config(text=f"Error starting capture: {e}")

    def stop_capture(self):
        try:
            self.status_message.config(text="Capture stopped...")
            self.camera_thread.stop_capture()

            if hasattr(self, 'save_thread') and self.save_thread is not None and self.save_thread.is_alive():
                self.save_thread.stop()
                self.save_thread.join(timeout=10)  # Increased timeout
                self.save_thread = None
                self.camera_thread.save_queue = None

            # Enable controls after capture
            self.exposure_time_entry.config(state='normal')
            self.save_data_checkbox.config(state='normal')
            self.start_button.config(state='normal')
            self.reset_button.config(state='normal')
            self.binning_menu.config(state='normal')
            self.bit_depth_menu.config(state='normal')
            self.readout_speed_menu.config(state='normal')
            self.sensor_mode_menu.config(state='normal')
            self.subarray_mode_menu.config(state='normal')

            # Enable subarray entries based on the current mode
            if self.subarray_mode_var.get() == "On":
                self.subarray_hpos_entry.config(state='normal')
                self.subarray_hsize_entry.config(state='normal')
                self.subarray_vpos_entry.config(state='normal')
                self.subarray_vsize_entry.config(state='normal')
            else:
                self.subarray_hpos_entry.config(state='disabled')
                self.subarray_hsize_entry.config(state='disabled')
                self.subarray_vpos_entry.config(state='disabled')
                self.subarray_vsize_entry.config(state='disabled')

            self.framebundle_checkbox.config(state='normal')
            self.frames_per_bundle_entry.config(state='normal')
        except Exception as e:
            logging.error(f"Error stopping capture: {e}")
            logging.error(traceback.format_exc())

    def reset_camera(self):
        try:
            # Disable GUI updates
            self.updating_camera_status = False
            self.updating_frame_display = False

            # Reset camera (pausing and resuming the camera thread)
            self.camera_thread.reset_camera()

            # Re-enable GUI updates
            self.updating_camera_status = True
            self.updating_frame_display = True
        except Exception as e:
            logging.error(f"Error resetting camera: {e}")
            logging.error(traceback.format_exc())

    def on_close(self):
        try:
            logging.info("Closing application...")
            
            # Stop updating loops
            self.updating_camera_status = False
            self.updating_frame_display = False
            self.updating_peripherals_status = False
            
            # Stop the camera thread
            if hasattr(self, 'camera_thread') and self.camera_thread is not None and self.camera_thread.is_alive():
                self.camera_thread.stop()
                self.camera_thread.join(timeout=10)  # Wait for thread to finish

            # Stop the save thread if it exists
            if hasattr(self, 'save_thread') and self.save_thread is not None and self.save_thread.is_alive():
                self.save_thread.stop()
                self.save_thread.join(timeout=10)

            # Ensure the OpenCV window is closed
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                logging.error(f"Error closing OpenCV windows: {e}")

            # Disconnect peripherals if they are connected
            if hasattr(self, 'peripherals_thread') and self.peripherals_thread is not None:
                self.peripherals_thread.disconnect_peripherals()
                if self.peripherals_thread.is_alive():
                    self.peripherals_thread.join(timeout=10)

            # Close the GUI window
            self.destroy()
                
            logging.info("Application closed successfully")
        except Exception as e:
            logging.error(f"Error during application close: {e}")
            logging.error(traceback.format_exc())
            # Force exit if graceful shutdown fails
            import sys
            sys.exit(1)


if __name__ == "__main__":
    try:
        logging.info("Starting Camera Control Application...")
        
        shared_data = SharedData()
        frame_queue = queue.Queue(maxsize=10)  # Increased queue size slightly
        timestamp_queue = queue.Queue(maxsize=1000)

        # Create the root window first
        app = CameraGUI(shared_data, None, None, frame_queue, timestamp_queue)

        # Initialize the camera thread with a reference to the GUI
        camera_thread = CameraThread(shared_data, frame_queue, timestamp_queue, app)
        camera_thread.daemon = True  # Make thread daemon to ensure cleanup
        camera_thread.start()

        # Initialize peripheral devices
        peripherals_thread = PeripheralsThread(shared_data, frame_queue, timestamp_queue,
                                               "18.25.72.251", "/dev/ttyACM0", "/dev/ttyACM1", app)
        peripherals_thread.daemon = True  # Make thread daemon to ensure cleanup
        peripherals_thread.start()

        app.camera_thread = camera_thread
        app.peripherals_thread = peripherals_thread

        app.protocol("WM_DELETE_WINDOW", app.on_close)
        app.mainloop()
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        logging.error(traceback.format_exc())
        import sys
        sys.exit(1)
