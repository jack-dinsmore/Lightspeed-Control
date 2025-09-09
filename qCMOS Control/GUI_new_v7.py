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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)

# Configure logging - reduced console output, full file logging
logging.basicConfig(
    level=logging.INFO,  # Show INFO level to console for frame updates
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Add file handler with DEBUG level for full logging
file_handler = logging.FileHandler('camera_debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(threadName)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
))
logging.getLogger().addHandler(file_handler)

# Create a separate logger for verbose operations
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_logger.propagate = False
debug_logger.addHandler(file_handler)

def log_dcam_call(func):
    """Decorator to log all DCAM API calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        debug_logger.debug(f"Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            debug_logger.debug(f"{func.__name__} completed")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} failed: {e}")
            debug_logger.error(traceback.format_exc())
            raise
    return wrapper


class DCamLock:
    """Global lock for DCAM API calls to ensure thread safety"""
    _lock = threading.RLock()
    _lock_holder = None
    _lock_count = 0
    _lock_time = None
    
    @classmethod
    def __enter__(cls):
        acquired = cls._lock.acquire(timeout=10)
        if not acquired:
            logging.error(f"Failed to acquire DCamLock - possible deadlock. Holder: {cls._lock_holder}")
            # Try to diagnose the issue
            if cls._lock_time and time.time() - cls._lock_time > 30:
                logging.error(f"Lock held for {time.time() - cls._lock_time:.1f} seconds!")
            raise TimeoutError("Failed to acquire DCamLock")
        
        cls._lock_holder = threading.current_thread().name
        cls._lock_count += 1
        cls._lock_time = time.time()
        return cls
    
    @classmethod
    def __exit__(cls, *args):
        cls._lock_count -= 1
        if cls._lock_count == 0:
            cls._lock_holder = None
            cls._lock_time = None
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
            if frame is not None:
                self.frames[safe_index] = np.copy(frame)
            else:
                self.frames[safe_index] = None
    
    def get_frame(self, index):
        with self.lock:
            safe_index = index % self.size
            if safe_index in self.frames and self.frames[safe_index] is not None:
                return np.copy(self.frames[safe_index])
            return None
    
    def clear(self):
        with self.lock:
            self.frames.clear()
            gc.collect()


class SharedData:
    def __init__(self):
        self.camera_params = {}
        self.lock = threading.RLock()


class CameraThread(threading.Thread):
    def __init__(self, shared_data, frame_queue, timestamp_queue, gui_ref):
        super().__init__()
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.gui_ref = gui_ref
        self.dcam = None
        self.running = True
        self.capturing = False
        self.frame_index = 0
        self.modified_params = {}
        self.start_time = None
        self.paused = threading.Event()
        self.paused.set()
        self.first_run = True
        self.first_frame = True
        self.buffer_size = 200
        self.safe_buffer = SafeFrameBuffer(self.buffer_size)
        self.capture_lock = threading.RLock()
        self.save_queue = None
        self.needs_reconnect = False
        
        # Command queue for thread-safe property changes
        self.command_queue = queue.Queue(maxsize=100)
        self.command_response_queue = queue.Queue(maxsize=100)
        
        # Frame processing lock
        self.frame_processing_lock = threading.Lock()
        
        # Performance monitoring
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_print_time = time.time()
        
        # Watchdog for capture health
        self.last_capture_time = time.time()
        self.watchdog_enabled = False
        self.watchdog_reset_count = 0
        self.max_watchdog_resets = 3
        
        # Set thread name for debugging
        self.name = "CameraThread"

    def run(self):
        try:
            # Run the connection process in a separate thread
            threading.Thread(target=self.connect_camera, daemon=True, name="CameraConnect").start()
        except Exception as e:
            logging.error(f"Fatal error in camera thread: {e}")

    @log_dcam_call
    def connect_camera(self):
        retry_count = 0
        while self.running:
            retry_count += 1
            debug_logger.info(f"Camera connection attempt {retry_count}")
            
            # Initialize the API
            init_success = True
            try:
                with DCamLock():
                    ret = Dcamapi.init()
                    if ret is False:
                        err = Dcamapi.lasterr()
                        logging.warning(f"DCAM API init error: {err}")
                        init_success = False
            except Exception as e:
                logging.warning(f"Exception during DCAM init: {e}")
                init_success = False

            if not init_success:
                try:
                    with DCamLock():
                        Dcamapi.uninit()
                except:
                    pass
                if self.gui_ref:
                    self.gui_ref.update_status("Camera not connected.", "red")
                time.sleep(5)
                continue

            # Attempt to open the camera device
            try:
                with DCamLock():
                    cam = Dcam(0)
                    if cam.dev_open() is False:
                        err = cam.lasterr()
                        logging.warning(f"Error opening device: {err}")
                        raise RuntimeError(f"Device open failed: {err}")
                    logging.info("Camera connected successfully")
                    self.dcam = cam
            except Exception as e:
                logging.warning(f"Failed to open camera: {e}")
                if self.gui_ref:
                    self.gui_ref.update_status("Camera not connected.", "red")
                try:
                    with DCamLock():
                        Dcamapi.uninit()
                except:
                    pass
                time.sleep(5)
                continue

            break

        if not self.running or self.dcam is None:
            return

        # Clear needs_reconnect flag on successful connection
        self.needs_reconnect = False
        self.watchdog_reset_count = 0  # Reset watchdog counter

        # Configure camera
        if self.first_run:
            self.set_defaults()
            self.first_run = False
        else:
            self.restore_modified_params()

        self.update_camera_params()
        logging.info("Camera initialized and ready")
        
        # Main camera loop
        while self.running:
            try:
                # Check if reconnection is needed
                if self.needs_reconnect:
                    logging.warning("Camera needs reconnection - pausing operations")
                    if self.gui_ref:
                        self.gui_ref.update_status("Camera needs reset - please use Reset Camera button", "red")
                    time.sleep(1.0)
                    continue
                
                self.process_commands()
                self.paused.wait(timeout=0.001)
                
                if self.capturing:
                    # Check watchdog - if no frames for 10 seconds, try to recover
                    if self.watchdog_enabled and time.time() - self.last_capture_time > 600.0:
                        logging.error("Capture watchdog triggered - no frames for 10 seconds")
                        
                        # Check if we've exceeded max reset attempts
                        if self.watchdog_reset_count >= self.max_watchdog_resets:
                            logging.error(f"Watchdog reset limit reached ({self.max_watchdog_resets})")
                            self.capturing = False
                            self.watchdog_enabled = False
                            self.needs_reconnect = True
                            if self.gui_ref:
                                self.gui_ref.update_status("Camera unresponsive - manual reset required", "red")
                        else:
                            self.watchdog_reset_count += 1
                            logging.info(f"Watchdog reset attempt {self.watchdog_reset_count}/{self.max_watchdog_resets}")
                            self.reset_capture_state()
                            self.last_capture_time = time.time()
                    
                    self.capture_frame()
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"Error in camera loop: {e}")
                time.sleep(0.1)

    def process_commands(self):
        """Process commands from the queue"""
        # Skip command processing if camera needs reconnect
        if self.needs_reconnect:
            # Clear the queue
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                    self.command_response_queue.put({'success': False, 'error': 'Camera needs reconnect'})
                except:
                    pass
            return
        
        processed = 0
        max_commands_per_iteration = 10
        
        while processed < max_commands_per_iteration and not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                processed += 1
                cmd_type = command.get('type')
                
                # Try to acquire lock with timeout
                lock_acquired = False
                lock_start_time = time.time()
                
                while not lock_acquired and time.time() - lock_start_time < 2.0:  # 2 second timeout
                    try:
                        if DCamLock._lock.acquire(blocking=False):
                            lock_acquired = True
                            break
                    except:
                        pass
                    time.sleep(0.001)
                
                if not lock_acquired:
                    logging.error(f"Failed to acquire DCamLock for command {cmd_type}")
                    self.command_response_queue.put({'success': False, 'error': 'Lock timeout'})
                    continue
                
                try:
                    if cmd_type == 'set_property':
                        self._set_property_internal(command['prop_name'], command['value'])
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
                        self.command_response_queue.put({'success': False, 'error': 'Unknown command'})
                finally:
                    DCamLock._lock.release()
                    
            except queue.Empty:
                break
            except Exception as e:
                logging.error(f"Command processing error: {e}")
                try:
                    self.command_response_queue.put({'success': False, 'error': str(e)})
                except:
                    pass

    @log_dcam_call
    def disconnect_camera(self):
        logging.info("Disconnecting camera")
        with self.capture_lock:
            if self.capturing:
                # Try normal stop first, then force if needed
                try:
                    with DCamLock():
                        self._stop_capture_internal(force=False)
                except:
                    logging.error("Normal stop failed, using force stop")
                    self._stop_capture_internal(force=True)
            
            # Clean up DCAM connection
            try:
                with DCamLock():
                    if self.dcam is not None:
                        try:
                            self.dcam.dev_close()
                        except Exception as e:
                            logging.error(f"Error closing device: {e}")
                        self.dcam = None
                    try:
                        Dcamapi.uninit()
                    except Exception as e:
                        logging.error(f"Error uninitializing DCAM: {e}")
            except TimeoutError:
                logging.error("Timeout during camera disconnect - forcing cleanup")
                self.dcam = None

    def reset_camera(self):
        logging.info("Resetting camera")
        self.pause_thread()
        self.needs_reconnect = False  # Clear the flag
        self.disconnect_camera()
        threading.Thread(target=self.connect_camera, daemon=True, name="CameraReset").start()
        self.resume_thread()

    def pause_thread(self):
        debug_logger.info("Pausing camera thread")
        self.paused.clear()

    def resume_thread(self):
        debug_logger.info("Resuming camera thread")
        self.paused.set()

    @log_dcam_call
    def set_defaults(self):
        debug_logger.info("Setting default camera parameters")
        with DCamLock():
            defaults = {
                'READOUT_SPEED': 1.0,
                'EXPOSURE_TIME': 0.1,
                'TRIGGER_SOURCE': 1.0,
                'TRIGGER_MODE': 6.0,
                'OUTPUT_TRIG_KIND_0': 3.0,
                'OUTPUT_TRIG_ACTIVE_0': 1.0,
                'OUTPUT_TRIG_POLARITY_0': 1.0,
                'OUTPUT_TRIG_PERIOD_0': 10,
                'SENSOR_MODE': 18.0,
                'IMAGE_PIXEL_TYPE': 1.0
            }
            for prop, value in defaults.items():
                self._set_property_internal(prop, value)

    def _update_camera_params_internal(self):
        """Internal method to update camera params"""
        if self.dcam is None:
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

    def update_camera_params(self):
        """Direct parameter update without queue delays"""
        def _update():
            with DCamLock():
                self._update_camera_params_internal()
        
        threading.Thread(target=_update, daemon=True, name="UpdateParams").start()

    def stop(self):
        logging.info("Stopping camera thread")
        self.running = False
        self.watchdog_enabled = False
        
        # Stop capture with force if needed
        if self.capturing:
            try:
                self.stop_capture()
            except:
                logging.error("Error during normal stop, using force stop")
                self.capturing = False
                self.watchdog_enabled = False
        
        self.paused.set()
        
        # Clean up DCAM with timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                def cleanup_dcam():
                    with DCamLock():
                        if self.dcam is not None:
                            try:
                                self.dcam.dev_close()
                            except:
                                pass
                            self.dcam = None
                        try:
                            Dcamapi.uninit()
                        except:
                            pass
                
                future = executor.submit(cleanup_dcam)
                future.result(timeout=5.0)  # 5 second timeout for cleanup
        except (FutureTimeoutError, Exception) as e:
            logging.error(f"DCAM cleanup timeout/error: {e}")
            # Force cleanup
            self.dcam = None

    def _set_property_internal(self, prop_name, value):
        """Internal method to set property"""
        if prop_name in CAMERA_PARAMS:
            debug_logger.info(f"Setting {prop_name} = {value}")
            if self.dcam is not None:
                set_success = self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
                if set_success is False:
                    raise Exception(f"Failed to set {prop_name}: {self.dcam.lasterr()}")
                self._update_camera_params_internal()
                self.modified_params[prop_name] = value

    def set_property(self, prop_name, value):
        """Direct property setting without queue delays"""
        def _set():
            with DCamLock():
                self._set_property_internal(prop_name, value)
        
        threading.Thread(target=_set, daemon=True, name="SetProperty").start()

    @log_dcam_call
    def restore_modified_params(self):
        debug_logger.info("Restoring modified parameters")
        if not self.modified_params or self.dcam is None:
            return
            
        try:
            with DCamLock():
                for prop_name, value in self.modified_params.items():
                    if self.dcam is not None:
                        try:
                            self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
                        except Exception as e:
                            debug_logger.error(f"Failed to restore {prop_name}: {e}")
        except TimeoutError:
            logging.error("Timeout restoring parameters")
        except Exception as e:
            logging.error(f"Error restoring parameters: {e}")

    def _start_capture_internal(self):
        """Internal method to start capture"""
        logging.info("Starting capture immediately")
        
        # Check if camera needs reconnection
        if self.needs_reconnect:
            logging.error("Camera needs reconnection - cannot start capture")
            if self.gui_ref:
                self.gui_ref.update_status("Camera needs reset - please use Reset Camera button", "red")
            return
        
        try:
            GPS_time.clear_buffer()
            time.sleep(0.05)
        except Exception as e:
            debug_logger.error(f"GPS buffer clear error: {e}")

        if self.capturing:
            debug_logger.warning("Already capturing, stopping first")
            self._stop_capture_internal()

        if self.dcam is None:
            logging.error("Camera not initialized")
            return
        
        self.safe_buffer.clear()
        
        # Try buffer allocation with timeout
        buffer_allocated = False
        try:
            def alloc_buffer():
                return self.dcam.buf_alloc(self.buffer_size)
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(alloc_buffer)
                buffer_allocated = future.result(timeout=3.0)
        except (FutureTimeoutError, Exception) as e:
            logging.error(f"Buffer allocation timeout/error: {e}")
            buffer_allocated = False

        if buffer_allocated is not False:
            self.capturing = True
            self.frame_index = 0
            self.first_frame = True
            self.frame_count = 0
            self.last_frame_time = time.time()
            self.last_print_time = time.time()
            self.last_capture_time = time.time()
            self.watchdog_enabled = True
            self.watchdog_reset_count = 0  # Reset watchdog counter
            self.consecutive_errors = 0
            
            # Reset timestamp tracking for rollover handling
            self.timestamp_offset = 0
            self.last_raw_timestamp = 0
            self.framestamp_offset = 0
            self.last_raw_framestamp = 0

            try:
                self.dcam.prop_setgetvalue(CAMERA_PARAMS['TIME_STAMP_PRODUCER'], 1)
            except Exception as e:
                logging.error(f"Error setting timestamp producer: {e}")

            # Try to start capture with timeout
            capture_started = False
            try:
                def start_cap():
                    return self.dcam.cap_start()
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(start_cap)
                    capture_started = future.result(timeout=3.0)
            except (FutureTimeoutError, Exception) as e:
                logging.error(f"cap_start timeout/error: {e}")
                capture_started = False

            if capture_started is False:
                logging.error(f"Capture start failed")
                self.capturing = False
                self.watchdog_enabled = False
                try:
                    self.dcam.buf_release()
                except:
                    pass
                self.needs_reconnect = True
            else:
                logging.info("Capture started successfully")
        else:
            logging.error("Buffer allocation failed")
            self.needs_reconnect = True

    def start_capture(self):
        """Direct capture start without queue delays"""
        def _start():
            with self.capture_lock:
                with DCamLock():
                    self._start_capture_internal()
        
        # Start in a thread to avoid blocking GUI
        threading.Thread(target=_start, daemon=True, name="StartCapture").start()

    def _stop_capture_internal(self, force=False):
        """Internal method to stop capture
        
        Args:
            force: If True, skip DCAM API calls that might hang
        """
        debug_logger.info(f"Stopping capture (force={force})")
        self.capturing = False
        self.watchdog_enabled = False
        
        if self.dcam is not None and not force:
            try:
                # Try to stop capture with timeout
                stop_success = False
                start_time = time.time()
                
                # Run cap_stop in a thread with timeout
                def stop_capture_thread():
                    return self.dcam.cap_stop()
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    try:
                        future = executor.submit(stop_capture_thread)
                        stop_success = future.result(timeout=2.0)  # 2 second timeout
                    except (FutureTimeoutError, Exception) as e:
                        logging.error(f"cap_stop timeout/error: {e}")
                        stop_success = False
                
                if stop_success is not False:
                    # Try to release buffer with timeout
                    def release_buffer_thread():
                        return self.dcam.buf_release()
                    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        try:
                            future = executor.submit(release_buffer_thread)
                            future.result(timeout=1.0)  # 1 second timeout
                        except (FutureTimeoutError, Exception) as e:
                            logging.error(f"buf_release timeout/error: {e}")
                
                logging.info("Capture stopped")
            except Exception as e:
                logging.error(f"Error stopping capture: {e}")
        elif force:
            logging.warning("Force stop - skipping DCAM API calls")
        
        try:
            self.restore_modified_params()
        except Exception as e:
            logging.error(f"Error restoring params: {e}")

    def stop_capture(self):
        """Direct capture stop without queue delays"""
        def _stop():
            with self.capture_lock:
                try:
                    # Try normal stop with DCamLock
                    lock_acquired = False
                    lock_start_time = time.time()
                    
                    while not lock_acquired and time.time() - lock_start_time < 5.0:
                        try:
                            if DCamLock._lock.acquire(blocking=False):
                                lock_acquired = True
                                break
                        except:
                            pass
                        time.sleep(0.01)
                    
                    if lock_acquired:
                        try:
                            self._stop_capture_internal(force=False)
                        finally:
                            DCamLock._lock.release()
                    else:
                        logging.error("Failed to acquire DCamLock for stop - using force stop")
                        self._stop_capture_internal(force=True)
                except Exception as e:
                    logging.error(f"Error during stop_capture: {e}")
                    # Force stop on any error
                    self._stop_capture_internal(force=True)
        
        threading.Thread(target=_stop, daemon=True, name="StopCapture").start()

    def capture_frame(self):
        timeout_milisec = 1000
        try:
            # Use trylock to avoid blocking
            if not self.frame_processing_lock.acquire(blocking=False):
                return
                
            try:
                # Add watchdog for DCamLock acquisition
                lock_acquired = False
                lock_start_time = time.time()
                
                while not lock_acquired and time.time() - lock_start_time < 5.0:  # 5 second timeout
                    try:
                        if DCamLock._lock.acquire(blocking=False):
                            lock_acquired = True
                            break
                    except:
                        pass
                    time.sleep(0.001)
                
                if not lock_acquired:
                    logging.error("Failed to acquire DCamLock in capture_frame - possible deadlock")
                    # Try to recover by releasing and reacquiring
                    self.frame_processing_lock.release()
                    self.reset_capture_state()
                    return
                
                try:
                    if self.dcam is None or not self.capturing:
                        return
                        
                    if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                        frame_index_safe = self.frame_index % self.buffer_size
                        result = self.dcam.buf_getframe_with_timestamp_and_framestamp(frame_index_safe)
                        
                        if result is not False:
                            frame, npBuf, timestamp, framestamp = result
                            
                            # Handle timestamp rollover (happens around 4294 seconds)
                            raw_timestamp = timestamp.sec + timestamp.microsec / 1e6
                            
                            # Initialize rollover tracking
                            if not hasattr(self, 'timestamp_offset'):
                                self.timestamp_offset = 0
                                self.last_raw_timestamp = 0
                                self.framestamp_offset = 0
                                self.last_raw_framestamp = 0
                            
                            # Detect timestamp rollover (timestamp suddenly drops by a large amount)
                            if raw_timestamp < self.last_raw_timestamp - 4000:  # Rollover detected
                                self.timestamp_offset += 4294.967296  # Add 2^32 microseconds
                                logging.warning(f"Timestamp rollover detected at frame {self.frame_index}: "
                                              f"raw={raw_timestamp:.6f}, corrected={raw_timestamp + self.timestamp_offset:.6f}")
                            
                            self.last_raw_timestamp = raw_timestamp
                            corrected_timestamp = raw_timestamp + self.timestamp_offset
                            
                            # Handle framestamp rollover (happens at 65536 - 16-bit limit)
                            raw_framestamp = framestamp
                            
                            # Detect framestamp rollover (framestamp suddenly drops by a large amount)
                            if raw_framestamp < self.last_raw_framestamp - 60000:  # Rollover detected
                                self.framestamp_offset += 65536  # Add 2^16
                                logging.warning(f"Framestamp rollover detected at frame {self.frame_index}")
                            
                            self.last_raw_framestamp = raw_framestamp
                            corrected_framestamp = raw_framestamp + self.framestamp_offset
                            
                            # Print frame info periodically (every second)
                            current_time = time.time()
                            if current_time - self.last_print_time > 1.0:
                                logging.info(f"Frame {self.frame_index}: timestamp={corrected_timestamp:.6f}, framestamp={corrected_framestamp}")
                                self.last_print_time = current_time
                            
                            # Update watchdog
                            self.last_capture_time = current_time
                            # Reset watchdog counter on successful frame
                            if self.watchdog_reset_count > 0:
                                logging.info("Capture recovered - resetting watchdog counter")
                                self.watchdog_reset_count = 0
                            
                            # Monitor frame rate
                            self.frame_count += 1
                            if current_time - self.last_frame_time > 10:
                                fps = self.frame_count / (current_time - self.last_frame_time)
                                logging.info(f"FPS: {fps:.2f}")
                                self.frame_count = 0
                                self.last_frame_time = current_time
                            
                            frame_copy = np.copy(npBuf)
                            self.safe_buffer.add_frame(self.frame_index, frame_copy)
                            
                            # Non-blocking queue operations
                            try:
                                if self.frame_queue.full():
                                    self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame_copy)
                            except:
                                pass

                            try:
                                self.timestamp_queue.put_nowait((corrected_timestamp, corrected_framestamp))
                            except queue.Full:
                                pass
                            
                            if self.save_queue is not None:
                                try:
                                    self.save_queue.put_nowait((frame_copy, corrected_timestamp, corrected_framestamp))
                                except queue.Full:
                                    debug_logger.warning("Save queue full")

                            # Fetch the GPS timestamp only once at the start - with timeout protection
                            if self.first_frame:
                                # Run GPS fetch in separate thread with timeout
                                def fetch_gps():
                                    try:
                                        return GPS_time.get_first_timestamp()
                                    except:
                                        return None
                                
                                with ThreadPoolExecutor(max_workers=1) as executor:
                                    try:
                                        future = executor.submit(fetch_gps)
                                        self.start_time = future.result(timeout=1.0)  # 1 second timeout
                                        if self.start_time:
                                            logging.info(f"GPS timestamp acquired: {self.start_time.isot}")
                                            if self.gui_ref:
                                                self.gui_ref.update_gps_timestamp(self.start_time.isot)
                                        else:
                                            logging.warning("No GPS timestamp available")
                                            if self.gui_ref:
                                                self.gui_ref.update_gps_timestamp("No GPS timestamp")
                                    except (FutureTimeoutError, Exception) as e:
                                        logging.error(f"GPS timestamp error: {e}")
                                        self.start_time = None
                                        if self.gui_ref:
                                            self.gui_ref.update_gps_timestamp("GPS timeout")
                                
                                self.first_frame = False

                            self.frame_index += 1
                        else:
                            dcamerr = self.dcam.lasterr()
                            if not dcamerr.is_timeout():
                                debug_logger.error(f'Frame capture error: {dcamerr}')
                                # On persistent errors, try to recover
                                if hasattr(self, 'consecutive_errors'):
                                    self.consecutive_errors += 1
                                    if self.consecutive_errors > 10:
                                        logging.error("Too many consecutive capture errors, attempting recovery")
                                        self.reset_capture_state()
                                        self.consecutive_errors = 0
                                else:
                                    self.consecutive_errors = 1
                            else:
                                self.consecutive_errors = 0  # Reset on successful timeout
                finally:
                    DCamLock._lock.release()
                    
            finally:
                self.frame_processing_lock.release()
                
        except Exception as e:
            logging.error(f"Capture frame error: {e}")
            debug_logger.error(traceback.format_exc())
            # Try to recover from errors
            try:
                if self.frame_processing_lock.locked():
                    self.frame_processing_lock.release()
            except:
                pass
            time.sleep(0.1)
    
    def reset_capture_state(self):
        """Reset capture state to recover from errors"""
        logging.warning("Resetting capture state")
        try:
            # Stop and restart capture with proper locking
            was_capturing = self.capturing
            if was_capturing:
                # Use non-blocking lock acquisition
                lock_acquired = False
                lock_start_time = time.time()
                
                while not lock_acquired and time.time() - lock_start_time < 5.0:
                    try:
                        if self.capture_lock.acquire(blocking=False):
                            lock_acquired = True
                            break
                    except:
                        pass
                    time.sleep(0.01)
                
                if lock_acquired:
                    try:
                        # Try to acquire DCamLock
                        dcam_lock_acquired = False
                        dcam_lock_start = time.time()
                        
                        while not dcam_lock_acquired and time.time() - dcam_lock_start < 5.0:
                            try:
                                if DCamLock._lock.acquire(blocking=False):
                                    dcam_lock_acquired = True
                                    break
                            except:
                                pass
                            time.sleep(0.01)
                        
                        if dcam_lock_acquired:
                            try:
                                # First try normal stop
                                self._stop_capture_internal(force=False)
                                time.sleep(0.5)
                                
                                # Check if camera is still responsive
                                camera_responsive = False
                                try:
                                    # Test if camera is responsive with a simple property read
                                    def test_camera():
                                        if self.dcam:
                                            return self.dcam.prop_getvalue(CAMERA_PARAMS['EXPOSURE_TIME'])
                                        return False
                                    
                                    with ThreadPoolExecutor(max_workers=1) as executor:
                                        future = executor.submit(test_camera)
                                        result = future.result(timeout=1.0)
                                        camera_responsive = (result is not False)
                                except:
                                    camera_responsive = False
                                
                                if camera_responsive:
                                    self._start_capture_internal()
                                    if self.capturing:
                                        logging.info("Capture state reset successfully")
                                        # Don't reset watchdog counter here - let successful frame capture do it
                                    else:
                                        logging.error("Failed to restart capture after reset")
                                        self.needs_reconnect = True
                                else:
                                    logging.error("Camera unresponsive - attempting reconnection")
                                    # Force stop and mark camera for reconnection
                                    self._stop_capture_internal(force=True)
                                    self.needs_reconnect = True
                                    if self.gui_ref:
                                        self.gui_ref.update_status("Camera unresponsive - please reset", "red")
                            finally:
                                DCamLock._lock.release()
                        else:
                            logging.error("Failed to acquire DCamLock for reset - forcing stop")
                            # Force stop without lock
                            self.capturing = False
                            self.watchdog_enabled = False
                            self.needs_reconnect = True
                    finally:
                        self.capture_lock.release()
                else:
                    logging.error("Failed to acquire capture_lock for reset - forcing stop")
                    # Force stop without lock
                    self.capturing = False
                    self.watchdog_enabled = False
                    self.needs_reconnect = True
        except Exception as e:
            logging.error(f"Failed to reset capture state: {e}")
            # Last resort - force everything off
            self.capturing = False
            self.watchdog_enabled = False
            self.needs_reconnect = True


class SaveThread(threading.Thread):
    def __init__(self, save_queue, timestamp_queue, camera_thread, object_name, shared_data):
        super().__init__()
        self.save_queue = save_queue
        self.timestamp_queue = timestamp_queue
        self.running = True
        self.camera_thread = camera_thread
        self.object_name = object_name
        self.batch_size = 100
        self.frame_buffer = []
        self.timestamp_buffer = []
        self.framestamp_buffer = []
        self.cube_index = 0
        self.shared_data = shared_data
        self.name = "SaveThread"

    def run(self):
        try:
            logging.info("Save thread started")
            if not self.running:
                return

            start_time = self.camera_thread.start_time
            current_time = time.localtime()
            start_time_filename_str = time.strftime('%Y%m%d_%H%M%S', current_time)

            os.makedirs("captures", exist_ok=True)

            while self.running or not self.save_queue.empty():
                try:
                    # Note: timestamps and framestamps are already corrected for rollovers
                    frame, timestamp, framestamp = self.save_queue.get(timeout=1)
                    self.frame_buffer.append(frame)
                    self.timestamp_buffer.append(timestamp)
                    self.framestamp_buffer.append(framestamp)

                    if len(self.frame_buffer) >= self.batch_size:
                        self.write_cube_to_disk(start_time_filename_str)
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Save thread error: {e}")

            if self.frame_buffer:
                self.write_cube_to_disk(start_time_filename_str)
                
        except Exception as e:
            logging.error(f"Fatal save thread error: {e}")

    @log_dcam_call
    def write_cube_to_disk(self, start_time_filename_str):
        try:
            self.cube_index += 1
            filename = f"{self.object_name}_{start_time_filename_str}_cube{self.cube_index:03d}.fits"
            filepath = os.path.join("captures", filename)
            
            logging.info(f"Writing cube {self.cube_index} ({len(self.frame_buffer)} frames)")

            primary_hdu = fits.PrimaryHDU()
            primary_hdu.header['OBJECT'] = (self.object_name, 'Object name')
            primary_hdu.header['CUBEIDX'] = (self.cube_index, 'Cube index number')
            primary_hdu.header['COMMENT'] = 'Timestamps and framestamps corrected for rollovers'
            primary_hdu.header['COMMENT'] = 'Timestamp rollover at 4294.967296 seconds (32-bit)'
            primary_hdu.header['COMMENT'] = 'Framestamp rollover at 65536 frames (16-bit)'

            data_cube = np.stack(self.frame_buffer, axis=0)
            image_hdu = fits.ImageHDU(data=data_cube)
            image_hdu.header['EXTNAME'] = 'DATA_CUBE'
            
            with self.shared_data.lock:
                for key, value in self.shared_data.camera_params.items():
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        try:
                            image_hdu.header[key] = value
                        except:
                            pass

            col1 = fits.Column(name='TIMESTAMP', format='D', array=self.timestamp_buffer)
            col2 = fits.Column(name='FRAMESTAMP', format='K', array=self.framestamp_buffer)
            cols = fits.ColDefs([col1, col2])
            timestamp_hdu = fits.BinTableHDU.from_columns(cols)
            timestamp_hdu.header['EXTNAME'] = 'TIMESTAMPS'

            hdulist = fits.HDUList([primary_hdu, image_hdu, timestamp_hdu])
            hdulist.writeto(filepath, overwrite=True)
            hdulist.close()

            logging.info(f"Saved: {filepath}")

            self.frame_buffer.clear()
            self.timestamp_buffer.clear()
            self.framestamp_buffer.clear()
            
        except Exception as e:
            logging.error(f"Write cube error: {e}")

    def stop(self):
        self.running = False


class PeripheralsThread(threading.Thread):
    def __init__(self, shared_data, frame_queue, timestamp_queue, pdu_ip,
                 xmcc1_port, xmcc2_port, gui_ref):
        super().__init__()
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.gui_ref = gui_ref
        self.efw = None
        self.pdu_ip = pdu_ip
        self.pdu = None
        self.ljm_handle = None
        self.xmcc1_port = xmcc1_port
        self.xmcc2_port = xmcc2_port
        self.ax_a_1 = None
        self.ax_a_2 = None
        self.ax_b_1 = None
        self.ax_b_2 = None
        self.ax_b_3 = None
        self.peripherals_lock = threading.RLock()
        self.name = "PeripheralsThread"
        self.connection_a = None
        self.connection_b = None
        self.xmcc_a = None
        self.xmcc_b = None

    def run(self):
        try:
            threading.Thread(target=self.thread_target, daemon=True, name="PeripheralsConnect").start()
        except Exception as e:
            logging.error(f"Peripherals thread error: {e}")

    def thread_target(self):
        asyncio.run(self.connect_peripherals())

    async def connect_peripherals(self):
        with self.peripherals_lock:
            self.connect_efw()
            self.connect_zaber_axes()
            logging.info("Connecting to PDU")
            try:
                self.pdu = CyberPowerPDU(ip_address=self.pdu_ip, simulate=False)
                await self.pdu.initialize()
                logging.info("PDU connected")
            except Exception as e:
                logging.warning(f"PDU connection failed: {e}")
            self.connect_labjack()

    def connect_labjack(self):
        try:
            with self.peripherals_lock:
                self.ljm_handle = ljm.openS("T4", "ANY", "ANY")
                logging.info("LabJack connected")
        except Exception as e:
            logging.warning(f"LabJack connection failed: {e}")

    def connect_efw(self):
        debug_logger.info("Connecting to filter wheel")
        try:
            with self.peripherals_lock:
                self.efw = EFW(verbose=False)
                self.efw.GetPosition(0)
                logging.info("Filter wheel connected")
        except Exception as e:
            logging.warning(f"Filter wheel connection failed: {e}")
            self.efw = None

    def connect_zaber_axes(self):
        debug_logger.info("Connecting to Zaber devices")
        
        XMCC1_SERIAL = 137816
        XMCC2_SERIAL = 137819
        
        ports_to_try = [self.xmcc1_port, self.xmcc2_port]
        devices_found = {}
        
        for port in ports_to_try:
            try:
                with self.peripherals_lock:
                    connection = Connection.open_serial_port(port)
                    connection.enable_alerts()
                    devices = connection.detect_devices()
                    
                    if devices:
                        device = devices[0]
                        serial_number = device.serial_number
                        debug_logger.info(f"Found device SN {serial_number} on {port}")
                        
                        if serial_number == XMCC1_SERIAL:
                            devices_found['xmcc1'] = {
                                'connection': connection,
                                'device': device,
                                'port': port
                            }
                            logging.info(f"X-MCC1 connected on {port}")
                        elif serial_number == XMCC2_SERIAL:
                            devices_found['xmcc2'] = {
                                'connection': connection,
                                'device': device,
                                'port': port
                            }
                            logging.info(f"X-MCC2 connected on {port}")
                        else:
                            debug_logger.warning(f"Unknown device SN {serial_number}")
                            connection.close()
                    else:
                        connection.close()
                        
            except Exception as e:
                debug_logger.warning(f"Port {port} error: {e}")
        
        with self.peripherals_lock:
            if 'xmcc1' in devices_found:
                self.connection_a = devices_found['xmcc1']['connection']
                self.xmcc_a = devices_found['xmcc1']['device']
                
                try:
                    self.ax_a_1 = self.xmcc_a.get_axis(1)
                    self.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
                    debug_logger.info("Slit stage ready")
                except Exception as e:
                    debug_logger.error(f"Slit stage init error: {e}")
                    self.ax_a_1 = None
                
                try:
                    self.ax_a_2 = self.xmcc_a.get_axis(2)
                    self.ax_a_2.get_position(Units.ANGLE_DEGREES)
                    debug_logger.info("Zoom stepper ready")
                except Exception as e:
                    debug_logger.error(f"Zoom stepper init error: {e}")
                    self.ax_a_2 = None
            else:
                self.connection_a = None
                self.xmcc_a = None
                self.ax_a_1 = None
                self.ax_a_2 = None
                logging.warning("X-MCC1 not found")
            
            if 'xmcc2' in devices_found:
                self.connection_b = devices_found['xmcc2']['connection']
                self.xmcc_b = devices_found['xmcc2']['device']
                
                try:
                    self.ax_b_1 = self.xmcc_b.get_axis(1)
                    self.ax_b_1.get_position(Units.ANGLE_DEGREES)
                    debug_logger.info("Focus stepper ready")
                except Exception as e:
                    debug_logger.error(f"Focus stepper init error: {e}")
                    self.ax_b_1 = None
                
                try:
                    self.ax_b_2 = self.xmcc_b.get_axis(2)
                    self.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
                    debug_logger.info("Polarization stage ready")
                except Exception as e:
                    debug_logger.error(f"Polarization stage init error: {e}")
                    self.ax_b_2 = None
                
                try:
                    self.ax_b_3 = self.xmcc_b.get_axis(3)
                    self.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
                    debug_logger.info("Halpha/QWP stage ready")
                except Exception as e:
                    debug_logger.error(f"Halpha/QWP stage init error: {e}")
                    self.ax_b_3 = None
            else:
                self.connection_b = None
                self.xmcc_b = None
                self.ax_b_1 = None
                self.ax_b_2 = None
                self.ax_b_3 = None
                logging.warning("X-MCC2 not found")

    def disconnect_peripherals(self):
        logging.info("Disconnecting peripherals")
        with self.peripherals_lock:
            if self.efw is not None:
                try:
                    self.efw.Close(0)
                    debug_logger.info("Filter wheel disconnected")
                except:
                    pass

            if self.pdu is not None:
                try:
                    asyncio.run(self.pdu.close())
                    debug_logger.info("PDU disconnected")
                except:
                    pass

            if self.ljm_handle is not None:
                ljm.close(self.ljm_handle)
                debug_logger.info("LabJack disconnected")

            if self.connection_a is not None:
                try:
                    self.connection_a.close()
                    debug_logger.info("X-MCC1 disconnected")
                except:
                    pass
            
            if self.connection_b is not None:
                try:
                    self.connection_b.close()
                    debug_logger.info("X-MCC2 disconnected")
                except:
                    pass

    def command_outlet(self, outlet, command):
        with self.peripherals_lock:
            if self.pdu is not None:
                try:
                    asyncio.run(self.pdu.send_outlet_command(outlet, command))
                    debug_logger.info(f"Outlet {outlet} command: {command}")
                except Exception as e:
                    logging.error(f"Outlet command error: {e}")

    async def get_all_outlet_states(self):
        with self.peripherals_lock:
            if self.pdu is not None:
                try:
                    states = await self.pdu.get_all_outlet_states()
                    return states
                except Exception as e:
                    logging.error(f"Get outlet states error: {e}")
                    return {}
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
        self.updating_camera_status = True
        self.updating_frame_display = True
        self.updating_peripherals_status = True
        self.display_lock = threading.Lock()
        self.save_thread = None
        self.last_frame = None

        self.min_val = tk.StringVar(value="0")
        self.max_val = tk.StringVar(value="200")

        self.title("Lightspeed Prototype Control GUI")
        self.geometry("1000x1050")  # Slightly taller for GPS display

        self.setup_gui()
        
        # Initialize peripheral update flag
        self._peripheral_update_running = False

    def setup_gui(self):
        """Set up all GUI elements"""
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # Camera Parameters
        camera_params_frame = LabelFrame(self.main_frame, text="Camera Parameters", padx=5, pady=5)
        camera_params_frame.grid(row=0, column=0, rowspan=10, sticky='nsew')
        self.camera_status = tk.Label(camera_params_frame, text="", justify=tk.LEFT, anchor="w", 
                                      width=60, height=80, wraplength=400)
        self.camera_status.pack(fill='both', expand=True)

        # Status messages
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="w", 
                                       width=40, wraplength=400, fg="blue")
        self.status_message.grid(row=5, column=0, columnspan=2, sticky='nsew')

        # GPS Timestamp display
        gps_frame = LabelFrame(self.main_frame, text="GPS Timestamp", padx=5, pady=5)
        gps_frame.grid(row=6, column=0, columnspan=2, sticky='ew')
        self.gps_timestamp_label = tk.Label(gps_frame, text="No capture active", font=("Courier", 12))
        self.gps_timestamp_label.pack()

        # Camera Controls
        self.setup_camera_controls()
        
        # Camera Settings
        self.setup_camera_settings()
        
        # Subarray Controls
        self.setup_subarray_controls()
        
        # Advanced Controls
        self.setup_advanced_controls()
        
        # Display Controls
        self.setup_display_controls()
        
        # Peripherals Controls
        self.setup_peripherals_controls()
        
        # Start update loops with staggered timing
        self.after(100, self.update_camera_status)
        self.after(200, self.update_frame_display)
        self.after(10000, self.update_peripherals_status)  # Start peripheral updates after 10 seconds

    def update_status(self, message, color="blue"):
        """Thread-safe status update"""
        self.after(0, lambda: self.status_message.config(text=message, fg=color))

    def update_gps_timestamp(self, timestamp_str):
        """Thread-safe GPS timestamp update"""
        self.after(0, lambda: self.gps_timestamp_label.config(text=timestamp_str))

    def setup_camera_controls(self):
        camera_controls_frame = LabelFrame(self.main_frame, text="Camera Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1, sticky='n')

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=0, column=0)
        self.exposure_time_var = tk.DoubleVar()
        self.exposure_time_var.set(100)
        self.exposure_time_var.trace_add("write", self.update_exposure_time)
        self.exposure_time_entry = Entry(camera_controls_frame, textvariable=self.exposure_time_var)
        self.exposure_time_entry.grid(row=0, column=1)

        self.save_data_var = tk.BooleanVar()
        self.save_data_checkbox = Checkbutton(camera_controls_frame, text="Save Data to Disk", 
                                              variable=self.save_data_var)
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

        self.power_cycle_button = Button(camera_controls_frame, text="Power Cycle Camera",
                                         command=self.power_cycle_camera)
        self.power_cycle_button.grid(row=6, column=0, columnspan=2)

    def setup_camera_settings(self):
        camera_settings_frame = LabelFrame(self.main_frame, text="Camera Settings", padx=5, pady=5)
        camera_settings_frame.grid(row=1, column=1, sticky='n')

        Label(camera_settings_frame, text="Binning:").grid(row=0, column=0)
        self.binning_var = StringVar(self)
        self.binning_var.set("1x1")
        self.binning_menu = OptionMenu(camera_settings_frame, self.binning_var, 
                                       "1x1", "2x2", "4x4", command=self.change_binning)
        self.binning_menu.grid(row=0, column=1)

        Label(camera_settings_frame, text="Bit Depth:").grid(row=1, column=0)
        self.bit_depth_var = StringVar(self)
        self.bit_depth_var.set("8-bit")
        self.bit_depth_menu = OptionMenu(camera_settings_frame, self.bit_depth_var, 
                                         "8-bit", "16-bit", command=self.change_bit_depth)
        self.bit_depth_menu.grid(row=1, column=1)

        Label(camera_settings_frame, text="Readout Speed:").grid(row=2, column=0)
        self.readout_speed_var = StringVar(self)
        self.readout_speed_var.set("Ultra Quiet Mode")
        self.readout_speed_menu = OptionMenu(camera_settings_frame, self.readout_speed_var, 
                                             "Ultra Quiet Mode", "Standard Mode", 
                                             command=self.change_readout_speed)
        self.readout_speed_menu.grid(row=2, column=1)

        Label(camera_settings_frame, text="Sensor Mode:").grid(row=3, column=0)
        self.sensor_mode_var = StringVar(self)
        self.sensor_mode_var.set("Photon Number Resolving")
        self.sensor_mode_menu = OptionMenu(camera_settings_frame, self.sensor_mode_var, 
                                           "Photon Number Resolving", "Standard", 
                                           command=self.change_sensor_mode)
        self.sensor_mode_menu.grid(row=3, column=1)

    def setup_subarray_controls(self):
        subarray_controls_frame = LabelFrame(self.main_frame, text="Subarray Controls", padx=5, pady=5)
        subarray_controls_frame.grid(row=2, column=1, sticky='n')

        Label(subarray_controls_frame, text="Subarray Mode:").grid(row=0, column=0)
        self.subarray_mode_var = StringVar(self)
        self.subarray_mode_var.set("Off")
        self.subarray_mode_menu = OptionMenu(subarray_controls_frame, self.subarray_mode_var, 
                                             "Off", "On", command=self.change_subarray_mode)
        self.subarray_mode_menu.grid(row=0, column=1)

        Label(subarray_controls_frame, text="Subarray HPOS:").grid(row=1, column=0)
        self.subarray_hpos_var = tk.IntVar()
        self.subarray_hpos_var.set(0)
        self.subarray_hpos_var.trace_add("write", self.update_subarray)
        self.subarray_hpos_entry = Entry(subarray_controls_frame, textvariable=self.subarray_hpos_var)
        self.subarray_hpos_entry.grid(row=1, column=1)
        self.subarray_hpos_entry.config(state='disabled')

        Label(subarray_controls_frame, text="Subarray HSIZE:").grid(row=2, column=0)
        self.subarray_hsize_var = tk.IntVar()
        self.subarray_hsize_var.set(4096)
        self.subarray_hsize_var.trace_add("write", self.update_subarray)
        self.subarray_hsize_entry = Entry(subarray_controls_frame, textvariable=self.subarray_hsize_var)
        self.subarray_hsize_entry.grid(row=2, column=1)
        self.subarray_hsize_entry.config(state='disabled')

        Label(subarray_controls_frame, text="Subarray VPOS:").grid(row=3, column=0)
        self.subarray_vpos_var = tk.IntVar()
        self.subarray_vpos_var.set(0)
        self.subarray_vpos_var.trace_add("write", self.update_subarray)
        self.subarray_vpos_entry = Entry(subarray_controls_frame, textvariable=self.subarray_vpos_var)
        self.subarray_vpos_entry.grid(row=3, column=1)
        self.subarray_vpos_entry.config(state='disabled')

        Label(subarray_controls_frame, text="Subarray VSIZE:").grid(row=4, column=0)
        self.subarray_vsize_var = tk.IntVar()
        self.subarray_vsize_var.set(2304)
        self.subarray_vsize_var.trace_add("write", self.update_subarray)
        self.subarray_vsize_entry = Entry(subarray_controls_frame, textvariable=self.subarray_vsize_var)
        self.subarray_vsize_entry.grid(row=4, column=1)
        self.subarray_vsize_entry.config(state='disabled')

        Label(subarray_controls_frame, text="Note: Values will be rounded to nearest factor of 4.").grid(
            row=5, column=0, columnspan=2)

    def setup_advanced_controls(self):
        advanced_controls_frame = LabelFrame(self.main_frame, text="Advanced Controls", padx=5, pady=5)
        advanced_controls_frame.grid(row=3, column=1, sticky='n')

        self.framebundle_var = tk.BooleanVar()
        self.framebundle_checkbox = Checkbutton(advanced_controls_frame, text="Enable Frame Bundle", 
                                                variable=self.framebundle_var, 
                                                command=self.update_framebundle)
        self.framebundle_checkbox.grid(row=0, column=0, columnspan=2)

        Label(advanced_controls_frame, text="Frames Per Bundle:").grid(row=1, column=0)
        self.frames_per_bundle_var = tk.IntVar()
        self.frames_per_bundle_var.set(100)
        self.frames_per_bundle_var.trace_add("write", self.update_frames_per_bundle)
        self.frames_per_bundle_entry = Entry(advanced_controls_frame, 
                                             textvariable=self.frames_per_bundle_var)
        self.frames_per_bundle_entry.grid(row=1, column=1)
        
        Label(advanced_controls_frame, 
              text="When enabled, this many frames will \nbe concatenated into one image.").grid(
              row=2, column=0, columnspan=2)

    def setup_display_controls(self):
        display_controls_frame = LabelFrame(self.main_frame, text="Display Controls", padx=5, pady=5)
        display_controls_frame.grid(row=4, column=1, sticky='n')

        Label(display_controls_frame, text="Min Count:").grid(row=0, column=0)
        self.min_entry = Entry(display_controls_frame, textvariable=self.min_val)
        self.min_entry.grid(row=0, column=1)

        Label(display_controls_frame, text="Max Count:").grid(row=0, column=2)
        self.max_entry = Entry(display_controls_frame, textvariable=self.max_val)
        self.max_val.trace_add("write", self.refresh_frame_display)
        self.max_entry.grid(row=0, column=3)

    def setup_peripherals_controls(self):
        self.peripherals_controls_frame = LabelFrame(self.main_frame, text="Peripherals Controls", 
                                                     padx=5, pady=5)
        self.peripherals_controls_frame.grid(row=7, column=1, sticky='n')

        # Filter control
        Label(self.peripherals_controls_frame, text="Filter:").grid(row=0, column=0)
        self.filter_position_var = tk.StringVar()
        self.filter_options = {'0 (Open)': 0, '1 (u\')': 1, '2 (g\')': 2, '3 (r\')': 3,
                               '4 (i\')': 4, '5 (z\')': 5}
        self.filter_position_menu = OptionMenu(self.peripherals_controls_frame, self.filter_position_var,
                                               *self.filter_options.keys(),
                                               command=self.update_filter_position)
        self.filter_position_menu.grid(row=0, column=1)

        # Shutter control
        Label(self.peripherals_controls_frame, text="Shutter:").grid(row=0, column=2)
        self.shutter_var = tk.StringVar()
        self.shutter_var.set('Open')
        self.shutter_options = ['Open', 'Closed']
        self.shutter_menu = OptionMenu(self.peripherals_controls_frame, self.shutter_var,
                                       *self.shutter_options, command=self.update_shutter)
        self.shutter_menu.grid(row=0, column=3)

        # Slit control
        Label(self.peripherals_controls_frame, text="Slit:").grid(row=1, column=0)
        self.slit_position_var = tk.StringVar()
        self.slit_position_var.set('Out of beam')
        self.slit_options = ['In beam', 'Out of beam']
        self.slit_position_menu = OptionMenu(self.peripherals_controls_frame, self.slit_position_var,
                                             *self.slit_options, command=self.update_slit_position)
        self.slit_position_menu.grid(row=1, column=1)

        # Halpha/QWP control
        Label(self.peripherals_controls_frame, text="Halpha/QWP:").grid(row=1, column=2)
        self.halpha_qwp_var = tk.StringVar()
        self.halpha_qwp_var.set('Neither')
        self.halpha_qwp_options = ['Halpha', 'QWP', 'Neither']
        self.halpha_qwp_menu = OptionMenu(self.peripherals_controls_frame, self.halpha_qwp_var,
                                          *self.halpha_qwp_options, command=self.update_halpha_qwp)
        self.halpha_qwp_menu.grid(row=1, column=3)

        # Polarization stage control
        Label(self.peripherals_controls_frame, text="Pol. Stage:").grid(row=2, column=0)
        self.wire_grid_var = tk.StringVar()
        self.wire_grid_var.set('Neither')
        self.wire_grid_options = ['WeDoWo', 'Wire Grid', 'Neither']
        self.wire_grid_menu = OptionMenu(self.peripherals_controls_frame, self.wire_grid_var,
                                         *self.wire_grid_options, command=self.update_pol_stage)
        self.wire_grid_menu.grid(row=2, column=1)

        # Zoom control
        Label(self.peripherals_controls_frame, text="Zoom-out:").grid(row=2, column=2)
        self.zoom_stepper_var = tk.StringVar()
        self.zoom_stepper_var.set('4x')
        self.zoom_stepper_options = ['1x', '2x', '3x', '4x']
        self.zoom_stepper_menu = OptionMenu(self.peripherals_controls_frame, self.zoom_stepper_var,
                                            *self.zoom_stepper_options, command=self.update_zoom_stepper)
        self.zoom_stepper_menu.grid(row=2, column=3)

        # Focus control
        Label(self.peripherals_controls_frame, text="Focus Position (um):").grid(row=3, column=0, columnspan=2)
        self.focus_position_var = tk.StringVar()
        self.focus_position_var.set('0')
        self.focus_conversion_factor = 1
        self.focus_position_entry = Entry(self.peripherals_controls_frame, 
                                          textvariable=self.focus_position_var)
        self.focus_position_entry.grid(row=3, column=2)
        self.set_focus_button = Button(self.peripherals_controls_frame, text="Set Focus",
                                       command=self.update_focus_position)
        self.set_focus_button.grid(row=3, column=3)

        # PDU outlet controls
        Label(self.peripherals_controls_frame, text="PDU Outlet States").grid(row=4, column=0, columnspan=4)
        self.pdu_outlet_dict = {1: 'Rotator', 2: 'Switch', 3: 'Shutter', 4: 'Empty',
                                5: 'Empty', 6: 'Empty', 7: 'Empty', 8: 'Empty',
                                9: 'Ctrl PC', 10: 'X-MCC4 A', 11: 'X-MCC B', 12: 'qCMOS',
                                13: 'Empty', 14: 'Empty', 15: 'Empty', 16: 'Empty'}
        self.pdu_outlet_vars = {}
        self.pdu_outlet_buttons = {}
        for idx, name in self.pdu_outlet_dict.items():
            row = (idx - 1) % 8 + 8
            col = (idx - 1) // 8 * 2
            name = str(idx) + ': ' + name
            tk.Label(self.peripherals_controls_frame, text=name, width=12, anchor='w')\
              .grid(row=row, column=col, padx=2, pady=2)
            var = tk.BooleanVar(value=True)
            self.pdu_outlet_vars[idx] = var
            btn = tk.Checkbutton(self.peripherals_controls_frame, text='ON', relief='sunken',
                                 fg='green', variable=var, indicatoron=False, width=3,
                                 command=lambda i=idx: self.toggle_outlet(i))
            btn.grid(row=row, column=col + 1, padx=2, pady=2)
            self.pdu_outlet_buttons[idx] = btn

    def update_camera_status(self):
        if self.updating_camera_status:
            # Run status update in background thread
            def _update():
                try:
                    status_text = ""
                    with self.shared_data.lock:
                        for key, value in self.shared_data.camera_params.items():
                            if key in DISPLAY_PARAMS.keys():
                                status_text += f"{DISPLAY_PARAMS[key]}: {value}\n"
                    # Update GUI in main thread
                    self.after(0, lambda: self.camera_status.config(text=status_text))
                except Exception as e:
                    debug_logger.error(f"Camera status error: {e}")
            
            threading.Thread(target=_update, daemon=True).start()
        
        self.after(2000, self.update_camera_status)  # Update every 2 seconds instead of 1

    def update_frame_display(self):
        if self.updating_frame_display:
            try:
                self.refresh_frame_display()
            except Exception as e:
                debug_logger.error(f"Frame display error: {e}")
        self.after(17, self.update_frame_display)

    def update_peripherals_status(self):
        """Schedule periodic peripheral updates - completely non-blocking"""
        if self.updating_peripherals_status and self.peripherals_thread is not None:
            # Skip if previous update is still running
            if hasattr(self, '_peripheral_update_running') and self._peripheral_update_running:
                debug_logger.debug("Skipping peripheral update - previous still running")
            else:
                self._peripheral_update_running = True
                # Run completely in background thread
                threading.Thread(target=self._update_peripherals_background, daemon=True).start()
        
        self.after(30000, self.update_peripherals_status)  # Update every 30 seconds
    
    def _update_peripherals_background(self):
        """Completely background peripheral status update"""
        try:
            results = {}
            
            # Get all peripheral states in background thread with timeouts
            with ThreadPoolExecutor(max_workers=1) as executor:
                
                # Filter position with timeout
                if self.peripherals_thread.efw is not None:
                    try:
                        future = executor.submit(self._get_filter_position)
                        position = future.result(timeout=2.0)
                        if position is not None:
                            results['filter_position'] = position
                    except (FutureTimeoutError, Exception) as e:
                        debug_logger.error(f"Filter check timeout/error: {e}")
                
                # PDU outlet states with timeout
                if self.peripherals_thread.pdu is not None:
                    try:
                        future = executor.submit(self._get_outlet_states)
                        outlet_states = future.result(timeout=3.0)
                        if outlet_states:
                            results['outlet_states'] = outlet_states
                    except (FutureTimeoutError, Exception) as e:
                        debug_logger.error(f"PDU check timeout/error: {e}")
                
                # Shutter state with timeout
                if self.peripherals_thread.ljm_handle is not None:
                    try:
                        future = executor.submit(self._get_shutter_state)
                        shutter_state = future.result(timeout=1.0)
                        if shutter_state is not None:
                            results['shutter_state'] = shutter_state
                    except (FutureTimeoutError, Exception) as e:
                        debug_logger.error(f"Shutter check timeout/error: {e}")
                
                # Get motor positions with timeout
                try:
                    future = executor.submit(self._get_motor_positions)
                    motor_positions = future.result(timeout=3.0)
                    if motor_positions:
                        results.update(motor_positions)
                except (FutureTimeoutError, Exception) as e:
                    debug_logger.error(f"Motor positions timeout/error: {e}")
            
            # Apply updates to GUI - only this part runs in GUI thread
            if results:
                self.after(0, lambda: self._apply_peripheral_updates(results))
                
        except Exception as e:
            debug_logger.error(f"Peripheral background update error: {e}")
        finally:
            self._peripheral_update_running = False
    
    def _get_filter_position(self):
        """Get filter position - runs in background thread"""
        try:
            with self.peripherals_thread.peripherals_lock:
                if self.peripherals_thread.efw is not None:
                    position = self.peripherals_thread.efw.GetPosition(0)
                    return list(self.filter_options.keys())[position]
        except Exception as e:
            debug_logger.error(f"Get filter position error: {e}")
            # Try to reconnect filter wheel
            self.peripherals_thread.efw = None
            self.peripherals_thread.connect_efw()
        return None
    
    def _get_outlet_states(self):
        """Get PDU outlet states - runs in background thread"""
        try:
            return asyncio.run(asyncio.wait_for(
                self.peripherals_thread.get_all_outlet_states(),
                timeout=2.0
            ))
        except asyncio.TimeoutError:
            debug_logger.error("PDU timeout")
        except Exception as e:
            debug_logger.error(f"Get outlet states error: {e}")
        return None
    
    def _get_shutter_state(self):
        """Get shutter state - runs in background thread"""
        try:
            with self.peripherals_thread.peripherals_lock:
                if self.peripherals_thread.ljm_handle is not None:
                    mask = ljm.eReadName(self.peripherals_thread.ljm_handle, "FIO_STATE")
                    fio4_state = (int(mask) >> 4) & 1
                    return 'Open' if fio4_state == 0 else 'Closed'
        except Exception as e:
            debug_logger.error(f"Get shutter state error: {e}")
        return None
    
    def _get_motor_positions(self):
        """Get all motor positions - runs in background thread"""
        positions = {}
        try:
            with self.peripherals_thread.peripherals_lock:
                # Slit position
                if self.peripherals_thread.ax_a_1 is not None:
                    try:
                        pos = self.peripherals_thread.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
                        positions['slit_position'] = 'In beam' if abs(pos - 70) < 0.01 else 'Out of beam'
                    except Exception as e:
                        debug_logger.error(f"Slit position error: {e}")
                
                # Zoom position
                if self.peripherals_thread.ax_a_2 is not None:
                    try:
                        pos = self.peripherals_thread.ax_a_2.get_position(Units.ANGLE_DEGREES)
                        zoom_map = {0: '1x', 90: '2x', 180: '3x', 270: '4x'}
                        for angle, zoom in zoom_map.items():
                            if abs(pos - angle) < 5:
                                positions['zoom_position'] = zoom
                                break
                    except Exception as e:
                        debug_logger.error(f"Zoom position error: {e}")
                
                # Focus position
                if self.peripherals_thread.ax_b_1 is not None:
                    try:
                        pos = self.peripherals_thread.ax_b_1.get_position(Units.ANGLE_DEGREES)
                        positions['focus_position'] = str(int(pos * self.focus_conversion_factor))
                    except Exception as e:
                        debug_logger.error(f"Focus position error: {e}")
                
                # Polarization stage position
                if self.peripherals_thread.ax_b_2 is not None:
                    try:
                        pos = self.peripherals_thread.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
                        if abs(pos - 15.5) < 0.01:
                            positions['pol_stage'] = 'WeDoWo'
                        elif abs(pos - 128.5) < 0.01:
                            positions['pol_stage'] = 'Wire Grid'
                        else:
                            positions['pol_stage'] = 'Neither'
                    except Exception as e:
                        debug_logger.error(f"Pol stage position error: {e}")
                
                # Halpha/QWP position
                if self.peripherals_thread.ax_b_3 is not None:
                    try:
                        pos = self.peripherals_thread.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
                        if abs(pos - 138) < 0.01:
                            positions['halpha_qwp'] = 'Halpha'
                        elif abs(pos - 13) < 0.01:
                            positions['halpha_qwp'] = 'QWP'
                        else:
                            positions['halpha_qwp'] = 'Neither'
                    except Exception as e:
                        debug_logger.error(f"Halpha/QWP position error: {e}")
                        
        except Exception as e:
            debug_logger.error(f"Get motor positions error: {e}")
        
        return positions
    
    def _apply_peripheral_updates(self, results):
        """Apply peripheral updates to GUI - runs in GUI thread"""
        try:
            if 'filter_position' in results:
                self.filter_position_var.set(results['filter_position'])
            
            if 'outlet_states' in results:
                outlet_states = results['outlet_states']
                for i in range(1, 17):
                    if outlet_states and len(outlet_states) >= i:
                        state = outlet_states[i - 1]
                        self.pdu_outlet_vars[i].set(state)
                        btn = self.pdu_outlet_buttons[i]
                        if state:
                            btn.config(text='ON', fg='green', relief='sunken')
                        else:
                            btn.config(text='OFF', fg='red', relief='raised')
            
            if 'shutter_state' in results:
                self.shutter_var.set(results['shutter_state'])
            
            if 'slit_position' in results:
                self.slit_position_var.set(results['slit_position'])
            
            if 'zoom_position' in results:
                self.zoom_stepper_var.set(results['zoom_position'])
            
            if 'focus_position' in results:
                self.focus_position_var.set(results['focus_position'])
            
            if 'pol_stage' in results:
                self.wire_grid_var.set(results['pol_stage'])
            
            if 'halpha_qwp' in results:
                self.halpha_qwp_var.set(results['halpha_qwp'])
                
        except Exception as e:
            debug_logger.error(f"Apply peripheral updates error: {e}")

    def refresh_frame_display(self, *_):
        try:
            # Try to get multiple frames if available to catch up
            frames_processed = 0
            while frames_processed < 3 and not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                    self.last_frame = frame  # Keep last frame
                    frames_processed += 1
                except queue.Empty:
                    break
            
            # Process only the last frame for display
            if frames_processed > 0 and self.last_frame is not None:
                self.process_frame(self.last_frame)
        except Exception as e:
            debug_logger.error(f"Frame display error: {e}")
        
        cv2.waitKey(1)

    def process_frame(self, data):
        try:
            # Skip processing if window is being updated
            if not self.display_lock.acquire(blocking=False):
                return
                
            try:
                if data.dtype == np.uint16 or data.dtype == np.uint8:
                    try:
                        min_val = int(self.min_val.get())
                    except:
                        min_val = 0
                    try:
                        max_val = int(self.max_val.get())
                    except:
                        max_val = 200

                    # Use faster scaling for 16-bit images
                    if data.dtype == np.uint16:
                        # Faster method: shift right by 8 bits for rough scaling
                        if max_val - min_val > 0:
                            scale = 65535.0 / (max_val - min_val)
                            scaled_data = np.clip((data.astype(np.float32) - min_val) * scale, 0, 65535).astype(np.uint16)
                        else:
                            scaled_data = data
                    else:
                        scaled_data = np.clip((data.astype(np.float32) - min_val) / (max_val - min_val) * 255, 
                                            0, 255).astype(np.uint8)

                    scaled_data = cv2.flip(scaled_data, 1)
                    
                    # Convert to BGR only if needed
                    if len(scaled_data.shape) == 2:
                        scaled_data_bgr = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2BGR)
                    else:
                        scaled_data_bgr = scaled_data

                    if hasattr(self, 'circle_center'):
                        cv2.circle(scaled_data_bgr, self.circle_center, 2, (255, 0, 0), 2)

                    if not hasattr(self, 'opencv_window_created'):
                        cv2.namedWindow('Captured Frame', cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback('Captured Frame', self.on_right_click)
                        self.opencv_window_created = True

                    cv2.imshow('Captured Frame', scaled_data_bgr)
                    
            finally:
                self.display_lock.release()
                
        except Exception as e:
            debug_logger.error(f"Frame processing error: {e}")

    def on_right_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.after(0, lambda: self.show_context_menu(x, y))

    def show_context_menu(self, x, y):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Draw Circle", command=lambda: self.draw_circle(x, y))
        menu.add_command(label="Clear Markers", command=self.clear_markers)
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        except:
            pass

    def draw_circle(self, x, y):
        self.circle_center = (x, y)
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            self.process_frame(self.last_frame)

    def clear_markers(self):
        if hasattr(self, 'circle_center'):
            del self.circle_center
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            self.process_frame(self.last_frame)

    def update_exposure_time(self, *_):
        try:
            exposure_time = float(self.exposure_time_entry.get()) / 1000
            if self.camera_thread.capturing:
                self.status_message.config(text="Cannot change during capture", fg="orange")
            else:
                self.camera_thread.set_property('EXPOSURE_TIME', exposure_time)
                self.status_message.config(text=f"Exposure: {exposure_time*1000:.1f}ms", fg="green")
        except ValueError:
            self.status_message.config(text="Invalid exposure time", fg="red")

    def update_batch_size(self, *_):
        try:
            self.batch_size = int(self.cube_size_entry.get())
            debug_logger.info(f"Batch size: {self.batch_size}")
        except:
            self.batch_size = 100

    def update_shutter(self, *_):
        """Update shutter - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.ljm_handle is None:
                    self.peripherals_thread.connect_labjack()
                    if self.peripherals_thread.ljm_handle is None:
                        return
                with self.peripherals_thread.peripherals_lock:
                    mask = ljm.eReadName(self.peripherals_thread.ljm_handle, "FIO_STATE")
                    fio4_state = (int(mask) >> 4) & 1
                    shutter_state = 'Open' if fio4_state == 0 else 'Closed'
                    if self.shutter_var.get() == shutter_state:
                        return
                    if self.shutter_var.get() == 'Open':
                        ljm.eWriteName(self.peripherals_thread.ljm_handle, "DIO4", 0)
                    elif self.shutter_var.get() == 'Closed':
                        ljm.eWriteName(self.peripherals_thread.ljm_handle, "DIO4", 1)
            except Exception as e:
                debug_logger.error(f"Shutter error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()

    def update_filter_position(self, *_):
        """Update filter position - runs in background thread"""
        def _update():
            try:
                selected_filter = self.filter_position_var.get()
                with self.peripherals_thread.peripherals_lock:
                    try:
                        if self.peripherals_thread.efw is None:
                            raise Exception("EFW not connected")
                        current_position = self.peripherals_thread.efw.GetPosition(0)
                    except:
                        self.peripherals_thread.efw = None
                        self.peripherals_thread.connect_efw()
                        if self.peripherals_thread.efw is None:
                            self.after(0, lambda: self.filter_position_menu.config(state='disabled'))
                            return
                        else:
                            self.after(0, lambda: self.filter_position_menu.config(state='normal'))
                        return
                    if selected_filter == '':
                        current_position = self.peripherals_thread.efw.GetPosition(0)
                        self.after(0, lambda: self.filter_position_var.set(
                            list(self.filter_options.keys())[current_position]))
                    else:
                        selected_position = self.filter_options[self.filter_position_var.get()]
                        if selected_position == self.peripherals_thread.efw.GetPosition(0):
                            return
                        try:
                            self.peripherals_thread.efw.SetPosition(0, selected_position)
                            debug_logger.info(f"Filter position: {self.filter_position_var.get()}")
                        except Exception as e:
                            debug_logger.error(f"Filter position error: {e}")
            except Exception as e:
                debug_logger.error(f"Update filter error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()

    def update_outlet_states(self):
        """Update outlet states - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.pdu is None:
                    return
                outlet_states = asyncio.run(self.peripherals_thread.get_all_outlet_states())
                for i in range(1, 17):
                    if outlet_states and len(outlet_states) >= i:
                        if self.pdu_outlet_vars[i].get() != outlet_states[i - 1]:
                            self.after(0, lambda i=i, state=outlet_states[i-1]: (
                                self.pdu_outlet_vars[i].set(state),
                                self.toggle_outlet(i, override=True)
                            ))
            except Exception as e:
                debug_logger.error(f"Outlet states error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()

    def power_cycle_camera(self):
        try:
            self.peripherals_thread.command_outlet(12, OutletCommand.IMMEDIATE_OFF)
            self.after(1000, lambda: self.peripherals_thread.command_outlet(12, OutletCommand.IMMEDIATE_ON))
            logging.info("Camera power cycled")
            self.after(3000, self.reconnect_camera)
        except Exception as e:
            logging.error(f"Power cycle error: {e}")

    def reconnect_camera(self):
        self.camera_thread.stop()
        self.camera_thread = CameraThread(self.shared_data, self.frame_queue,
                                          self.timestamp_queue, self)
        self.camera_thread.daemon = True
        self.camera_thread.start()

    def update_slit_position(self, *_):
        """Update slit position - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.ax_a_1 is None:
                    self.after(0, lambda: self.slit_position_menu.config(state='disabled'))
                    return
                slit_option = self.slit_position_var.get()
                with self.peripherals_thread.peripherals_lock:
                    curr_slit_pos = self.peripherals_thread.ax_a_1.get_position(Units.LENGTH_MILLIMETRES)
                    if slit_option == 'In beam' and abs(curr_slit_pos - 70) > 0.01:
                        debug_logger.info("Moving slit in beam")
                        self.peripherals_thread.ax_a_1.move_absolute(70, Units.LENGTH_MILLIMETRES)
                    elif slit_option == 'Out of beam' and abs(curr_slit_pos) > 0.01:
                        debug_logger.info("Moving slit out of beam")
                        self.peripherals_thread.ax_a_1.move_absolute(0, Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Slit position error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()
    
    def update_halpha_qwp(self, *_):
        """Update Halpha/QWP - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_3 is None:
                    self.after(0, lambda: (
                        self.halpha_qwp_var.set(''),
                        self.halpha_qwp_menu.config(state='disabled')
                    ))
                    return
                halpha_qwp_option = self.halpha_qwp_var.get()
                with self.peripherals_thread.peripherals_lock:
                    curr_pos = self.peripherals_thread.ax_b_3.get_position(Units.LENGTH_MILLIMETRES)
                    if halpha_qwp_option == 'Halpha' and abs(curr_pos - 138) > 0.01:
                        debug_logger.info("Moving Halpha in beam")
                        self.peripherals_thread.ax_b_3.move_absolute(138, Units.LENGTH_MILLIMETRES)
                    elif halpha_qwp_option == 'QWP' and abs(curr_pos - 13) > 0.01:
                        debug_logger.info("Moving QWP in beam")
                        self.peripherals_thread.ax_b_3.move_absolute(13, Units.LENGTH_MILLIMETRES)
                    elif halpha_qwp_option == 'Neither' and abs(curr_pos - 75.5) > 0.01:
                        debug_logger.info("Moving Halpha/QWP out")
                        self.peripherals_thread.ax_b_3.move_absolute(75.5, Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Halpha/QWP error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()

    def update_pol_stage(self, *_):
        """Update polarization stage - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_2 is None:
                    self.after(0, lambda: (
                        self.wire_grid_var.set(''),
                        self.wire_grid_menu.config(state='disabled')
                    ))
                    return
                option = self.wire_grid_var.get()
                with self.peripherals_thread.peripherals_lock:
                    curr_pos = self.peripherals_thread.ax_b_2.get_position(Units.LENGTH_MILLIMETRES)
                    if option == 'WeDoWo' and abs(curr_pos - 15.5) > 0.01:
                        debug_logger.info("Moving WeDoWo in beam")
                        self.peripherals_thread.ax_b_2.move_absolute(15.5, Units.LENGTH_MILLIMETRES)
                    elif option == 'Wire Grid' and abs(curr_pos - 128.5) > 0.01:
                        debug_logger.info("Moving Wire Grid in beam")
                        self.peripherals_thread.ax_b_2.move_absolute(128.5, Units.LENGTH_MILLIMETRES)
                    elif option == 'Neither' and abs(curr_pos - 72) > 0.01:
                        debug_logger.info("Moving polarizers out")
                        self.peripherals_thread.ax_b_2.move_absolute(72, Units.LENGTH_MILLIMETRES)
            except Exception as e:
                debug_logger.error(f"Pol stage error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()

    def update_zoom_stepper(self, *_):
        """Update zoom - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.ax_a_2 is None:
                    self.after(0, lambda: (
                        self.zoom_stepper_var.set(''),
                        self.zoom_stepper_menu.config(state='disabled')
                    ))
                    return
                zoom_option = self.zoom_stepper_var.get()
                zoom_positions = {'1x': 0, '2x': 90, '3x': 180, '4x': 270}
                desired_position = zoom_positions[zoom_option]
                with self.peripherals_thread.peripherals_lock:
                    curr_pos = self.peripherals_thread.ax_a_2.get_position(Units.ANGLE_DEGREES)
                    if abs(curr_pos - desired_position) > 0.01:
                        debug_logger.info(f"Setting zoom to {zoom_option}")
                        self.peripherals_thread.ax_a_2.move_absolute(desired_position, Units.ANGLE_DEGREES)
            except Exception as e:
                debug_logger.error(f"Zoom error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()
    
    def update_focus_position(self, *_):
        """Update focus position - runs in background thread"""
        def _update():
            try:
                if self.peripherals_thread.ax_b_1 is None:
                    self.after(0, lambda: (
                        self.focus_position_entry.config(state='disabled'),
                        self.set_focus_button.config(state='disabled')
                    ))
                    return
                focus_position = float(self.focus_position_var.get())
                with self.peripherals_thread.peripherals_lock:
                    curr_pos = self.peripherals_thread.ax_b_1.get_position(Units.ANGLE_DEGREES)
                    desired_pos = focus_position / self.focus_conversion_factor
                    if abs(curr_pos - desired_pos) > 0.01:
                        debug_logger.info(f"Setting focus to {focus_position} um")
                        self.peripherals_thread.ax_b_1.move_absolute(desired_pos, Units.ANGLE_DEGREES)
            except Exception as e:
                debug_logger.error(f"Focus error: {e}")
        
        threading.Thread(target=_update, daemon=True).start()

    def toggle_outlet(self, idx, override=False):
        """Toggle outlet on/off"""
        try:
            state = self.pdu_outlet_vars[idx].get()
            if not override:
                response = messagebox.askyesno("Confirm", 
                    f"Turn {'ON' if state else 'OFF'} outlet {idx}?")
                if not response:
                    self.pdu_outlet_vars[idx].set(not state)
                    return
            cmd = OutletCommand.IMMEDIATE_ON if state else OutletCommand.IMMEDIATE_OFF
            self.peripherals_thread.command_outlet(idx, cmd)
            btn = self.pdu_outlet_buttons[idx]
            if state:
                btn.config(text='ON', fg='green', relief='sunken')
            else:
                btn.config(text='OFF', fg='red', relief='raised')
        except Exception as e:
            debug_logger.error(f"Toggle outlet error: {e}")

    def change_binning(self, selected_binning):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
            self.binning_var.set(self.binning_var.get())
        else:
            binning_value = {"1x1": 1, "2x2": 2, "4x4": 4}[selected_binning]
            self.camera_thread.set_property('BINNING', binning_value)

    def change_bit_depth(self, selected_bit_depth):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
            self.bit_depth_var.set(self.bit_depth_var.get())
        else:
            bit_depth_value = {"8-bit": 1, "16-bit": 2}[selected_bit_depth]
            self.camera_thread.set_property('IMAGE_PIXEL_TYPE', bit_depth_value)

    def change_readout_speed(self, selected_mode):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
            self.readout_speed_var.set(self.readout_speed_var.get())
        else:
            readout_speed_value = {"Ultra Quiet Mode": 1.0, "Standard Mode": 2.0}[selected_mode]
            if selected_mode == "Standard Mode":
                self.sensor_mode_var.set("Standard")
                self.change_sensor_mode("Standard")
            self.camera_thread.set_property('READOUT_SPEED', readout_speed_value)

    def change_sensor_mode(self, selected_mode):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
            self.sensor_mode_var.set(self.sensor_mode_var.get())
        else:
            sensor_mode_value = {"Photon Number Resolving": 18.0, "Standard": 1.0}[selected_mode]
            self.camera_thread.set_property('SENSOR_MODE', sensor_mode_value)
            if selected_mode == "Photon Number Resolving":
                self.readout_speed_var.set("Ultra Quiet Mode")
                self.change_readout_speed("Ultra Quiet Mode")

    def change_subarray_mode(self, selected_mode):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
            self.subarray_mode_var.set(self.subarray_mode_var.get())
        else:
            subarray_mode_value = {"Off": 1.0, "On": 2.0}[selected_mode]
            self.camera_thread.set_property('SUBARRAY_MODE', subarray_mode_value)
            state = 'normal' if selected_mode == "On" else 'disabled'
            self.subarray_hpos_entry.config(state=state)
            self.subarray_hsize_entry.config(state=state)
            self.subarray_vpos_entry.config(state=state)
            self.subarray_vsize_entry.config(state=state)

    def update_subarray(self, *_):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
        else:
            try:
                hpos = float(round(float(self.subarray_hpos_entry.get()) / 4) * 4)
                hsize = float(round(float(self.subarray_hsize_entry.get()) / 4) * 4)
                vpos = float(round(float(self.subarray_vpos_entry.get()) / 4) * 4)
                vsize = float(round(float(self.subarray_vsize_entry.get()) / 4) * 4)

                self.camera_thread.set_property('SUBARRAY_HPOS', hpos)
                self.camera_thread.set_property('SUBARRAY_HSIZE', hsize)
                self.camera_thread.set_property('SUBARRAY_VPOS', vpos)
                self.camera_thread.set_property('SUBARRAY_VSIZE', vsize)
            except ValueError:
                debug_logger.error("Invalid subarray parameters")

    def update_framebundle(self):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
            self.framebundle_var.set(self.framebundle_var.get())
        else:
            framebundle_enabled = self.framebundle_var.get()
            self.camera_thread.set_property('FRAMEBUNDLE_MODE', 2.0 if framebundle_enabled else 1.0)

    def update_frames_per_bundle(self, *_):
        if self.camera_thread.capturing:
            self.status_message.config(text="Cannot change during capture", fg="orange")
        else:
            try:
                frames_per_bundle = int(self.frames_per_bundle_entry.get())
                self.camera_thread.set_property('FRAMEBUNDLE_NUMBER', frames_per_bundle)
            except ValueError:
                debug_logger.error("Invalid frames per bundle")

    def start_capture(self):
        try:
            # Check if camera needs reconnection
            if hasattr(self.camera_thread, 'needs_reconnect') and self.camera_thread.needs_reconnect:
                self.status_message.config(text="Camera needs reset - please use Reset Camera button", fg="red")
                messagebox.showerror("Camera Error", 
                                   "Camera is in an error state.\nPlease use the Reset Camera button to reconnect.")
                return
            
            self.status_message.config(text="Starting capture...", fg="blue")
            self.update_gps_timestamp("Waiting for GPS...")

            if self.save_data_var.get():
                self.save_queue = queue.Queue(maxsize=50000)
                self.camera_thread.save_queue = self.save_queue
                object_name = self.object_name_entry.get()
                self.save_thread = SaveThread(self.save_queue, self.timestamp_queue, 
                                             self.camera_thread, object_name, self.shared_data)
                self.save_thread.batch_size = self.batch_size
                self.save_thread.start()
            else:
                self.camera_thread.save_queue = None

            self.camera_thread.start_capture()
            
            # Wait a moment to check if capture actually started
            time.sleep(0.5)
            if not self.camera_thread.capturing:
                self.status_message.config(text="Failed to start capture - check camera", fg="red")
                if self.save_thread:
                    self.save_thread.stop()
                    self.save_thread = None
                return

            # Disable controls
            for widget in [self.exposure_time_entry, self.save_data_checkbox, 
                          self.start_button, self.reset_button, self.binning_menu,
                          self.bit_depth_menu, self.readout_speed_menu, self.sensor_mode_menu,
                          self.subarray_mode_menu, self.framebundle_checkbox, 
                          self.frames_per_bundle_entry]:
                widget.config(state='disabled')
            
            if self.subarray_mode_var.get() == "On":
                for widget in [self.subarray_hpos_entry, self.subarray_hsize_entry,
                              self.subarray_vpos_entry, self.subarray_vsize_entry]:
                    widget.config(state='disabled')
                    
            self.status_message.config(text="Capture running", fg="green")
            
        except Exception as e:
            logging.error(f"Start capture error: {e}")
            self.status_message.config(text=f"Error: {e}", fg="red")

    def stop_capture(self):
        try:
            self.status_message.config(text="Stopping capture...", fg="blue")
            self.camera_thread.stop_capture()

            if hasattr(self, 'save_thread') and self.save_thread is not None and self.save_thread.is_alive():
                self.save_thread.stop()
                self.save_thread.join(timeout=10)
                self.save_thread = None
                self.camera_thread.save_queue = None

            # Enable controls
            for widget in [self.exposure_time_entry, self.save_data_checkbox,
                          self.start_button, self.reset_button, self.binning_menu,
                          self.bit_depth_menu, self.readout_speed_menu, self.sensor_mode_menu,
                          self.subarray_mode_menu, self.framebundle_checkbox,
                          self.frames_per_bundle_entry]:
                widget.config(state='normal')

            if self.subarray_mode_var.get() == "On":
                for widget in [self.subarray_hpos_entry, self.subarray_hsize_entry,
                              self.subarray_vpos_entry, self.subarray_vsize_entry]:
                    widget.config(state='normal')
                    
            self.status_message.config(text="Capture stopped", fg="blue")
            self.update_gps_timestamp("No capture active")
            
        except Exception as e:
            logging.error(f"Stop capture error: {e}")

    def reset_camera(self):
        try:
            self.status_message.config(text="Resetting camera...", fg="blue")
            self.updating_camera_status = False
            self.updating_frame_display = False
            
            # Clear the needs_reconnect flag
            if hasattr(self.camera_thread, 'needs_reconnect'):
                self.camera_thread.needs_reconnect = False
            
            self.camera_thread.reset_camera()
            
            # Wait a moment for reset to complete
            time.sleep(1.0)
            
            self.updating_camera_status = True
            self.updating_frame_display = True
            self.status_message.config(text="Camera reset complete", fg="green")
        except Exception as e:
            logging.error(f"Reset camera error: {e}")
            self.status_message.config(text=f"Reset error: {e}", fg="red")

    def on_close(self):
        try:
            logging.info("Closing application")
            
            # Stop all updates immediately
            self.updating_camera_status = False
            self.updating_frame_display = False
            self.updating_peripherals_status = False
            
            # Stop save thread first if it exists
            if hasattr(self, 'save_thread') and self.save_thread and self.save_thread.is_alive():
                logging.info("Stopping save thread")
                self.save_thread.stop()
                self.save_thread.join(timeout=5)

            # Stop camera thread with timeout
            if hasattr(self, 'camera_thread') and self.camera_thread:
                logging.info("Stopping camera thread")
                self.camera_thread.running = False
                self.camera_thread.watchdog_enabled = False
                
                # Force stop capture if still running
                if self.camera_thread.capturing:
                    self.camera_thread.capturing = False
                    self.camera_thread.watchdog_enabled = False
                
                if self.camera_thread.is_alive():
                    # Give thread time to stop gracefully
                    self.camera_thread.stop()
                    self.camera_thread.join(timeout=3)
                    
                    # If still alive, force termination
                    if self.camera_thread.is_alive():
                        logging.warning("Camera thread did not stop gracefully")

            # Close OpenCV windows before camera cleanup
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)  # Process any pending window events
            except:
                pass

            # Disconnect peripherals
            if hasattr(self, 'peripherals_thread') and self.peripherals_thread:
                logging.info("Disconnecting peripherals")
                try:
                    self.peripherals_thread.disconnect_peripherals()
                    if self.peripherals_thread.is_alive():
                        self.peripherals_thread.join(timeout=3)
                except:
                    pass

            # Final cleanup - destroy GUI
            logging.info("Destroying GUI")
            self.quit()  # Stop mainloop
            self.destroy()  # Destroy window
            
            logging.info("Application closed successfully")
            
        except Exception as e:
            logging.error(f"Error during close: {e}")
            # Force exit on error
            import sys
            sys.exit(1)


if __name__ == "__main__":
    try:
        logging.info("Starting Camera Control Application")
        
        shared_data = SharedData()
        frame_queue = queue.Queue(maxsize=30)
        timestamp_queue = queue.Queue(maxsize=100000)  # Very large buffer
        
        app = CameraGUI(shared_data, None, None, frame_queue, timestamp_queue)
        
        camera_thread = CameraThread(shared_data, frame_queue, timestamp_queue, app)
        camera_thread.daemon = True
        camera_thread.start()
        
        peripherals_thread = PeripheralsThread(shared_data, frame_queue, timestamp_queue,
                                               "18.25.72.251", "/dev/ttyACM0", "/dev/ttyACM1", app)
        peripherals_thread.daemon = True
        peripherals_thread.start()
        
        app.camera_thread = camera_thread
        app.peripherals_thread = peripherals_thread
        
        app.protocol("WM_DELETE_WINDOW", app.on_close)
        app.mainloop()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import sys
        sys.exit(1)
