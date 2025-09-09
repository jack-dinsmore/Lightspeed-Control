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
    
    @classmethod
    def __enter__(cls):
        acquired = cls._lock.acquire(timeout=10)
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
        self.buffer_size = 2000
        self.safe_buffer = SafeFrameBuffer(self.buffer_size)
        self.capture_lock = threading.RLock()
        self.save_queue = None
        
        # Command queue for thread-safe property changes
        self.command_queue = queue.Queue(maxsize=100)
        self.command_response_queue = queue.Queue(maxsize=100)
        
        # Frame processing lock
        self.frame_processing_lock = threading.Lock()
        
        # Performance monitoring
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_print_time = time.time()
        
        # Set thread name for debugging
        self.name = "CameraThread"
        
        # Add flag for camera ready state
        self.camera_ready = threading.Event()

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

        # Configure camera
        if self.first_run:
            self.set_defaults()
            self.first_run = False
        else:
            self.restore_modified_params()

        self.update_camera_params()
        logging.info("Camera initialized and ready")
        
        # Signal that camera is ready
        self.camera_ready.set()
        
        # Main camera loop
        while self.running:
            try:
                self.process_commands()
                self.paused.wait(timeout=0.001)
                
                if self.capturing:
                    self.capture_frame()
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"Error in camera loop: {e}")
                time.sleep(0.1)

    def process_commands(self):
        """Process commands from the queue"""
        processed = 0
        max_commands_per_iteration = 10
        
        while processed < max_commands_per_iteration and not self.command_queue.empty():
            try:
                command = self.command_queue.get_nowait()
                processed += 1
                cmd_type = command.get('type')
                
                with DCamLock():
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
            self.stop_capture()
            with DCamLock():
                if self.dcam is not None:
                    self.dcam.dev_close()
                    self.dcam = None
                Dcamapi.uninit()

    def reset_camera(self):
        logging.info("Resetting camera")
        self.pause_thread()
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
                'OUTPUT_TRIG_KIND_0': 2.0,
                'OUTPUT_TRIG_ACTIVE_0': 1.0,
                'OUTPUT_TRIG_POLARITY_0': 1.0,
                'OUTPUT_TRIG_PERIOD_0': 1.0,
                'OUTPUT_TRIG_KIND_1': 1.0,
                'OUTPUT_TRIG_ACTIVE_1': 1.0,
                'OUTPUT_TRIG_POLARITY_1': 2.0,
                'OUTPUT_TRIG_PERIOD_1': 1.0,
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
        self.stop_capture()
        self.paused.set()
        with DCamLock():
            if self.dcam is not None:
                self.dcam.dev_close()
                self.dcam = None
            Dcamapi.uninit()

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
        with DCamLock():
            for prop_name, value in self.modified_params.items():
                if self.dcam is not None:
                    self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)

    def _start_capture_internal(self):
        """Internal method to start capture - OPTIMIZED"""
        start_time = time.time()
        logging.info("Starting capture immediately")
        
        # Clear GPS buffer in background
        threading.Thread(target=GPS_time.clear_buffer, daemon=True).start()

        if self.capturing:
            debug_logger.warning("Already capturing, stopping first")
            self._stop_capture_internal()

        if self.dcam is None:
            logging.error("Camera not initialized")
            return
        
        self.safe_buffer.clear()
        
        # Clear display queue to prevent old frames
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass

        if self.dcam.buf_alloc(self.buffer_size) is not False:
            self.capturing = True
            self.frame_index = 0
            self.first_frame = True
            self.frame_count = 0
            self.last_frame_time = time.time()
            self.last_print_time = time.time()

            self.dcam.prop_setgetvalue(CAMERA_PARAMS['TIME_STAMP_PRODUCER'], 1)

            if self.dcam.cap_start() is False:
                logging.error(f"Capture start failed: {self.dcam.lasterr()}")
                self.capturing = False
                self.dcam.buf_release()
            else:
                elapsed = time.time() - start_time
                logging.info(f"Capture started successfully in {elapsed:.3f} seconds")
        else:
            logging.error("Buffer allocation failed")

    def start_capture(self):
        """Direct capture start without queue delays - OPTIMIZED"""
        def _start():
            with self.capture_lock:
                with DCamLock():
                    self._start_capture_internal()
        
        # Start immediately in current thread if camera is ready
        if self.camera_ready.is_set():
            _start()
        else:
            # Wait for camera in background thread
            def _wait_and_start():
                self.camera_ready.wait(timeout=30)
                _start()
            threading.Thread(target=_wait_and_start, daemon=True, name="StartCapture").start()

    def _stop_capture_internal(self):
        """Internal method to stop capture"""
        debug_logger.info("Stopping capture")
        self.capturing = False
        
        # Clear queues immediately
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
                
        if self.dcam is not None:
            if self.dcam.cap_stop() is not False:
                self.dcam.buf_release()
            logging.info("Capture stopped")
        self.restore_modified_params()

    def stop_capture(self):
        """Direct capture stop without queue delays"""
        def _stop():
            with self.capture_lock:
                with DCamLock():
                    self._stop_capture_internal()
        
        threading.Thread(target=_stop, daemon=True, name="StopCapture").start()

    def capture_frame(self):
        timeout_milisec = 1000
        try:
            # Use trylock to avoid blocking
            if not self.frame_processing_lock.acquire(blocking=False):
                return
                
            try:
                with DCamLock():
                    if self.dcam is None or not self.capturing:
                        return
                        
                    if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                        frame_index_safe = self.frame_index % self.buffer_size
                        result = self.dcam.buf_getframe_with_timestamp_and_framestamp(frame_index_safe)
                        
                        if result is not False:
                            frame, npBuf, timestamp, framestamp = result
                            
                            # Print frame info periodically (every second)
                            current_time = time.time()
                            if current_time - self.last_print_time > 1.0:
                                logging.info(f"Frame {self.frame_index}: timestamp={timestamp.sec+timestamp.microsec/1e6:.6f}, framestamp={framestamp}")
                                self.last_print_time = current_time
                            
                            # Monitor frame rate
                            self.frame_count += 1
                            if current_time - self.last_frame_time > 10:
                                fps = self.frame_count / (current_time - self.last_frame_time)
                                logging.info(f"FPS: {fps:.2f}")
                                self.frame_count = 0
                                self.last_frame_time = current_time
                            
                            frame_copy = np.copy(npBuf)
                            self.safe_buffer.add_frame(self.frame_index, frame_copy)
                            
                            # OPTIMIZED: Drop old frames from display queue
                            queue_size = self.frame_queue.qsize()
                            if queue_size > 5:  # Keep only recent frames
                                for _ in range(queue_size - 5):
                                    try:
                                        self.frame_queue.get_nowait()
                                    except:
                                        break
                            
                            # Add new frame
                            try:
                                self.frame_queue.put_nowait(frame_copy)
                            except queue.Full:
                                # Drop oldest frame if queue is full
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put_nowait(frame_copy)
                                except:
                                    pass

                            try:
                                self.timestamp_queue.put_nowait((timestamp.sec + timestamp.microsec / 1e6, framestamp))
                            except queue.Full:
                                pass
                            
                            if self.save_queue is not None:
                                try:
                                    self.save_queue.put_nowait((frame_copy, timestamp.sec + timestamp.microsec / 1e6, framestamp))
                                except queue.Full:
                                    debug_logger.warning("Save queue full")

                            # Fetch the GPS timestamp only once at the start of the capture sequence
                            if self.first_frame:
                                # Do GPS fetch in background to avoid blocking
                                def fetch_gps():
                                    try:
                                        self.start_time = GPS_time.get_first_timestamp()
                                        if self.start_time:
                                            logging.info(f"GPS timestamp acquired: {self.start_time.isot}")
                                            if self.gui_ref:
                                                self.gui_ref.update_gps_timestamp(self.start_time.isot)
                                        else:
                                            logging.warning("No GPS timestamp available")
                                            if self.gui_ref:
                                                self.gui_ref.update_gps_timestamp("No GPS timestamp")
                                    except Exception as e:
                                        logging.error(f"GPS timestamp error: {e}")
                                        self.start_time = None
                                        if self.gui_ref:
                                            self.gui_ref.update_gps_timestamp("GPS error")
                                
                                threading.Thread(target=fetch_gps, daemon=True).start()
                                self.first_frame = False

                            self.frame_index += 1
                        else:
                            dcamerr = self.dcam.lasterr()
                            if not dcamerr.is_timeout():
                                debug_logger.error(f'Frame capture error: {dcamerr}')
            finally:
                self.frame_processing_lock.release()
                
        except Exception as e:
            logging.error(f"Capture frame error: {e}")
            time.sleep(0.1)


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
            # Delay peripheral connections to prioritize camera
            time.sleep(2)
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