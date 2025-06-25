import tkinter as tk
from tkinter import StringVar, OptionMenu, Checkbutton, Label, Entry, Button, Scale, Frame, LabelFrame
from dcam import Dcamapi, Dcam, DCAMERR
from camera_params import CAMERA_PARAMS, DISPLAY_PARAMS
# import GPS_time  # Importing the GPS_time module
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
# from PyZWOEFW import EFW
import ctypes
ctypes.CDLL("libudev.so.1", mode=ctypes.RTLD_GLOBAL)


class SharedData:
    def __init__(self):
        self.camera_params = {}
        self.lock = threading.Lock()


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

    def run(self):
        # Run the connection process in a separate thread to avoid blocking the GUI
        threading.Thread(target=self.connect_camera, daemon=True).start()

    def connect_camera(self):
        print("Initializing DCAM API...")
        if Dcamapi.init() is not False:
            try:
                print("Opening camera device...")
                self.dcam = Dcam(0)  # Assuming first device (index 0)
                if self.dcam.dev_open() is not False:
                    print("Camera device opened successfully.")

                    # On the first run, set defaults; otherwise, restore modified parameters
                    if self.first_run:
                        self.set_defaults()
                        self.first_run = False  # Set flag to False after the first run
                    else:
                        self.restore_modified_params()

                    self.update_camera_params()
                    print("Entering main camera loop.")
                    while self.running:
                        self.paused.wait()  # Wait if the thread is paused
                        if self.capturing:
                            self.capture_frame()
                        else:
                            time.sleep(0.1)  # Sleep only when not capturing
                else:
                    print(f"Error opening device: {self.dcam.lasterr()}")
            except Exception as e:
                print(f"Exception in connect_camera: {e}")
        else:
            print(f"Error initializing DCAM API: {Dcamapi.lasterr()}")

    def disconnect_camera(self):
        print("Disconnecting camera...")
        self.stop_capture()
        if self.dcam is not None:
            self.dcam.dev_close()
        Dcamapi.uninit()
        print("Camera disconnected.")

    def reset_camera(self):
        print("Resetting camera...")
        self.pause_thread()  # Pause the thread before resetting
        self.disconnect_camera()
        # Run the camera connection in a separate thread to avoid blocking the GUI
        threading.Thread(target=self.connect_camera, daemon=True).start()
        self.resume_thread()  # Resume the thread after resetting
        print("Camera has been reset.")

    def pause_thread(self):
        print("Pausing camera thread...")
        self.paused.clear()  # Pauses the thread

    def resume_thread(self):
        print("Resuming camera thread...")
        self.paused.set()  # Resumes the thread

    def set_defaults(self):
        print("Setting default camera parameters...")
        self.set_property('READOUT_SPEED', 1.0)
        self.set_property('EXPOSURE_TIME', 0.1)
        self.set_property('TRIGGER_SOURCE', 1.0)  # 1.0 corresponds to INTERNAL
        self.set_property('TRIGGER_MODE', 6.0)  # 6.0 corresponds to START
        self.set_property('OUTPUT_TRIG_KIND_0', 1.0) # 1.0 corresonds to LOW
        self.set_property('OUTPUT_TRIG_ACTIVE_0', 1.0)
        self.set_property('OUTPUT_TRIG_POLARITY_0', 2.0)
        self.set_property('OUTPUT_TRIG_PERIOD_0', 1.0)
        self.set_property('SENSOR_MODE', 18.0)
        self.set_property('IMAGE_PIXEL_TYPE', 1.0)

    def update_camera_params(self):
        print("Updating camera parameters...")
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

    def stop(self):
        print("Stopping camera thread...")
        self.running = False
        self.stop_capture()
        if self.dcam is not None:
            self.dcam.dev_close()
        Dcamapi.uninit()
        self.join()  # Wait for the thread to finish before exiting
        print("Camera thread stopped.")

    def set_property(self, prop_name, value):
        if prop_name in CAMERA_PARAMS:
            print(f"Setting property: {prop_name} = {value}")
            set_success = self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
            if set_success is False:
                raise Exception(f"Failed to set property {prop_name}: {self.dcam.lasterr()}")
            self.update_camera_params()
            # Track the modified parameter in the dictionary
            self.modified_params[prop_name] = value

    def restore_modified_params(self):
        print("Restoring modified camera parameters...")
        # Restore all modified parameters to their original values
        for prop_name, value in self.modified_params.items():
            self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
        print("Restored all modified camera parameters.")

    def start_capture(self):
        print("Starting capture...")
        # Clear the GPS timestamp buffer before starting a new capture
        # GPS_time.clear_buffer()

        # Ensure the camera is not capturing before starting a new capture session
        if self.capturing:
            print("Capture is already running. Stopping previous capture.")
            self.stop_capture()

        # Allocate buffer for image capture before triggering
        if self.dcam.buf_alloc(self.buffer_size) is not False:  # Adjust buffer size as necessary
            self.capturing = True
            self.frame_index = 0  # Reset the frame index when starting capture
            self.first_frame = True  # Initialize the first_frame flag

            # Set the time stamp producer property
            self.dcam.prop_setgetvalue(CAMERA_PARAMS['TIME_STAMP_PRODUCER'], 1)

            # Start capturing
            if self.dcam.cap_start() is False:
                print(f"Error starting capture: {self.dcam.lasterr()}")
        else:
            print("Buffer allocation failed. Capture not started.")

    def stop_capture(self):
        print("Stopping capture...")
        self.capturing = False
        if self.dcam.cap_stop() is not False:
            self.dcam.buf_release()
        print("Capture stopped and buffer released.")
        # Restore modified parameters after stopping capture
        self.restore_modified_params()

    def capture_frame(self):
        timeout_milisec = 1  # Adjust timeout if necessary
        while self.capturing:
            if self.dcam.wait_capevent_frameready(timeout_milisec) is not False:
                # Capture the frame at the current index
                result = self.dcam.buf_getframe_with_timestamp_and_framestamp(self.frame_index % self.buffer_size)
                if result is not False:
                    frame, npBuf, timestamp, framestamp = result
                    # Optionally comment out this print statement
                    print(f"Frame: {self.frame_index}, Timestamp: {timestamp.sec+timestamp.microsec/1e6}, Framestamp: {framestamp}")

                    # Insert frame and timestamp into queues, dropping old frames if queue is full
                    try:
                        self.frame_queue.put_nowait(npBuf)
                    except queue.Full:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(npBuf)

                    self.timestamp_queue.put((timestamp.sec + timestamp.microsec / 1e6, framestamp))
                    if hasattr(self, 'save_queue') and self.save_queue is not None:
                        # Copy the frame to ensure it's not overwritten
                        self.save_queue.put((npBuf.copy(), timestamp.sec + timestamp.microsec / 1e6, framestamp))  # Include timestamp and framestamp

                    # Fetch the GPS timestamp only once at the start of the capture sequence
                    if self.first_frame:
                        # self.start_time = GPS_time.get_first_timestamp()
                        self.first_frame = False  # Prevent further updates

                    # Increment the frame index
                    self.frame_index += 1
                else:
                    dcamerr = self.dcam.lasterr()
                    if dcamerr.is_timeout():
                        print('===: timeout')
                    else:
                        print(f'-NG: Dcam.wait_event() fails with error {dcamerr}')
            else:
                time.sleep(0.01)  # Avoid tight loop if wait fails


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
        # Wait until camera_thread.start_time is available
        #while self.camera_thread.start_time is None and self.running:
            #time.sleep(0.1)

        print(self.running)
        if not self.running:
            return

        # Get start_time and format it
        start_time = self.camera_thread.start_time

        # For the filename, remove characters not suitable for filenames
        # start_time_filename_str = start_time.strftime('%Y%m%d_%H%M%S%f')
        current_time = time.localtime()
        start_time_filename_str = time.strftime('%Y%m%d_%H%M%S', current_time)

        os.makedirs("captures", exist_ok=True)

        # Start processing frames
        while self.running or not self.save_queue.empty():
            try:
                frame, timestamp, framestamp = self.save_queue.get(timeout=1)
                #print(f"SaveThread: Got frame, timestamp {timestamp}, framestamp {framestamp}")
                self.frame_buffer.append(frame)
                self.timestamp_buffer.append(timestamp)
                self.framestamp_buffer.append(framestamp)

                # If we have enough frames, write them to disk
                if len(self.frame_buffer) >= self.batch_size:
                    self.write_cube_to_disk(start_time_filename_str)
            except queue.Empty:
                time.sleep(0.1)

        # After loop ends, write any remaining frames
        if self.frame_buffer:
            self.write_cube_to_disk(start_time_filename_str)

    def write_cube_to_disk(self, start_time_filename_str):
        # Increment cube index
        self.cube_index += 1

        # Generate filename with cube index
        filename = f"{self.object_name}_{start_time_filename_str}_cube{self.cube_index:03d}.fits"
        filepath = os.path.join("captures", filename)
        #print(f"SaveThread: Writing cube to {filepath}")

        # Create Primary HDU
        primary_hdu = fits.PrimaryHDU()
        # primary_hdu.header['GPSSTART'] = (str(self.camera_thread.start_time.isot), 'GPS timestamp when data acquisition started')
        primary_hdu.header['OBJECT'] = (self.object_name, 'Object name')
        primary_hdu.header['CUBEIDX'] = (self.cube_index, 'Cube index number')

        # Stack frames into a 3D numpy array
        data_cube = np.stack(self.frame_buffer, axis=0)

        # Create ImageHDU for the data cube
        image_hdu = fits.ImageHDU(data=data_cube)
        image_hdu.header['EXTNAME'] = 'DATA_CUBE'
        for key, value in self.shared_data.camera_params.items():
            with warnings.catch_warnings():
                # Ignore warnings for keys being too long
                warnings.filterwarnings("ignore")
                image_hdu.header[key] = value

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

        #print(f"SaveThread: Saved cube to {filepath}")

        # Clear the buffers
        self.frame_buffer.clear()
        self.timestamp_buffer.clear()
        self.framestamp_buffer.clear()

    def stop(self):
        self.running = False
        # Do not call self.join() here; let the calling code handle it

class PeripheralsThread(threading.Thread):
    def __init__(self, shared_data, frame_queue, timestamp_queue, gui_ref):
        super().__init__()
        self.shared_data = shared_data
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.gui_ref = gui_ref  # Reference to the GUI
        self.efw = None # ZWO 7-position filter wheel

    def run(self):
        # Run the connection process in a separate thread to avoid blocking the GUI
        threading.Thread(target=self.connect_peripherals, daemon=True).start()

    def connect_peripherals(self):
        print("Connecting to ZWO filter wheel...")
        try:
            self.efw = EFW()
            # Need to read the position before setting position will work
            self.efw.GetPosition(0)
        except:
            print("Failed to connect to ZWO filter wheel.")
            return

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

        # Initialize percentile variables within the Tkinter root window context
        self.min_val = tk.StringVar(value="0")  # Initial min percentile set to 0%
        self.max_val = tk.StringVar(value="200")  # Initial max percentile set to 200%

        self.title("Camera Parameters")
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

        # Label for status messages - move it to the top so it's defined early
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="w", width=40, wraplength=400, fg="blue")
        self.status_message.grid(row=5, column=0, columnspan=2, sticky='nsew')

        # Camera Controls
        camera_controls_frame = LabelFrame(self.main_frame, text="Camera Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1, sticky='n')

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=0, column=0)
        self.exposure_time_var = tk.DoubleVar()
        self.exposure_time_var.set(200)
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

        # Make menu to select output trigger kind
        Label(camera_controls_frame, text="Output Trigger Kind:").grid(row=6, column=0)
        self.output_trigger_kind_var = tk.StringVar()
        self.output_trigger_kind_var.set('LOW')
        self.output_trigger_kind_options = {'LOW': 1, 'GLOBALEXPOSURE': 2, 'HIGH': 5}
        self.output_trigger_kind_menu = OptionMenu(camera_controls_frame, self.output_trigger_kind_var,
                                                   *self.output_trigger_kind_options.keys(),
                                                   command=self.update_output_trigger)
        self.output_trigger_kind_menu.grid(row=6, column=1)

        # Make menu to select filter position
        Label(camera_controls_frame, text="Filter Position:").grid(row=7, column=0)
        self.filter_position_var = tk.StringVar()
        self.filter_position_options = {'Open': 0, 'u\'': 1, 'g\'': 2, 'r\'': 3,
                                        'i\'': 4, 'z\'': 5}
        self.filter_position_menu = OptionMenu(camera_controls_frame, self.filter_position_var,
                                               *self.filter_position_options.keys(),
                                               command=self.update_filter_position)
        self.filter_position_menu.grid(row=7, column=1)

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
        # self.min_slider = Scale(display_controls_frame, from_=0, to=100, orient='horizontal', variable=self.min_val)
        # self.min_slider.grid(row=0, column=1)

        Label(display_controls_frame, text="Max Count:").grid(row=0, column=2)
        self.max_entry = Entry(display_controls_frame, textvariable=self.max_val)
        self.max_val.trace_add("write", self.refresh_frame_display)
        self.max_entry.grid(row=0, column=3)
        # self.max_slider = Scale(display_controls_frame, from_=0, to=200, orient='horizontal', variable=self.max_val)
        # self.max_slider.grid(row=0, column=3)

        # Start the update loops
        self.update_camera_status()
        self.update_frame_display()
        self.update_peripherals_status()

    def update_camera_status(self):
        if self.updating_camera_status:
            self.refresh_camera_status()
        self.after(1000, self.update_camera_status)  # Update camera status every second

    def update_frame_display(self):
        if self.updating_frame_display:
            self.refresh_frame_display()
        self.after(17, self.update_frame_display)  # Update frame display every ~17 ms (about 60 FPS)

    def update_peripherals_status(self):
        if self.updating_peripherals_status and self.peripherals_thread is not None:
            self.refresh_peripherals_status()
        self.after(1000, self.update_peripherals_status)  # Update peripherals status every second

    def refresh_camera_status(self):
        status_text = ""
        with self.shared_data.lock:
            for key, value in self.shared_data.camera_params.items():
                if key in DISPLAY_PARAMS.keys():
                    status_text += f"{DISPLAY_PARAMS[key]}: {value}\n"
                # status_text += f"{key}: {value}\n"
        self.camera_status.config(text=status_text)

    def refresh_frame_display(self, *_):
        try:
            frame = self.frame_queue.get_nowait()
            self.process_frame(frame)  # Display the frame using OpenCV
        except queue.Empty:
            cv2.waitKey(1)  # Keep the window interactive when no frames are available

    def refresh_peripherals_status(self):
        self.update_filter_position()

    def process_frame(self, data):
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

            # Scale data appropriately
            if data.dtype == np.uint16:
                scaled_data = np.clip((data - min_val) / (max_val - min_val) * 65535, 0, 65535).astype(np.uint16)
            elif data.dtype == np.uint8:
                scaled_data = np.clip((data - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)

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
            self.last_frame = data  # Store the last frame
        else:
            print(f"Unsupported data type: {data.dtype}")

    def on_right_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Draw Circle", command=lambda: self.draw_circle(x, y))
            menu.add_command(label="Other Option", command=lambda: print("Other Option Selected"))
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    def draw_circle(self, x, y):
        # Store the circle's position
        self.circle_center = (x, y)

        # Redraw the current frame with the circle
        if hasattr(self, 'last_frame'):
            self.process_frame(self.last_frame)

    def update_exposure_time(self, *_):
        try:
            exposure_time = float(self.exposure_time_entry.get()) / 1000
            if self.camera_thread.capturing:
                print("Cannot change exposure time during active capture.")
                self.status_message.config(text="Cannot change exposure time during active capture.")
            else:
                self.camera_thread.set_property('EXPOSURE_TIME', exposure_time)
        except ValueError:
            print("Invalid input for exposure time")

    def update_batch_size(self, *_):
        try:
            self.batch_size = int(self.cube_size_entry.get())
            print("Batch size set to ", self.batch_size, "frames per cube.")
        except:
            print("Invalid number of frames per cube. Setting to 100.")
            self.batch_size = 100

    def update_output_trigger(self, *_):
        if self.camera_thread.capturing:
            print("Cannot change output trigger during active capture.")
            self.status_message.config(text="Cannot change output trigger during active capture.")
        else:
            trigger_kind = self.output_trigger_kind_options[self.output_trigger_kind_var.get()]
            self.camera_thread.set_property('OUTPUT_TRIG_KIND_0', trigger_kind)

    def update_filter_position(self, *_):
        if self.peripherals_thread.efw is None:
            self.filter_position_menu.config(state='disabled')
            self.status_message.config(text="Filter wheel not connected.", fg="red")
        else:
            position = self.filter_position_options[self.filter_position_var.get()]
            self.peripherals_thread.efw.SetPosition(0, position)
            print(f"Filter position set to {position}.")
        return

    def change_binning(self, selected_binning):
        if self.camera_thread.capturing:
            print("Cannot change binning during active capture.")
            self.status_message.config(text="Cannot change binning during active capture.")
            # Reset to current value
            current_binning = self.binning_var.get()
            self.binning_var.set(current_binning)
        else:
            binning_value = {"1x1": 1, "2x2": 2, "4x4": 4}[selected_binning]
            self.camera_thread.set_property('BINNING', binning_value)

    def change_bit_depth(self, selected_bit_depth):
        if self.camera_thread.capturing:
            print("Cannot change bit depth during active capture.")
            self.status_message.config(text="Cannot change bit depth during active capture.")
            # Reset to current value
            current_bit_depth = self.bit_depth_var.get()
            self.bit_depth_var.set(current_bit_depth)
        else:
            bit_depth_value = {"8-bit": 1, "16-bit": 2}[selected_bit_depth]
            self.camera_thread.set_property('IMAGE_PIXEL_TYPE', bit_depth_value)

    def change_readout_speed(self, selected_mode):
        if self.camera_thread.capturing:
            print("Cannot change readout speed during active capture.")
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
            print("Cannot change sensor mode during active capture.")
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
            print("Cannot change subarray mode during active capture.")
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
            print("Cannot change subarray parameters during active capture.")
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
                print("Invalid input for subarray parameters")

    def update_framebundle(self):
        if self.camera_thread.capturing:
            print("Cannot change frame bundle during active capture.")
            self.status_message.config(text="Cannot change frame bundle during active capture.")
            # Reset to current value
            current_framebundle = self.framebundle_var.get()
            self.framebundle_var.set(current_framebundle)
        else:
            framebundle_enabled = self.framebundle_var.get()
            self.camera_thread.set_property('FRAMEBUNDLE_MODE', 2.0 if framebundle_enabled else 1.0)

    def update_frames_per_bundle(self, *_):
        if self.camera_thread.capturing:
            print("Cannot change frames per bundle during active capture.")
            self.status_message.config(text="Cannot change frames per bundle during active capture.")
        else:
            try:
                frames_per_bundle = int(self.frames_per_bundle_entry.get())
                self.camera_thread.set_property('FRAMEBUNDLE_NUMBER', frames_per_bundle)
            except ValueError:
                print("Invalid input for frames per bundle")

    def start_capture(self):
        self.status_message.config(text="Capture started...")

        if self.save_data_var.get():
            self.save_queue = queue.Queue()  # Make the queue unbounded
            self.camera_thread.save_queue = self.save_queue
            object_name = self.object_name_entry.get()
            self.save_thread = SaveThread(self.save_queue, self.timestamp_queue, self.camera_thread, object_name, self.shared_data)
            self.save_thread.batch_size = self.batch_size
            self.save_thread.start()

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

    def stop_capture(self):
        self.status_message.config(text="Capture stopped...")
        self.camera_thread.stop_capture()

        if hasattr(self, 'save_thread'):
            self.save_thread.stop()
            self.save_thread.join()

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

    def reset_camera(self):
        # Disable GUI updates
        self.updating_camera_status = False
        self.updating_frame_display = False

        # Reset camera (pausing and resuming the camera thread)
        self.camera_thread.reset_camera()

        # Re-enable GUI updates
        self.updating_camera_status = True
        self.updating_frame_display = True

    def on_close(self):
        # Stop the camera thread
        if self.camera_thread.is_alive():
            self.camera_thread.stop()

        # Stop the save thread if it exists
        if hasattr(self, 'save_thread') and self.save_thread.is_alive():
            self.save_thread.stop()
            self.save_thread.join()

        # Ensure the OpenCV window is closed
        cv2.destroyAllWindows()

        # Close the GUI window
        self.destroy()

        # Exit the application
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            self.camera_thread.join()  # Wait for the camera thread to finish

if __name__ == "__main__":
    shared_data = SharedData()
    frame_queue = queue.Queue(maxsize=3)  # Limit the size of the frame queue
    timestamp_queue = queue.Queue()

    # Create the root window first
    app = CameraGUI(shared_data, None, None, frame_queue, timestamp_queue)

    # Initialize the camera thread with a reference to the GUI
    camera_thread = CameraThread(shared_data, frame_queue, timestamp_queue, app)
    camera_thread.start()

    # Initialize peripheral devices
    peripherals_thread = PeripheralsThread(shared_data, frame_queue, timestamp_queue, app)
    peripherals_thread.start()

    app.camera_thread = camera_thread
    app.peripherals_thread = peripherals_thread

    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()

