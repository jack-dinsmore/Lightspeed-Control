import tkinter as tk
from tkinter import StringVar, OptionMenu, Checkbutton, Label, Entry, Button, Scale, Frame, LabelFrame
from dcam import Dcamapi, Dcam, DCAMERR
from camera_params import CAMERA_PARAMS
import GPS_time  # Importing the GPS_time module
import threading
import time
import numpy as np
import cv2
import queue
from astropy.io import fits
from datetime import datetime
from astropy.time import Time
import os
import json


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
        self.set_property('OUTPUT_TRIG_KIND_0', 2.0)
        self.set_property('OUTPUT_TRIG_ACTIVE_0', 1.0)
        self.set_property('OUTPUT_TRIG_POLARITY_0', 1.0)
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
            self.dcam.prop_setvalue(CAMERA_PARAMS[prop_name], value)
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
        GPS_time.clear_buffer()

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
                        self.start_time = GPS_time.get_first_timestamp()
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
    def __init__(self, save_queue, timestamp_queue, camera_thread, object_name):
        super().__init__()
        self.save_queue = save_queue
        self.timestamp_queue = timestamp_queue
        self.running = True
        self.camera_thread = camera_thread  # Reference to the camera thread to access properties
        self.object_name = object_name
        self.batch_size = 1000  # Number of frames per cube; adjust based on available memory
        self.frame_buffer = []  # Buffer to accumulate frames
        self.timestamp_buffer = []  # Buffer to accumulate timestamps
        self.framestamp_buffer = []  # Buffer to accumulate framestamps
        self.cube_index = 0  # Index to keep track of cube number

    def run(self):
        # Wait until camera_thread.start_time is available
        while self.camera_thread.start_time is None and self.running:
            time.sleep(0.1)

        if not self.running:
            return

        # Get start_time and format it
        start_time = self.camera_thread.start_time

        # For the filename, remove characters not suitable for filenames
        start_time_filename_str = start_time.strftime('%Y%m%d_%H%M%S%f')

        # Create captures directory if it doesn't exist
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
        primary_hdu.header['GPSSTART'] = (str(self.camera_thread.start_time.isot), 'GPS timestamp when data acquisition started')
        primary_hdu.header['OBJECT'] = (self.object_name, 'Object name')
        primary_hdu.header['CUBEIDX'] = (self.cube_index, 'Cube index number')

        # Stack frames into a 3D numpy array
        data_cube = np.stack(self.frame_buffer, axis=0)

        # Create ImageHDU for the data cube
        image_hdu = fits.ImageHDU(data=data_cube)
        image_hdu.header['EXTNAME'] = 'DATA_CUBE'

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


class CameraGUI(tk.Tk):
    def __init__(self, shared_data, camera_thread, frame_queue, timestamp_queue):
        super().__init__()
        self.shared_data = shared_data
        self.camera_thread = camera_thread
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.updating_camera_status = True  # Flag for camera status update
        self.updating_frame_display = True  # Flag for frame display update
        self.corner_size = 100  # Size of corner regions to display
        
        # Frame averaging variables
        self.averaging_enabled = False
        self.frames_to_average = 10
        self.frame_buffer_for_averaging = []
        
        # Settings file path
        self.settings_file = "camera_settings.json"

        # Initialize scaling variables for each corner and center
        self.min_val_tl = tk.IntVar(value=0)  # Top-left min
        self.max_val_tl = tk.IntVar(value=65000)  # Top-left max
        self.min_val_tr = tk.IntVar(value=0)  # Top-right min
        self.max_val_tr = tk.IntVar(value=65000)  # Top-right max
        self.min_val_bl = tk.IntVar(value=0)  # Bottom-left min
        self.max_val_bl = tk.IntVar(value=65000)  # Bottom-left max
        self.min_val_br = tk.IntVar(value=0)  # Bottom-right min
        self.max_val_br = tk.IntVar(value=65000)  # Bottom-right max
        self.min_val_c = tk.IntVar(value=0)  # Center min
        self.max_val_c = tk.IntVar(value=65000)  # Center max

        self.title("Camera Parameters")
        self.geometry("1000x1400")  # Adjust window size to tighten the layout

        # GUI code snippet
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)

        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # LabelFrame for Camera Parameters
        camera_params_frame = LabelFrame(self.main_frame, text="Camera Parameters", padx=5, pady=5)
        camera_params_frame.grid(row=0, column=0, rowspan=5, sticky='nsew')
        self.camera_status = tk.Label(camera_params_frame, text="", justify=tk.LEFT, anchor="w", width=60, height=80, wraplength=400)
        self.camera_status.pack(fill='both', expand=True)

        # Label for status messages - move it to the top so it's defined early
        self.status_message = tk.Label(self.main_frame, text="", justify=tk.LEFT, anchor="w", width=40, wraplength=400, fg="blue")
        self.status_message.grid(row=5, column=0, columnspan=2, sticky='nsew')

        # Camera Controls
        camera_controls_frame = LabelFrame(self.main_frame, text="Camera Controls", padx=5, pady=5)
        camera_controls_frame.grid(row=0, column=1, sticky='n')

        Label(camera_controls_frame, text="Exposure Time (ms):").grid(row=0, column=0)
        self.exposure_time_entry = Entry(camera_controls_frame)
        self.exposure_time_entry.insert(0, "65000")
        self.exposure_time_entry.grid(row=0, column=1)
        self.exposure_time_entry.bind("<Return>", self.update_exposure_time)

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
        self.subarray_hpos_entry = Entry(subarray_controls_frame)
        self.subarray_hpos_entry.grid(row=1, column=1)
        self.subarray_hpos_entry.insert(0, "0.0")  # Insert default value here
        self.subarray_hpos_entry.config(state='disabled')  # Disable after inserting value
        self.subarray_hpos_entry.bind("<Return>", self.update_subarray)
        self.subarray_hpos_entry.bind("<FocusOut>", self.update_subarray)

        Label(subarray_controls_frame, text="Subarray HSIZE:").grid(row=2, column=0)
        self.subarray_hsize_entry = Entry(subarray_controls_frame)
        self.subarray_hsize_entry.grid(row=2, column=1)
        self.subarray_hsize_entry.insert(0, "4096.0")  # Insert default value here
        self.subarray_hsize_entry.config(state='disabled')  # Disable after inserting value
        self.subarray_hsize_entry.bind("<Return>", self.update_subarray)
        self.subarray_hsize_entry.bind("<FocusOut>", self.update_subarray)

        Label(subarray_controls_frame, text="Subarray VPOS:").grid(row=3, column=0)
        self.subarray_vpos_entry = Entry(subarray_controls_frame)
        self.subarray_vpos_entry.grid(row=3, column=1)
        self.subarray_vpos_entry.insert(0, "0.0")  # Insert default value here
        self.subarray_vpos_entry.config(state='disabled')  # Disable after inserting value
        self.subarray_vpos_entry.bind("<Return>", self.update_subarray)
        self.subarray_vpos_entry.bind("<FocusOut>", self.update_subarray)

        Label(subarray_controls_frame, text="Subarray VSIZE:").grid(row=4, column=0)
        self.subarray_vsize_entry = Entry(subarray_controls_frame)
        self.subarray_vsize_entry.grid(row=4, column=1)
        self.subarray_vsize_entry.insert(0, "2304.0")  # Insert default value here
        self.subarray_vsize_entry.config(state='disabled')  # Disable after inserting value
        self.subarray_vsize_entry.bind("<Return>", self.update_subarray)
        self.subarray_vsize_entry.bind("<FocusOut>", self.update_subarray)

        # Advanced Controls
        advanced_controls_frame = LabelFrame(self.main_frame, text="Advanced Controls", padx=5, pady=5)
        advanced_controls_frame.grid(row=3, column=1, sticky='n')

        self.framebundle_var = tk.BooleanVar()
        self.framebundle_checkbox = Checkbutton(advanced_controls_frame, text="Enable Frame Bundle", variable=self.framebundle_var, command=self.update_framebundle)
        self.framebundle_checkbox.grid(row=0, column=0, columnspan=2)

        Label(advanced_controls_frame, text="Frames Per Bundle:").grid(row=1, column=0)
        self.frames_per_bundle_entry = Entry(advanced_controls_frame)
        self.frames_per_bundle_entry.insert(0, "100")
        self.frames_per_bundle_entry.grid(row=1, column=1)
        self.frames_per_bundle_entry.bind("<Return>", self.update_frames_per_bundle)
        self.frames_per_bundle_entry.bind("<FocusOut>", self.update_frames_per_bundle)

        # Display Controls with individual scaling for each corner and center
        display_controls_frame = LabelFrame(self.main_frame, text="Display Controls", padx=5, pady=5)
        display_controls_frame.grid(row=4, column=1, sticky='n')
        
        # Frame averaging controls
        averaging_frame = Frame(display_controls_frame)
        averaging_frame.grid(row=0, column=0, columnspan=5, pady=5)
        
        self.averaging_var = tk.BooleanVar()
        self.averaging_checkbox = Checkbutton(averaging_frame, text="Enable Averaging", variable=self.averaging_var, command=self.toggle_averaging)
        self.averaging_checkbox.grid(row=0, column=0)
        
        Label(averaging_frame, text="Frames to Average:").grid(row=0, column=1, padx=(10, 5))
        self.averaging_entry = Entry(averaging_frame, width=8)
        self.averaging_entry.insert(0, "10")
        self.averaging_entry.grid(row=0, column=2)
        self.averaging_entry.bind("<Return>", self.update_averaging_count)
        self.averaging_entry.bind("<FocusOut>", self.update_averaging_count)
        
        # Separator
        tk.Frame(display_controls_frame, height=2, bd=1, relief='sunken').grid(row=1, column=0, columnspan=5, sticky='ew', pady=5)

        # Top-left corner controls
        Label(display_controls_frame, text="Top-Left:").grid(row=2, column=0, sticky='w')
        Label(display_controls_frame, text="Min:").grid(row=2, column=1)
        self.min_entry_tl = Entry(display_controls_frame, width=8)
        self.min_entry_tl.insert(0, "0")
        self.min_entry_tl.grid(row=2, column=2)
        self.min_entry_tl.bind("<Return>", lambda e: self.update_corner_range('tl'))
        self.min_entry_tl.bind("<FocusOut>", lambda e: self.update_corner_range('tl'))
        
        Label(display_controls_frame, text="Max:").grid(row=2, column=3)
        self.max_entry_tl = Entry(display_controls_frame, width=8)
        self.max_entry_tl.insert(0, "65000")
        self.max_entry_tl.grid(row=2, column=4)
        self.max_entry_tl.bind("<Return>", lambda e: self.update_corner_range('tl'))
        self.max_entry_tl.bind("<FocusOut>", lambda e: self.update_corner_range('tl'))

        # Top-right corner controls
        Label(display_controls_frame, text="Top-Right:").grid(row=3, column=0, sticky='w')
        Label(display_controls_frame, text="Min:").grid(row=3, column=1)
        self.min_entry_tr = Entry(display_controls_frame, width=8)
        self.min_entry_tr.insert(0, "0")
        self.min_entry_tr.grid(row=3, column=2)
        self.min_entry_tr.bind("<Return>", lambda e: self.update_corner_range('tr'))
        self.min_entry_tr.bind("<FocusOut>", lambda e: self.update_corner_range('tr'))
        
        Label(display_controls_frame, text="Max:").grid(row=3, column=3)
        self.max_entry_tr = Entry(display_controls_frame, width=8)
        self.max_entry_tr.insert(0, "65000")
        self.max_entry_tr.grid(row=3, column=4)
        self.max_entry_tr.bind("<Return>", lambda e: self.update_corner_range('tr'))
        self.max_entry_tr.bind("<FocusOut>", lambda e: self.update_corner_range('tr'))

        # Bottom-left corner controls
        Label(display_controls_frame, text="Bottom-Left:").grid(row=4, column=0, sticky='w')
        Label(display_controls_frame, text="Min:").grid(row=4, column=1)
        self.min_entry_bl = Entry(display_controls_frame, width=8)
        self.min_entry_bl.insert(0, "0")
        self.min_entry_bl.grid(row=4, column=2)
        self.min_entry_bl.bind("<Return>", lambda e: self.update_corner_range('bl'))
        self.min_entry_bl.bind("<FocusOut>", lambda e: self.update_corner_range('bl'))
        
        Label(display_controls_frame, text="Max:").grid(row=4, column=3)
        self.max_entry_bl = Entry(display_controls_frame, width=8)
        self.max_entry_bl.insert(0, "65000")
        self.max_entry_bl.grid(row=4, column=4)
        self.max_entry_bl.bind("<Return>", lambda e: self.update_corner_range('bl'))
        self.max_entry_bl.bind("<FocusOut>", lambda e: self.update_corner_range('bl'))

        # Bottom-right corner controls
        Label(display_controls_frame, text="Bottom-Right:").grid(row=5, column=0, sticky='w')
        Label(display_controls_frame, text="Min:").grid(row=5, column=1)
        self.min_entry_br = Entry(display_controls_frame, width=8)
        self.min_entry_br.insert(0, "0")
        self.min_entry_br.grid(row=5, column=2)
        self.min_entry_br.bind("<Return>", lambda e: self.update_corner_range('br'))
        self.min_entry_br.bind("<FocusOut>", lambda e: self.update_corner_range('br'))
        
        Label(display_controls_frame, text="Max:").grid(row=5, column=3)
        self.max_entry_br = Entry(display_controls_frame, width=8)
        self.max_entry_br.insert(0, "65000")
        self.max_entry_br.grid(row=5, column=4)
        self.max_entry_br.bind("<Return>", lambda e: self.update_corner_range('br'))
        self.max_entry_br.bind("<FocusOut>", lambda e: self.update_corner_range('br'))

        # Center controls
        Label(display_controls_frame, text="Center:").grid(row=6, column=0, sticky='w')
        Label(display_controls_frame, text="Min:").grid(row=6, column=1)
        self.min_entry_c = Entry(display_controls_frame, width=8)
        self.min_entry_c.insert(0, "0")
        self.min_entry_c.grid(row=6, column=2)
        self.min_entry_c.bind("<Return>", lambda e: self.update_corner_range('c'))
        self.min_entry_c.bind("<FocusOut>", lambda e: self.update_corner_range('c'))
        
        Label(display_controls_frame, text="Max:").grid(row=6, column=3)
        self.max_entry_c = Entry(display_controls_frame, width=8)
        self.max_entry_c.insert(0, "65000")
        self.max_entry_c.grid(row=6, column=4)
        self.max_entry_c.bind("<Return>", lambda e: self.update_corner_range('c'))
        self.max_entry_c.bind("<FocusOut>", lambda e: self.update_corner_range('c'))

        # Apply same scale to all button
        self.apply_all_button = Button(display_controls_frame, text="Apply Top-Left to All", command=self.apply_scale_to_all)
        self.apply_all_button.grid(row=7, column=0, columnspan=5, pady=5)

        # Corner size control
        Label(display_controls_frame, text="Corner Size (pixels):").grid(row=8, column=0, columnspan=2)
        self.corner_size_var = tk.IntVar(value=100)
        self.corner_size_slider = Scale(display_controls_frame, from_=50, to=1000, orient='horizontal', variable=self.corner_size_var, command=self.update_corner_size)
        self.corner_size_slider.grid(row=8, column=2, columnspan=3)

        # Load saved settings if they exist
        self.load_settings()

        # Start the update loops
        self.update_camera_status()
        self.update_frame_display()

    def save_settings(self):
        """Save current settings to a JSON file."""
        settings = {
            # Display settings
            'min_val_tl': self.min_val_tl.get(),
            'max_val_tl': self.max_val_tl.get(),
            'min_val_tr': self.min_val_tr.get(),
            'max_val_tr': self.max_val_tr.get(),
            'min_val_bl': self.min_val_bl.get(),
            'max_val_bl': self.max_val_bl.get(),
            'min_val_br': self.min_val_br.get(),
            'max_val_br': self.max_val_br.get(),
            'min_val_c': self.min_val_c.get(),
            'max_val_c': self.max_val_c.get(),
            'corner_size': self.corner_size_var.get(),
            
            # Camera settings
            'exposure_time': self.exposure_time_entry.get(),
            'binning': self.binning_var.get(),
            'bit_depth': self.bit_depth_var.get(),
            'readout_speed': self.readout_speed_var.get(),
            'sensor_mode': self.sensor_mode_var.get(),
            
            # Subarray settings
            'subarray_mode': self.subarray_mode_var.get(),
            'subarray_hpos': self.subarray_hpos_entry.get(),
            'subarray_hsize': self.subarray_hsize_entry.get(),
            'subarray_vpos': self.subarray_vpos_entry.get(),
            'subarray_vsize': self.subarray_vsize_entry.get(),
            
            # Frame bundle settings
            'framebundle_enabled': self.framebundle_var.get(),
            'frames_per_bundle': self.frames_per_bundle_entry.get(),
            
            # Averaging settings
            'averaging_enabled': self.averaging_var.get(),
            'frames_to_average': self.averaging_entry.get(),
            
            # Object name
            'object_name': self.object_name_entry.get()
        }
        
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"Settings saved to {self.settings_file}")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        """Load settings from JSON file if it exists."""
        if not os.path.exists(self.settings_file):
            print(f"No settings file found at {self.settings_file}")
            return
        
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            
            # Load display settings
            self.min_val_tl.set(settings.get('min_val_tl', 0))
            self.max_val_tl.set(settings.get('max_val_tl', 65000))
            self.min_val_tr.set(settings.get('min_val_tr', 0))
            self.max_val_tr.set(settings.get('max_val_tr', 65000))
            self.min_val_bl.set(settings.get('min_val_bl', 0))
            self.max_val_bl.set(settings.get('max_val_bl', 65000))
            self.min_val_br.set(settings.get('min_val_br', 0))
            self.max_val_br.set(settings.get('max_val_br', 65000))
            self.min_val_c.set(settings.get('min_val_c', 0))
            self.max_val_c.set(settings.get('max_val_c', 65000))
            self.corner_size_var.set(settings.get('corner_size', 100))
            self.corner_size = settings.get('corner_size', 100)  # Also set the actual corner_size
            
            # Update entry fields for display settings
            self.min_entry_tl.delete(0, tk.END)
            self.min_entry_tl.insert(0, str(self.min_val_tl.get()))
            self.max_entry_tl.delete(0, tk.END)
            self.max_entry_tl.insert(0, str(self.max_val_tl.get()))
            
            self.min_entry_tr.delete(0, tk.END)
            self.min_entry_tr.insert(0, str(self.min_val_tr.get()))
            self.max_entry_tr.delete(0, tk.END)
            self.max_entry_tr.insert(0, str(self.max_val_tr.get()))
            
            self.min_entry_bl.delete(0, tk.END)
            self.min_entry_bl.insert(0, str(self.min_val_bl.get()))
            self.max_entry_bl.delete(0, tk.END)
            self.max_entry_bl.insert(0, str(self.max_val_bl.get()))
            
            self.min_entry_br.delete(0, tk.END)
            self.min_entry_br.insert(0, str(self.min_val_br.get()))
            self.max_entry_br.delete(0, tk.END)
            self.max_entry_br.insert(0, str(self.max_val_br.get()))
            
            self.min_entry_c.delete(0, tk.END)
            self.min_entry_c.insert(0, str(self.min_val_c.get()))
            self.max_entry_c.delete(0, tk.END)
            self.max_entry_c.insert(0, str(self.max_val_c.get()))
            
            # Load camera settings
            self.exposure_time_entry.delete(0, tk.END)
            self.exposure_time_entry.insert(0, settings.get('exposure_time', '65000'))
            self.binning_var.set(settings.get('binning', '1x1'))
            self.bit_depth_var.set(settings.get('bit_depth', '8-bit'))
            self.readout_speed_var.set(settings.get('readout_speed', 'Ultra Quiet Mode'))
            self.sensor_mode_var.set(settings.get('sensor_mode', 'Photon Number Resolving'))
            
            # Load subarray settings
            self.subarray_mode_var.set(settings.get('subarray_mode', 'Off'))
            self.subarray_hpos_entry.delete(0, tk.END)
            self.subarray_hpos_entry.insert(0, settings.get('subarray_hpos', '0.0'))
            self.subarray_hsize_entry.delete(0, tk.END)
            self.subarray_hsize_entry.insert(0, settings.get('subarray_hsize', '4096.0'))
            self.subarray_vpos_entry.delete(0, tk.END)
            self.subarray_vpos_entry.insert(0, settings.get('subarray_vpos', '0.0'))
            self.subarray_vsize_entry.delete(0, tk.END)
            self.subarray_vsize_entry.insert(0, settings.get('subarray_vsize', '2304.0'))
            
            # Load frame bundle settings
            self.framebundle_var.set(settings.get('framebundle_enabled', False))
            self.frames_per_bundle_entry.delete(0, tk.END)
            self.frames_per_bundle_entry.insert(0, settings.get('frames_per_bundle', '100'))
            
            # Load averaging settings
            self.averaging_var.set(settings.get('averaging_enabled', False))
            self.averaging_enabled = settings.get('averaging_enabled', False)  # Set the actual flag
            self.averaging_entry.delete(0, tk.END)
            self.averaging_entry.insert(0, settings.get('frames_to_average', '10'))
            self.frames_to_average = int(settings.get('frames_to_average', '10'))  # Set the actual value
            
            # Load object name
            self.object_name_entry.delete(0, tk.END)
            self.object_name_entry.insert(0, settings.get('object_name', ''))
            
            print(f"Settings loaded from {self.settings_file}")
            
            # Schedule camera settings application after camera is initialized
            self.after(1000, self.apply_camera_settings_from_load)
            
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def apply_camera_settings_from_load(self):
        """Apply camera settings after the camera thread is initialized."""
        if self.camera_thread and self.camera_thread.dcam:
            try:
                # Apply exposure time
                exposure_time = float(self.exposure_time_entry.get()) / 1000
                self.camera_thread.set_property('EXPOSURE_TIME', exposure_time)
                
                # Apply binning
                binning_value = {"1x1": 1, "2x2": 2, "4x4": 4}[self.binning_var.get()]
                self.camera_thread.set_property('BINNING', binning_value)
                
                # Apply bit depth
                bit_depth_value = {"8-bit": 1, "16-bit": 2}[self.bit_depth_var.get()]
                self.camera_thread.set_property('IMAGE_PIXEL_TYPE', bit_depth_value)
                
                # Apply readout speed
                readout_speed_value = {"Ultra Quiet Mode": 1.0, "Standard Mode": 2.0}[self.readout_speed_var.get()]
                self.camera_thread.set_property('READOUT_SPEED', readout_speed_value)
                
                # Apply sensor mode
                sensor_mode_value = {"Photon Number Resolving": 18.0, "Standard": 1.0}[self.sensor_mode_var.get()]
                self.camera_thread.set_property('SENSOR_MODE', sensor_mode_value)
                
                # Apply subarray settings if enabled
                if self.subarray_mode_var.get() == "On":
                    self.camera_thread.set_property('SUBARRAY_MODE', 2.0)
                    self.camera_thread.set_property('SUBARRAY_HPOS', float(self.subarray_hpos_entry.get()))
                    self.camera_thread.set_property('SUBARRAY_HSIZE', float(self.subarray_hsize_entry.get()))
                    self.camera_thread.set_property('SUBARRAY_VPOS', float(self.subarray_vpos_entry.get()))
                    self.camera_thread.set_property('SUBARRAY_VSIZE', float(self.subarray_vsize_entry.get()))
                    # Enable the entry fields
                    self.subarray_hpos_entry.config(state='normal')
                    self.subarray_hsize_entry.config(state='normal')
                    self.subarray_vpos_entry.config(state='normal')
                    self.subarray_vsize_entry.config(state='normal')
                else:
                    self.camera_thread.set_property('SUBARRAY_MODE', 1.0)
                
                # Apply frame bundle settings
                if self.framebundle_var.get():
                    self.camera_thread.set_property('FRAMEBUNDLE_MODE', 2.0)
                    self.camera_thread.set_property('FRAMEBUNDLE_NUMBER', int(self.frames_per_bundle_entry.get()))
                else:
                    self.camera_thread.set_property('FRAMEBUNDLE_MODE', 1.0)
                
                print("Camera settings applied from saved configuration")
                
            except Exception as e:
                print(f"Error applying camera settings: {e}")
                # If camera not ready yet, try again in a second
                self.after(1000, self.apply_camera_settings_from_load)
        else:
            # Camera not ready yet, try again in a second
            self.after(1000, self.apply_camera_settings_from_load)

    def toggle_averaging(self):
        """Toggle frame averaging on/off."""
        self.averaging_enabled = self.averaging_var.get()
        if not self.averaging_enabled:
            # Clear the buffer when disabling averaging
            self.frame_buffer_for_averaging.clear()
        print(f"Frame averaging {'enabled' if self.averaging_enabled else 'disabled'}")
        self.save_settings()

    def update_averaging_count(self, event=None):
        """Update the number of frames to average."""
        try:
            new_count = int(self.averaging_entry.get())
            if new_count > 0:
                self.frames_to_average = new_count
                # Clear buffer when changing average count
                self.frame_buffer_for_averaging.clear()
                print(f"Averaging count set to {self.frames_to_average}")
                self.save_settings()
            else:
                print("Averaging count must be positive")
                self.averaging_entry.delete(0, tk.END)
                self.averaging_entry.insert(0, str(self.frames_to_average))
        except ValueError:
            print("Invalid input for averaging count")
            self.averaging_entry.delete(0, tk.END)
            self.averaging_entry.insert(0, str(self.frames_to_average))

    def update_corner_range(self, corner):
        """Update the display range for a specific corner or center."""
        try:
            if corner == 'tl':
                min_value = int(self.min_entry_tl.get())
                max_value = int(self.max_entry_tl.get())
                self.min_val_tl.set(min_value)
                self.max_val_tl.set(max_value)
            elif corner == 'tr':
                min_value = int(self.min_entry_tr.get())
                max_value = int(self.max_entry_tr.get())
                self.min_val_tr.set(min_value)
                self.max_val_tr.set(max_value)
            elif corner == 'bl':
                min_value = int(self.min_entry_bl.get())
                max_value = int(self.max_entry_bl.get())
                self.min_val_bl.set(min_value)
                self.max_val_bl.set(max_value)
            elif corner == 'br':
                min_value = int(self.min_entry_br.get())
                max_value = int(self.max_entry_br.get())
                self.min_val_br.set(min_value)
                self.max_val_br.set(max_value)
            elif corner == 'c':
                min_value = int(self.min_entry_c.get())
                max_value = int(self.max_entry_c.get())
                self.min_val_c.set(min_value)
                self.max_val_c.set(max_value)
            
            # Save settings after update
            self.save_settings()
            
        except ValueError:
            print(f"Invalid input for {corner} display range values")

    def apply_scale_to_all(self):
        """Apply the top-left corner's scale to all corners and center."""
        try:
            min_value = self.min_val_tl.get()
            max_value = self.max_val_tl.get()
            
            # Update all corners and center with the same values
            for corner, min_entry, max_entry, min_var, max_var in [
                ('tr', self.min_entry_tr, self.max_entry_tr, self.min_val_tr, self.max_val_tr),
                ('bl', self.min_entry_bl, self.max_entry_bl, self.min_val_bl, self.max_val_bl),
                ('br', self.min_entry_br, self.max_entry_br, self.min_val_br, self.max_val_br),
                ('c', self.min_entry_c, self.max_entry_c, self.min_val_c, self.max_val_c)
            ]:
                min_entry.delete(0, tk.END)
                min_entry.insert(0, str(min_value))
                max_entry.delete(0, tk.END)
                max_entry.insert(0, str(max_value))
                min_var.set(min_value)
                max_var.set(max_value)
        except Exception as e:
            print(f"Error applying scale to all regions: {e}")

    def update_corner_size(self, value):
        self.corner_size = int(value)
        self.save_settings()

    def update_camera_status(self):
        if self.updating_camera_status:
            self.refresh_camera_status()
        self.after(1000, self.update_camera_status)  # Update camera status every second

    def update_frame_display(self):
        if self.updating_frame_display:
            self.refresh_frame_display()
        self.after(17, self.update_frame_display)  # Update frame display every ~17 ms (about 60 FPS)

    def refresh_camera_status(self):
        status_text = ""
        with self.shared_data.lock:
            for key, value in self.shared_data.camera_params.items():
                status_text += f"{key}: {value}\n"
        self.camera_status.config(text=status_text)

    def refresh_frame_display(self):
        try:
            frame = self.frame_queue.get_nowait()
            
            # Handle frame averaging if enabled
            if self.averaging_enabled:
                # Add frame to buffer
                self.frame_buffer_for_averaging.append(frame.copy())
                
                # Keep only the last N frames
                if len(self.frame_buffer_for_averaging) > self.frames_to_average:
                    self.frame_buffer_for_averaging.pop(0)
                
                # If we have enough frames, compute the average
                if len(self.frame_buffer_for_averaging) >= self.frames_to_average:
                    # Stack frames and compute mean
                    stacked_frames = np.stack(self.frame_buffer_for_averaging, axis=0)
                    averaged_frame = np.mean(stacked_frames, axis=0)
                    
                    # Preserve the original dtype
                    if frame.dtype == np.uint16:
                        averaged_frame = averaged_frame.astype(np.uint16)
                    else:
                        averaged_frame = averaged_frame.astype(np.uint8)
                    
                    self.process_frame(averaged_frame)
                else:
                    # Not enough frames yet, display the current frame
                    self.process_frame(frame)
            else:
                # No averaging, display frame directly
                self.process_frame(frame)
                
        except queue.Empty:
            cv2.waitKey(1)  # Keep the window interactive when no frames are available

    def extract_corners_and_center(self, data):
        """Extract four corner regions and center from the image."""
        h, w = data.shape
        corner_size = min(self.corner_size, h//2, w//2)  # Ensure corner size doesn't exceed half the image
        
        # Extract corners
        top_left = data[:corner_size, :corner_size]
        top_right = data[:corner_size, -corner_size:]
        bottom_left = data[-corner_size:, :corner_size]
        bottom_right = data[-corner_size:, -corner_size:]
        
        # Extract center
        center_h = h // 2
        center_w = w // 2
        half_size = corner_size // 2
        center = data[center_h - half_size:center_h + half_size, 
                     center_w - half_size:center_w + half_size]
        
        return top_left, top_right, bottom_left, bottom_right, center

    def create_corner_display(self, regions):
        """Create a display with center on left and four corners forming a square on right."""
        top_left, top_right, bottom_left, bottom_right, center = regions
        
        # Create spacer between regions - reduced for better use of space
        spacer_h = 2  # Horizontal spacer width
        spacer_v = 2  # Vertical spacer height
        
        # Create the four corners as a 2x2 grid
        corners_top = np.hstack([
            top_left,
            np.zeros((top_left.shape[0], spacer_h), dtype=top_left.dtype),
            top_right
        ])
        
        corners_bottom = np.hstack([
            bottom_left,
            np.zeros((bottom_left.shape[0], spacer_h), dtype=bottom_left.dtype),
            bottom_right
        ])
        
        corners_v_spacer = np.zeros((spacer_v, corners_top.shape[1]), dtype=top_left.dtype)
        corners_combined = np.vstack([corners_top, corners_v_spacer, corners_bottom])
        
        # Scale the center image to match the height of the corners grid
        center_height = corners_combined.shape[0]
        center_width = int(center.shape[1] * (center_height / center.shape[0]))
        center_resized = cv2.resize(center, (center_width, center_height), interpolation=cv2.INTER_NEAREST)
        
        # Add spacer between center and corners
        spacer_middle = np.zeros((center_height, spacer_h * 5), dtype=top_left.dtype)
        
        # Combine center on left with corners on right
        combined = np.hstack([center_resized, spacer_middle, corners_combined])
        
        return combined

    def process_frame(self, data):
        # Handle both 8-bit and 16-bit data
        if data.dtype == np.uint16 or data.dtype == np.uint8:
            # Extract corners and center
            regions = self.extract_corners_and_center(data)
            top_left, top_right, bottom_left, bottom_right, center = regions
            
            # Apply individual scaling to each region
            scaled_regions = []
            
            # Scale each region with its own min/max values
            region_configs = [
                (top_left, self.min_val_tl.get(), self.max_val_tl.get()),
                (top_right, self.min_val_tr.get(), self.max_val_tr.get()),
                (bottom_left, self.min_val_bl.get(), self.max_val_bl.get()),
                (bottom_right, self.min_val_br.get(), self.max_val_br.get()),
                (center, self.min_val_c.get(), self.max_val_c.get())
            ]
            
            for region_data, min_val, max_val in region_configs:
                if data.dtype == np.uint16:
                    scaled = np.clip((region_data - min_val) / (max_val - min_val) * 65535, 0, 65535).astype(np.uint16)
                else:
                    scaled = np.clip((region_data - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                scaled_regions.append(scaled)
            
            # Create combined display
            combined_display = self.create_corner_display(scaled_regions)
            
            # Convert to BGR for display
            if combined_display.dtype == np.uint16:
                # Convert 16-bit to 8-bit for display
                combined_display_8bit = (combined_display / 256).astype(np.uint8)
                combined_display_bgr = cv2.cvtColor(combined_display_8bit, cv2.COLOR_GRAY2BGR)
            else:
                combined_display_bgr = cv2.cvtColor(combined_display, cv2.COLOR_GRAY2BGR)

            # Create or update OpenCV window
            if not hasattr(self, 'opencv_window_created'):
                cv2.namedWindow('Camera Corners Display', cv2.WINDOW_NORMAL)
                # Set initial window size to be larger
                cv2.resizeWindow('Camera Corners Display', 800, 800)
                self.opencv_window_created = True

            cv2.imshow('Camera Corners Display', combined_display_bgr)
            
            # Get current window size and resize image to fit
            # This allows the image to scale with the window
            window_rect = cv2.getWindowImageRect('Camera Corners Display')
            if window_rect[2] > 0 and window_rect[3] > 0:  # width and height are valid
                # Calculate scaling to fit window while maintaining aspect ratio
                h, w = combined_display_bgr.shape[:2]
                scale = min(window_rect[2] / w, window_rect[3] / h)
                if scale > 1.0:  # Only upscale if window is larger
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_display = cv2.resize(combined_display_bgr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow('Camera Corners Display', resized_display)
            
            cv2.waitKey(1)
            self.last_frame = data  # Store the last frame
        else:
            print(f"Unsupported data type: {data.dtype}")

    def update_exposure_time(self, event=None):
        try:
            exposure_time = float(self.exposure_time_entry.get()) / 1000
            if self.camera_thread.capturing:
                print("Cannot change exposure time during active capture.")
                self.status_message.config(text="Cannot change exposure time during active capture.")
            else:
                self.camera_thread.set_property('EXPOSURE_TIME', exposure_time)
                self.save_settings()
        except ValueError:
            print("Invalid input for exposure time")

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

    def update_subarray(self, event=None):
        if self.camera_thread.capturing:
            print("Cannot change subarray parameters during active capture.")
            self.status_message.config(text="Cannot change subarray parameters during active capture.")
        else:
            try:
                hpos = float(self.subarray_hpos_entry.get())
                hsize = float(self.subarray_hsize_entry.get())
                vpos = float(self.subarray_vpos_entry.get())
                vsize = float(self.subarray_vsize_entry.get())

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

    def update_frames_per_bundle(self, event=None):
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
            self.save_thread = SaveThread(self.save_queue, self.timestamp_queue, self.camera_thread, object_name)
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
        # Save settings before closing
        self.save_settings()
        
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
    app = CameraGUI(shared_data, None, frame_queue, timestamp_queue)

    # Initialize the camera thread with a reference to the GUI
    camera_thread = CameraThread(shared_data, frame_queue, timestamp_queue, app)
    camera_thread.start()

    app.camera_thread = camera_thread

    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
