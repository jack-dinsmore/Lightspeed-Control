import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve
import tkinter as tk
from tkinter import Label, Entry, Button, Scale, LabelFrame, Checkbutton
import threading
from astropy.time import Time
# import keyboard
import cv2
import queue

def get_tk_value(tk_value):
    try:
        return tk_value.get()
    except tk.TclError:
        if type(tk_value) is tk.IntVar:
            return 0
        else:
            return np.nan

class RollingBuffer:
    def __init__(self, limit, image_size):
        self.image_size = image_size
        self.data = np.zeros((limit, *image_size))
        self.data_index = 0
        self.data_valid = np.zeros(limit, bool) # Start with every data set not being valid

        chunk_limit = limit//50 + 1 # The number of images to stack into a chunk before adding the chunk to the buffer
        self.current_chunk = np.zeros((chunk_limit, *image_size))
        self.chunk_index = 0
        # No need to create a chunk_valid variable. Everything after the chunk index is invalid.

    def push(self, image):
        if len(self.current_chunk) <= 1:
            self.data[self.data_index] = image
            self.data_valid[self.data_index] = True
            self.data_index += 1
        else:
            # Add the image to the chunk
            self.current_chunk[self.chunk_index] = image
            self.chunk_index += 1
            if self.chunk_index >= len(self.current_chunk):
                # If the chunk is done, add it to the data
                self.chunk_index = 0
                self.data[self.data_index] = np.sum(self.current_chunk, axis=0)
                self.data_valid[self.data_index] = True
                self.data_index += 1

        # Loop the data index
        self.data_index = self.data_index % len(self.data)

    def get(self):
        output = np.sum(self.data[self.data_valid], axis=0)
        if len(self.current_chunk) > 1:
            output += np.sum(self.current_chunk[:self.chunk_index], axis=0)
        return output

    def clear(self):
        self.data_index = 0
        self.data_valid &= False # Set all data to invalid

    def extend(self, new_limit):
        if new_limit == len(self.data):
            # The new limit is the same as the old limit. Do nothing
            return
        
        # Commit the current chunk to data
        self.data[self.data_index] += np.sum(self.current_chunk, axis=0)
        self.data_index += 1
        self.data_index = self.data_index % len(self.data)

        # Extend the current chunk array
        chunk_limit = new_limit//50 + 1 # The number of images to stack into a chunk before adding the chunk to the buffer
        self.current_chunk = np.zeros((chunk_limit, *self.image_size))
        self.chunk_index = 0

        # Create the new data storage
        new_data = np.zeros((new_limit, *self.image_size))
        new_data_valid = np.zeros(new_limit, bool)
        new_data_index = 0
        
        # Iterate through all the currently stored data and save all the valid options
        for _ in range(len(self.data)):
            if self.data_valid[self.data_index]:
                new_data[new_data_index] = self.data[self.data_index]
                new_data_valid[new_data_index] = True
                new_data_index += 1
                new_data_index = new_data_index % len(new_data)
            self.data_index += 1
            self.data_index = self.data_index % len(self.data)
        
        # Overwrite the old data
        self.data_index = new_data_index
        self.data = new_data
        self.data_valid = new_data_valid

class PhaseRangePlot:
    def __init__(self, ax, color, label):
        self.span1 = ax.axvspan(-1, -0.5, color=color, label=label, alpha=0.3)
        self.span2 = ax.axvspan(-1, -0.5, color=color, alpha=0.3)

    def update(self, phase_range):
        if phase_range is None:
            self.span1.set_x(-1)
            self.span1.set_width(0.5)
            self.span2.set_x(-1)
            self.span2.set_width(0.5)
        elif phase_range[1] >= phase_range[0]:
            self.span1.set_x(phase_range[0])
            self.span1.set_width(phase_range[1] - phase_range[0])
            self.span2.set_x(-1)
            self.span2.set_width(0.5)
        else:
            self.span1.set_x(0)
            self.span1.set_width(phase_range[1])
            self.span2.set_x(phase_range[0])
            self.span2.set_width(1 - phase_range[0])
        
def check_phase(phase, phase_range):
    """
    Returns a bool indicating whether phase is inside phase_range.
    """
    if phase_range[0] < phase_range[1]:
        return (phase_range[0] < phase) and (phase < phase_range[1])
    else:
        return (phase_range[0] > phase) or (phase > phase_range[1])
    
def get_phase_duration(phase_range):
    """
    Returns the length of the phase bin
    """
    if phase_range[0] < phase_range[1]:
        return phase_range[1] - phase_range[0]
    else:
        return 1 + phase_range[1] - phase_range[0]

class PhaseGUI(tk.Tk):
    def __init__(self, frame_queue, timestamp_queue, feed):
        super().__init__()
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.roi_moved = None
        self.ranges_moved = None

        self.roi_center = None # Center of the ROI (pix)
        self.roi_width = None # Full width of the square ROI (pix)
        self.on_range = None # On phase range
        self.off_range = None # Off phase range
        self.temporary_range = None # Temporary range used by the UI when setting a new range

        self.on_image = None # To be initialized later
        self.off_image = None
        self.lc_fluxes = None
        self.lc_phase_bin_edges = None

        self.lc_window_created = False
        self.on_window_created = False
        self.off_window_created = False

        # Set up GUI main frame
        self.title("Lightspeed Phasing GUI")
        self.geometry("350x450")  # Adjust window size to tighten the layout
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)

        # LabelFrame for ephemeris
        ephemeris_frame = LabelFrame(self.main_frame, text="Ephemeris", padx=5, pady=5)
        ephemeris_frame.grid(row=0, column=0, sticky='nsew')

        Label(ephemeris_frame, text="PEPOCH (MJD):").grid(row=0, column=0)
        self.pepoch_var = tk.DoubleVar()
        self.pepoch_var.set(60629)
        self.pepoch_var.trace_add("write", lambda *_: self.clear_data())
        self.pepoch_entry = Entry(ephemeris_frame, textvariable=self.pepoch_var)
        self.pepoch_entry.grid(row=0, column=1)

        Label(ephemeris_frame, text="Frequency (Hz):").grid(row=1, column=0)
        self.freq_var = tk.DoubleVar()
        self.freq_var.set(29.5552429346)
        self.freq_var.trace_add("write", lambda *_: self.clear_data())
        self.freq_entry = Entry(ephemeris_frame, textvariable=self.freq_var)
        self.freq_entry.grid(row=1, column=1)

        # LabelFrame for lightcurve parameters
        lc_params_frame = LabelFrame(self.main_frame, text="Lightcurve parameters", padx=5, pady=5)
        lc_params_frame.grid(row=1, column=0, sticky='nsew')

        Label(lc_params_frame, text="# LC Bins:").grid(row=0, column=0)
        self.n_bin_var = tk.IntVar()
        self.n_bin_var.set(10)
        self.n_bin_var.trace_add("write", lambda *_: self.clear_lc())
        self.n_bin_entry = Entry(lc_params_frame, textvariable=self.n_bin_var)
        self.n_bin_entry.grid(row=0, column=1)

        Label(lc_params_frame, text="# Buffer size (s):").grid(row=1, column=0)
        self.buffer_size_var = tk.DoubleVar()
        self.buffer_size_var.set(0.5)
        self.buffer_size_var.trace_add("write", lambda *_: self.extend_buffers())
        self.buffer_size_entry = Entry(lc_params_frame, textvariable=self.buffer_size_var)
        self.buffer_size_entry.grid(row=1, column=1)

        Label(lc_params_frame, text="ROI width (px):").grid(row=2, column=0)
        self.width_var = tk.IntVar()
        self.width_var.set(24)
        self.width_var.trace_add("write", lambda *_: self.clear_lc())
        self.width_entry = Entry(lc_params_frame, textvariable=self.width_var)
        self.width_entry.grid(row=2, column=1)

        Label(lc_params_frame, text="Lock ROI:").grid(row=3, column=0)
        self.lock_roi_var = tk.IntVar()
        self.lock_roi_var.set(0)
        self.lock_roi_button = Checkbutton(lc_params_frame, variable=self.lock_roi_var, onvalue=1, offvalue=0)
        self.lock_roi_button.grid(row=3, column=1)

        Label(lc_params_frame, text="Blur:").grid(row=4, column=0)
        self.blur_var = tk.DoubleVar()
        self.blur_var.set(0)
        self.blur_scale = Scale(lc_params_frame, from_=0, to_=6, resolution=0.1, length=150, variable=self.blur_var, orient=tk.HORIZONTAL)
        self.blur_scale.grid(row=4, column=1)

        self.reset_image_button = Button(lc_params_frame, text="Reset images", command=self.clear_image)
        self.reset_image_button.grid(row=5, column=0)
        self.reset_lc_button = Button(lc_params_frame, text="Reset lightcurve", command=self.clear_lc)
        self.reset_lc_button.grid(row=5, column=1)

        self.status_message = tk.Label(lc_params_frame, text="Initialized", justify=tk.LEFT, anchor="w", width=30)
        self.status_message.grid(row=6, column=0, columnspan=2, sticky='nsew')

        # Instructions
        lc_params_frame = LabelFrame(self.main_frame, text="Instructions", padx=5, pady=5)
        lc_params_frame.grid(row=2, column=0, sticky='nsew')
        Label(lc_params_frame, text="Click in the image to set the lightcurve ROI").grid(row=0, column=0)
        Label(lc_params_frame, text="Then click and drag in the LC to set \"on\" range").grid(row=1, column=0)
        Label(lc_params_frame, text="Hold shift to set the \"off\" range").grid(row=2, column=0)
        Label(lc_params_frame, text="The on window shows the phase-subtracted image").grid(row=3, column=0)

        # Lightcurve plot
        self.lc_fig, self.lc_ax = plt.subplots(figsize=(6,4), dpi=60)
        self.lc_line = self.lc_ax.step([], [], color='k', where='mid')[0]
        self.lc_ebar = self.lc_ax.errorbar([], [], [], color='k', lw=1, ls='none')
        self.lc_on_span = PhaseRangePlot(self.lc_ax, color='C0', label="On")
        self.lc_off_span = PhaseRangePlot(self.lc_ax, color='C1', label="Off")
        self.lc_temporary_span = PhaseRangePlot(self.lc_ax, color='gray', label=None)
        self.lc_ax.legend()
        self.lc_ax.set_xlim(0, 1)
        self.lc_ax.set_xlabel("Phase")
        self.lc_ax.set_ylabel("Normalized flux")
        self.lc_fig.tight_layout()

        # Initialize data
        if type(feed) is SavedDataThread:
            self.set_camera_data_from_file(feed)
        # elif type(feed) is CameraThread:
        #     self.set_camera_data_from_camera_thread(feed)
        else:
            raise Exception(f"Feed of type {type(feed)} is not supported")
        
        self.clear_data()
        self.update_frame_display()

    def on_close(self):
        cv2.destroyAllWindows()
        self.destroy()

    def set_camera_data_from_camera_thread(self, camera_thread):
        """
        Set the camera feed metadata based on the camera thread
        """
        raise NotImplementedError()
    
        # self.t_start = 
        # self.n_stacks = 
        # self.delta_t = 
        # self.image_shape = 

    def set_camera_data_from_file(self, saved_data_feed):
        """
        Set the camera feed metadata based on a file's header
        """
        with fits.open(saved_data_feed.filename) as hdul:
            # Get the start time of the observation in MJD
            start_time = hdul[0].header["GPSSTART"]
            mjd = Time(start_time).jd - 2400000.5
            delta_time_stamp = hdul[2].data["TIMESTAMP"][1] - hdul[2].data["TIMESTAMP"][0]
            frame_shape = hdul[1].data[0].shape

        self.t_start = mjd * 3600 * 24 # Time at which the camera started observing (seconds)
        self.n_stacks = 100 # TODO Number of stacked images in a given frame
        self.delta_t = delta_time_stamp / self.n_stacks # Time between stacked images
        self.image_shape = (frame_shape[0]//self.n_stacks, frame_shape[1])

    def on_event_image(self, event, x, y, flags, param):
        left_down = (flags & cv2.EVENT_FLAG_LBUTTON)!=0
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and left_down):
            self.roi_moved = (x, y)

    def on_event_lc(self, event, x, y, flags, param):
        shift_down = (flags & cv2.EVENT_FLAG_SHIFTKEY)!=0
        left_down = (flags & cv2.EVENT_FLAG_LBUTTON)!=0

        mouse_phase = self.lc_ax.transData.inverted().transform((x, y))[0]
        if 0 > mouse_phase or mouse_phase > 1:
            return
        
        # The cursor is in a valid position. Begin to move the phase windows
        if event == cv2.EVENT_LBUTTONDOWN:
            # Begin drawing a new boundary
            self.temporary_range = [mouse_phase, mouse_phase]
        elif event == cv2.EVENT_MOUSEMOVE and left_down and self.temporary_range is not None:
            self.temporary_range[1] = mouse_phase
        elif event == cv2.EVENT_LBUTTONUP and self.temporary_range is not None:
            self.temporary_range[1] = mouse_phase
            if shift_down:
                self.off_range = np.copy(self.temporary_range)
            else:
                self.on_range = np.copy(self.temporary_range)
            if self.off_range is not None and self.on_range is not None:
                self.ranges_moved = True
            self.temporary_range = None

    def extend_buffers(self):
        buffer_frame_size = self.get_buffer_size()
        self.lc_fluxes.extend(buffer_frame_size)
        self.on_image.extend(buffer_frame_size)
        self.off_image.extend(buffer_frame_size)

    def get_buffer_size(self):
        buffer_time_limit = get_tk_value(self.buffer_size_var)
        if np.isnan(buffer_time_limit):
            return 1
        buffer_frame_limit = int(np.round(buffer_time_limit / self.delta_t))
        return max(buffer_frame_limit, 1)

    def get_phase(self, timestamp):
        """
        Get the phase of an event with the provided ephemeris and timestamp
        """
        pepoch = get_tk_value(self.pepoch_var)
        freq  =get_tk_value(self.freq_var)
        delta_time = self.t_start + timestamp - pepoch * 3600 * 24
        phase = delta_time * freq
        phase -= np.floor(phase)
        return phase

    def clear_data(self):
        self.clear_lc()
        self.clear_image()

    def clear_lc(self):
        # Clearing can be done by both a CV callback and the main thread, hence the lock.
        n_bins = get_tk_value(self.n_bin_var)
        n_bins = max(n_bins, 1)
        self.lc_phase_bin_edges = np.linspace(0, 1, n_bins+1)
        
        if self.lc_fluxes is None or n_bins != len(self.lc_fluxes.get()):
            buffer_frame_size = self.get_buffer_size()
            self.lc_fluxes = RollingBuffer(buffer_frame_size, (n_bins,)) # Fluxes in each LC bin
        else:
            self.lc_fluxes.clear()

    def clear_image(self):
        if self.on_image is None or self.off_image is None:
            buffer_frame_size = self.get_buffer_size()
            self.on_image = RollingBuffer(buffer_frame_size, self.image_shape) # On data
            self.off_image = RollingBuffer(buffer_frame_size, self.image_shape) # Off data
        else:
            self.on_image.clear()
            self.off_image.clear()

    def process_lc(self, data, timestamp):
        if self.roi_center is None:
            return
        
        # Get flux within ROI
        xs, ys = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        width = get_tk_value(self.width_var)
        pixel_mask = np.abs(xs - self.roi_center[0]) <= width//2
        pixel_mask &= np.abs(ys - self.roi_center[1]) <= width//2
        flux = np.sum(data[pixel_mask])

        # Add to the LC
        phase = self.get_phase(timestamp)
        lc = np.zeros(len(self.lc_phase_bin_edges)-1)
        lc[np.digitize(phase, self.lc_phase_bin_edges)-1] = flux
        self.lc_fluxes.push(lc)

    def process_image(self, data, timestamp):
        if self.on_range is None or self.off_range is None:
            # Display a stacked, single image
            self.off_image.push(data)
        else:
            phase = self.get_phase(timestamp)
            if check_phase(phase, self.on_range):
                self.on_image.push(data)
            if check_phase(phase, self.off_range):
                self.off_image.push(data)

    def show_image(self, window_name, image, vmin=None, vmax=None):
        if vmin is None: vmin = np.min(image)
        if vmax is None: vmax = np.nanpercentile(image, 95)
        if vmin == vmax: 
            scaled_image = np.zeros_like(image)
        else:
            scaled_image = ((image - vmin) / (vmax - vmin))
        blur_scale = get_tk_value(self.blur_var)
        if blur_scale > 0:
            line = np.arange(-np.ceil(blur_scale)*3, np.ceil(blur_scale)*3+1)
            xs, ys = np.meshgrid(line,line)
            gauss = np.exp(-(xs**2 + ys**2) / (2*blur_scale**2))
            gauss /= np.sum(gauss)
            scaled_image = convolve(scaled_image, gauss, mode="same")
        scaled_image[~np.isfinite(scaled_image)] = 0
        scaled_image = np.clip(np.round(scaled_image*255), 0, 255).astype(np.uint8)
        scaled_data_bgr = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
        if self.roi_center is not None:
            width = get_tk_value(self.width_var)
            cv2.rectangle(scaled_data_bgr,
                (self.roi_center[0]-width//2, self.roi_center[1]-width//2),
                (self.roi_center[0]+width//2, self.roi_center[1]+width//2),
                (0, 255, 0), 1)
        cv2.imshow(window_name, scaled_data_bgr)
        cv2.waitKey(1)

    def display_lc(self):
        if self.roi_center is None:
            return
        
        if not self.lc_window_created:
            # Creat the window
            cv2.namedWindow('Lightcurve', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Lightcurve', self.on_event_lc)
            self.lc_window_created = True

        # Set the lightcurve data
        lc_fluxes = self.lc_fluxes.get()
        lc_errorbar = np.sqrt(lc_fluxes)
        if np.mean(lc_fluxes) > 0:
            lc_errorbar /= np.mean(lc_fluxes)
            lc_fluxes /= np.mean(lc_fluxes)
            self.lc_ax.set_ylim(0.9*np.min(lc_fluxes), 1.1*np.max(lc_fluxes))
        else:
            self.lc_ax.set_ylim(0, 1)
        bin_centers = (self.lc_phase_bin_edges[1:] + self.lc_phase_bin_edges[:-1])/2
        self.lc_line.set_data(bin_centers, lc_fluxes)

        # Update the error bars
        self.lc_ebar[2][0].set_segments([(
            (bin_centers[i], lc_fluxes[i] - lc_errorbar[i]),
            (bin_centers[i], lc_fluxes[i] + lc_errorbar[i]),
        ) for i in range(len(lc_fluxes))])

        # Set the phase windows
        self.lc_on_span.update(self.on_range)
        self.lc_off_span.update(self.off_range)
        self.lc_temporary_span.update(self.temporary_range)

        self.lc_fig.canvas.draw()
        img_plot = np.array(self.lc_fig.canvas.renderer.buffer_rgba())
        cv2.imshow('Lightcurve', cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR))
        cv2.waitKey(1)

    def display_image(self):
        if not self.off_window_created:
            cv2.namedWindow("Off frame", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Off frame', self.on_event_image)
            self.off_window_created = True

        self.show_image("Off frame", self.off_image.get())

        # Show the difference image if possible
        if self.on_range is None or self.off_range is None:
            return
        
        if not self.on_window_created:
            cv2.namedWindow('On frame', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('On frame', self.on_event_image)
            self.on_window_created = False

        # Show the on image
        image = self.on_image.get().astype(float) / get_phase_duration(self.on_range) - self.off_image.get().astype(float) / get_phase_duration(self.off_range)
        self.show_image("On frame", image, vmin=0)


    def update_frame_display(self):
        # Update the status message
        if self.roi_center is None:
            self.status_message.config(text="No ROI has been set", fg="red")
        elif self.on_range is None or self.off_range is None:
            self.status_message.config(text="No on / off bins have been set", fg="red")
        else:
            self.status_message.config(text="Running", fg="white")

        # Check to see if a callback triggered an action
        if self.roi_moved is not None:
            if get_tk_value(self.lock_roi_var) == 0:
                # ROI is not locked
                self.roi_center = (self.roi_moved[0], self.roi_moved[1])
                self.clear_lc()
                if self.on_range is not None and self.off_range is not None:
                    # Clear the images only if they were on-off folded
                    self.clear_images()
            self.roi_moved = None

        if self.ranges_moved:
            self.clear_image()
            self.ranges_moved = False
        
        # Process everything in the queue 
        while not self.frame_queue.empty():
            frame = self.frame_queue.get_nowait()
            timestamp = self.timestamp_queue.get_nowait()
            slice_width = frame.shape[0] // self.n_stacks # Width of each slice in pixels
            stripped_image = frame.reshape(-1, slice_width, frame.shape[1]).astype(np.uint32)
            for (strip_index, strip) in enumerate(stripped_image):
                self.process_lc(strip, timestamp + strip_index * self.delta_t)
                self.process_image(strip, timestamp + strip_index * self.delta_t)

        # Display the data
        self.display_lc()
        self.display_image()

        # Refresh
        self.after(100, self.update_frame_display)  # About 10 FPS

class SavedDataThread(threading.Thread):
    def __init__(self, filename, frame_queue, timestamp_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        self.filename = filename
        with fits.open(filename) as hdul:
            self.frames = np.array(hdul[1].data)
            self.timestamps = hdul[2].data["TIMESTAMP"]
            self.framestamps = hdul[2].data["FRAMESTAMP"]
        self.time_between_frames = self.timestamps[1] - self.timestamps[0]

    def run(self):
        threading.Thread(target=self.load_file, daemon=True).start()

    def load_file(self):
        for (frame, timestamp) in zip(self.frames, self.timestamps):
            # time.sleep(self.time_between_frames) # TODO
            time.sleep(0.5)
            try:
                self.frame_queue.put_nowait(frame)
                self.timestamp_queue.put_nowait(timestamp)
            except queue.Full:
                self.frame_queue.get_nowait()
                self.timestamp_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                self.timestamp_queue.put_nowait(timestamp)
        print("Done")

if __name__ == "__main__":
    import time
    from astropy.io import fits

    # Create shared queues containing the data coming from the camera / saved data
    QUEUE_MAXSIZE = 20
    frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    timestamp_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # Create the thread to feed the GUI data
    test_thread = SavedDataThread("../../data/crab1000_20241026_101129049763441_cube066.fits", frame_queue, timestamp_queue)
    test_thread.start()

    # Create the gui
    phase_gui = PhaseGUI(frame_queue, timestamp_queue, test_thread)

    # Run the gui
    phase_gui.camera_thread = test_thread
    phase_gui.protocol("WM_DELETE_WINDOW", phase_gui.on_close)
    phase_gui.mainloop()