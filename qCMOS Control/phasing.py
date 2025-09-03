import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import convolve
import tkinter as tk
from tkinter import StringVar, OptionMenu, Checkbutton, Label, Entry, Button, Scale, Frame, LabelFrame, messagebox
import threading
from astropy.time import Time
# import keyboard
import cv2
import queue


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
    def __init__(self, frame_queue, timestamp_queue):
        super().__init__()
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue

        self.roi_center = None # Center of the ROI (pix)
        self.roi_width = None # Full width of the square ROI (pix)

        self.lc_fluxes = None # Fluxes in each LC bin
        self.lc_phase_bin_edges = None # Edges of the LC bin (phase)

        self.on_image = None # On data
        self.off_image = None # Off data
        self.on_range = None # On phase range
        self.off_range = None # Off phase range
        self.temporary_range = None # Temporary range used by the UI when setting a new range
        self.t_start = None # Time at which the camera started observing
        self.n_stacks = None # Number of stacked images in a given frame
        self.delta_t = None # Time between frames

        self.lc_window_created = False
        self.on_window_created = False
        self.off_window_created = False

        # Set up GUI main frame
        self.title("Lightspeed Phasing GUI")
        self.geometry("350x400")  # Adjust window size to tighten the layout
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)

        # LabelFrame for Ephemeris
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

        Label(lc_params_frame, text="ROI width (px):").grid(row=1, column=0)
        self.width_var = tk.IntVar()
        self.width_var.set(24)
        self.width_var.trace_add("write", lambda *_: self.clear_lc())
        self.width_entry = Entry(lc_params_frame, textvariable=self.width_var)
        self.width_entry.grid(row=1, column=1)

        Label(lc_params_frame, text="Blur:").grid(row=2, column=0)
        self.blur_var = tk.IntVar()
        self.blur_var.set(0)
        self.blur_entry = Entry(lc_params_frame, textvariable=self.blur_var)
        self.blur_entry.grid(row=2, column=1)

        self.reset_image_button = Button(lc_params_frame, text="Reset images", command=self.clear_image)
        self.reset_image_button.grid(row=3, column=0)
        self.reset_lc_button = Button(lc_params_frame, text="Reset lightcurve", command=self.clear_lc)
        self.reset_lc_button.grid(row=3, column=1)

        self.status_message = tk.Label(lc_params_frame, text="Initialized", justify=tk.LEFT, anchor="w", width=30)
        self.status_message.grid(row=4, column=0, columnspan=2, sticky='nsew')

        # Instructions
        lc_params_frame = LabelFrame(self.main_frame, text="Instructions", padx=5, pady=5)
        lc_params_frame.grid(row=2, column=0, sticky='nsew')
        Label(lc_params_frame, text="Click in the image to set the lightcurve ROI").grid(row=0, column=0)
        Label(lc_params_frame, text="Then click and drag in the LC to set \"on\" range").grid(row=1, column=0)
        Label(lc_params_frame, text="Hold shift to set the \"off\" range").grid(row=2, column=0)
        Label(lc_params_frame, text="The on window shows the phase-subtracted image").grid(row=3, column=0)


        # Lightcurve plot
        self.lc_fig, self.lc_ax = plt.subplots(figsize=(6,4), dpi=60)
        self.lc_line = self.lc_ax.step([], [], color='k')[0]
        self.lc_on_span = PhaseRangePlot(self.lc_ax, color='C0', label="On")
        self.lc_off_span = PhaseRangePlot(self.lc_ax, color='C1', label="Off")
        self.lc_temporary_span = PhaseRangePlot(self.lc_ax, color='gray', label=None)
        self.lc_ax.legend()
        self.lc_ax.set_xlim(0, 1)
        self.lc_ax.set_xlabel("Phase")
        self.lc_ax.set_ylabel("Normalized flux")
        self.lc_fig.tight_layout()
        
        self.clear_lc()
        self.update_frame_display()

    def on_close(self):
        cv2.destroyAllWindows()
        self.destroy()

    def set_camera_data(self, ):
        pass # TODO

    def set_camera_from_file(self, filename):
        # TODO
        with fits.open(filename) as hdul:
            start_time = hdul[0].header["GPSSTART"]
            mjd = Time(start_time).jd - 2400000.5

            delta_time_stamp = hdul[2].data["TIMESTAMP"][1] - hdul[2].data["TIMESTAMP"][0]
        self.t_start = mjd * 3600 * 24
        self.n_stacks = 100
        self.delta_t = delta_time_stamp / self.n_stacks

    def on_event_image(self, event, x, y, flags, param):
        left_down = (flags & cv2.EVENT_FLAG_LBUTTON)!=0
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and left_down):
            # Place an ROI
            self.roi_center = x, y
            self.clear_data()

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
            self.temporary_range = None

    def get_phase(self, timestamp):
        """
        Get the phase of an event with the provided ephemeris and timestamp
        """
        pepoch = self.pepoch_var.get()
        freq = self.freq_var.get()
        delta_time = self.t_start + timestamp - pepoch * 3600 * 24
        phase = delta_time * freq
        phase -= np.floor(phase)
        return phase

    def clear_data(self):
        self.clear_lc()
        self.clear_image()

    def clear_lc(self):
        self.lc_phase_bin_edges = np.linspace(0, 1, self.n_bin_var.get()+1)
        self.lc_fluxes = np.zeros(len(self.lc_phase_bin_edges)-1)

    def clear_image(self):
        self.on_image = None
        self.off_image = None

    def process_lc(self, data, timestamp):
        if self.roi_center is None:
            return
        
        xs, ys = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        width = self.width_var.get()
        pixel_mask = np.abs(xs - self.roi_center[0]) <= width//2
        pixel_mask &= np.abs(ys - self.roi_center[1]) <= width//2
        phase = self.get_phase(timestamp)
        flux = np.sum(data[pixel_mask])

        self.lc_fluxes[np.digitize(phase, self.lc_phase_bin_edges)-1] += flux


    def process_image(self, data, timestamp):
        if self.on_range is None or self.off_range is None:
            # Display a stacked, single image
            if self.off_image is None:
                self.off_image = np.copy(data)
            else:
                self.off_image += data
        else:
            phase = self.get_phase(timestamp)
            if check_phase(phase, self.on_range):
                if self.on_image is None:
                    self.on_image = np.copy(data)
                else:
                    self.on_image += data
            if check_phase(phase, self.off_range):
                if self.off_image is None:
                    self.off_image = np.copy(data)
                else:
                    self.off_image += data

    def display_lc(self):
        if self.roi_center is None:
            return
        if not self.lc_window_created:
            cv2.namedWindow('Lightcurve', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Lightcurve', self.on_event_lc)
            self.lc_window_created = True

        # Set the lightcurve data
        if np.mean(self.lc_fluxes) > 0:
            display_fluxes = self.lc_fluxes / np.mean(self.lc_fluxes)
            self.lc_ax.set_ylim(0.9*np.min(display_fluxes), 1.1*np.max(display_fluxes))
        else:
            display_fluxes = self.lc_fluxes
            self.lc_ax.set_ylim(0, 1)
        bin_centers = (self.lc_phase_bin_edges[1:] + self.lc_phase_bin_edges[:-1])/2
        self.lc_line.set_data(bin_centers, display_fluxes)

        # Set the phase windows
        self.lc_on_span.update(self.on_range)
        self.lc_off_span.update(self.off_range)
        self.lc_temporary_span.update(self.temporary_range)

        self.lc_fig.canvas.draw()
        img_plot = np.array(self.lc_fig.canvas.renderer.buffer_rgba())
        cv2.imshow('Lightcurve', cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR))
        cv2.waitKey(1)

    def show_image(self, window_name, image, vmin=None, vmax=None):
        if vmin is None: vmin = np.min(image)
        if vmax is None: vmax = np.max(image)
        scaled_image = ((image - vmin) / (vmax - vmin))
        blur_scale = self.blur_var.get()
        if blur_scale > 0:
            line = np.arange(-np.ceil(blur_scale)*3, np.ceil(blur_scale)*3+1)
            xs, ys = np.meshgrid(line,line)
            gauss = np.exp(-(xs**2 + ys**2) / (2*blur_scale**2))
            gauss /= np.sum(gauss)
            scaled_image = convolve(scaled_image, gauss, mode="same")
        scaled_data_bgr = cv2.cvtColor((scaled_image*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if self.roi_center is not None:
            width = self.width_var.get()
            cv2.rectangle(scaled_data_bgr,
                (self.roi_center[0]-width//2, self.roi_center[1]-width//2),
                (self.roi_center[0]+width//2, self.roi_center[1]+width//2),
                (0, 255, 0), 1)
        cv2.imshow(window_name, scaled_data_bgr)
        cv2.waitKey(1)

    def display_image(self):
        if self.off_image is None:
            return
        if not self.off_window_created:
            cv2.namedWindow("Off frame", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Off frame', self.on_event_image)
            self.off_window_created = True

        # Show the off image
        self.show_image("Off frame", self.off_image)

        if self.on_image is None:
            return
        if self.on_range is None or self.off_range is None:
            return
        
        if not self.on_window_created:
            cv2.namedWindow('On frame', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('On frame', self.on_event_image)
            self.on_window_created = False

        # Show the on image
        image = self.on_image / get_phase_duration(self.on_range) - self.off_image / get_phase_duration(self.off_range)
        self.show_image("On frame", image, vmin=0)


    def update_frame_display(self):
        # Update the status message
        if self.t_start is None or self.n_stacks is None or self.delta_t is None:
            self.status_message.config(text="Camera settings have not been updated", fg="red")
        elif self.roi_center is None:
            self.status_message.config(text="No ROI has been set", fg="red")
        elif self.on_range is None or self.off_range is None:
            self.status_message.config(text="No on / off bins have been set", fg="red")
        else:
            self.status_message.config(text="Running", fg="white")
        
        # Process everything in the queue 
        while not self.frame_queue.empty():
            frame = self.frame_queue.get_nowait()
            timestamp = self.timestamp_queue.get_nowait()
            slice_width = frame.shape[0] // self.n_stacks # Width of each slice in pixels
            stripped_image = frame.reshape(-1, slice_width, frame.shape[1]).astype(np.uint8) # I'm not saving the frame, so why not cast to 255
            for (strip_index, strip) in enumerate(stripped_image):
                self.process_lc(strip, timestamp + strip_index * self.delta_t)
                self.process_image(strip, timestamp + strip_index * self.delta_t)

        # Display the data
        self.display_lc()
        self.display_image()

        # Refresh
        self.after(200, self.update_frame_display)  # Update frame display every ~50 ms (about 20 FPS)

class TestFromSavedData(threading.Thread):
    def __init__(self, filename, frame_queue, timestamp_queue, gui):
        super().__init__()
        gui.set_camera_from_file(filename)
        self.frame_queue = frame_queue
        self.timestamp_queue = timestamp_queue
        with fits.open(filename) as hdul:
            self.frames = np.array(hdul[1].data)
            self.timestamps = hdul[2].data["TIMESTAMP"]
            self.framestamps = hdul[2].data["FRAMESTAMP"]

    def run(self):
        threading.Thread(target=self.load_file, daemon=True).start()

    def load_file(self):
        for (frame, timestamp) in zip(self.frames, self.timestamps):
            time.sleep(0.5)
            try:
                self.frame_queue.put_nowait(frame)
                self.timestamp_queue.put_nowait(timestamp)
            except queue.Full:
                self.frame_queue.get_nowait()
                self.timestamp_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                self.timestamp_queue.put_nowait(timestamp)

if __name__ == "__main__":
    import time
    from astropy.io import fits

    QUEUE_MAXSIZE = 20
    frame_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)  # Limit the size of the frame queue
    timestamp_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)

    phase_gui = PhaseGUI(frame_queue, timestamp_queue)

    # Initialize the camera thread with a reference to the GUI
    test_thread = TestFromSavedData("../../data/crab1000_20241026_101129049763441_cube066.fits", frame_queue, timestamp_queue, phase_gui)
    test_thread.start()

    phase_gui.camera_thread = test_thread
    phase_gui.protocol("WM_DELETE_WINDOW", phase_gui.on_close)
    phase_gui.mainloop()