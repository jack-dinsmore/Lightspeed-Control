import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time, TimeDelta
from matplotlib.dates import DateFormatter, SecondLocator

# Open the FITS file
fits_filename = 'capture_20240822_144948.fits'  # Replace with your FITS filename
with fits.open(fits_filename) as hdul:
    # Extract image data and header
    image_data = hdul[0].data
    header = hdul[0].header

# Convert STARTIME from header to a Time object with nanosecond precision
start_time = Time(header['STARTIME'], precision=9)
print(start_time.isot)


# Use astropy TimeDelta for precise interval calculation
inter = 5.04e-5 + 7.2e-6  # Interval in seconds
fst_frame = 125051+65535#header['FRSTFRM']
print(fst_frame)

# Add the high-precision offset in seconds
offset_seconds = 57e-6+fst_frame*inter#+5.04e-5 + 7.2e-6  # Offset in seconds
print(offset_seconds)
start_time -= TimeDelta(offset_seconds, format='sec', precision=9)

# Check if frame bundle mode is enabled and get the number of frames per bundle
frame_bundle_mode = header.get('FMBLMD', 0)  # Default to 0 if not found
frames_per_bundle = int(header.get('FMBLNUM', 1))  # Default to 1 if not found

# Initialize lists to store individual frames and timestamps
individual_frames = []
timestamps = []

# Loop over each image (bundle) in the FITS file
for bundle_index in range(image_data.shape[0]):
    # Get the bundle (image) data
    bundle_data = image_data[bundle_index]

    # If frame bundle mode is enabled, break apart the bundle
    if frame_bundle_mode == 2:  # Frame bundle mode is on
        frame_height = bundle_data.shape[0] // frames_per_bundle  # Height of each frame
        for frame_idx in range(frames_per_bundle):
            # Extract the sub-image (frame) from the bundle
            sub_image = bundle_data[frame_idx * frame_height:(frame_idx + 1) * frame_height, :]
            individual_frames.append(sub_image)
            
            # Calculate the timestamp for this sub-image with nanosecond precision
            current_frame_number = fst_frame + bundle_index * frames_per_bundle + frame_idx
            timestamp = start_time + TimeDelta(current_frame_number * inter, format='sec', precision=9)
            timestamps.append(timestamp)
    else:
        # If not using frame bundle mode, just add the image as-is
        individual_frames.append(bundle_data)
        timestamp = start_time + TimeDelta(bundle_index * inter, format='sec', precision=9)
        timestamps.append(timestamp)

# Convert lists to numpy arrays for easy handling
individual_frames = np.array(individual_frames)
timestamps = Time(timestamps, precision=9)

# Convert timestamps to UTC ISO format with nanosecond precision
timestamps_iso = [t.isot for t in timestamps]

# Calculate the average counts per frame
average_counts = np.mean(individual_frames, axis=(1, 2))

# Identify the first point on the rising edge of the pulse
threshold = np.max(average_counts) * 0.1  # Set a threshold at 10% of the maximum
rising_edge_index = np.argmax(average_counts > threshold)  # Find the first index where the counts exceed the threshold

# Get the rising edge time with nanosecond accuracy
rising_edge_time = timestamps[rising_edge_index]

# Convert to ISO format string and ensure it includes microseconds
rising_edge_time_str = rising_edge_time.isot

# Manually format the time string to include full microsecond accuracy
rising_edge_time_str_micro = rising_edge_time_str[:26]  # Truncate or pad to ensure microseconds are included

print(f"Exact time of the first point on the rising edge of the pulse (1 microsecond accuracy): {rising_edge_time_str_micro}")

# Convert the target time (13:54:25) to a Time object
target_time = Time('2024-08-22T13:54:25', precision=9)

# Define the window for zooming in (plus or minus 500 microseconds)
time_window = TimeDelta(1500e-6, format='sec', precision=9)

# Set the limits for the x-axis
xlim_min = target_time - time_window
xlim_max = target_time + time_window

# Plot the average counts per frame vs time (UTC ISO format)
plt.figure(figsize=(10, 6))
plt.plot(timestamps.datetime, average_counts, marker='o', linestyle='-')
plt.xlabel('Time (UTC)')
plt.ylabel('Average Counts per Frame')
plt.title('Average Counts per Frame vs Time')
plt.grid(True)

# Highlight the first point on the rising edge
plt.axvline(timestamps.datetime[rising_edge_index], color='red', linestyle='--', label=f'Rising Edge: {rising_edge_time_str_micro}')
plt.legend()

# Set x-axis limits to zoom in around the target time
#plt.xlim(xlim_min.datetime, xlim_max.datetime)

# Set x-axis to integer second intervals
plt.gca().xaxis.set_major_locator(SecondLocator(interval=1))
plt.gca().xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))  # Format as hours:minutes:seconds

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate the phases based on a 1-second period
period = 1.0  # 1-second period
time_deltas = (timestamps - Time(timestamps[0].iso[:19] + '.000', precision=9)).sec  # Time difference from the nearest integer UTC second
phases = np.mod(time_deltas, period)  # Phase calculation

# Sort the phases and corresponding counts for better plotting
sorted_indices = np.argsort(phases)
sorted_phases = phases[sorted_indices]
sorted_average_counts = average_counts[sorted_indices]

# Plot the average counts per frame vs phase
plt.figure(figsize=(10, 6))
plt.plot(sorted_phases, sorted_average_counts, marker='o', linestyle='-', color='blue')
plt.xlabel('Phase (seconds)')
plt.ylabel('Average Counts per Frame')
plt.title('Phase Folded Average Counts per Frame')
plt.grid(True)
plt.tight_layout()
plt.show()

