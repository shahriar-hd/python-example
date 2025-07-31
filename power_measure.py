# Measure the power consume and co2 while training the model.
# Only meature the GPU power consume.
# Run this code next to your model code or use thread for multitasking.
# When the folder of Dataset creates it works up to when seen the model folder.
# This code write by Shahriar-hd and Gemini.

import subprocess
import time
import csv
import multiprocessing
import os
import sys

# Function to monitor GPU power consumption
def monitor_gpu_power(log_file_path, interval=0.1, file_to_watch='model'):
    """
    Monitors GPU power consumption at specified intervals and saves it to a CSV file.
    :param stop_event: A multiprocessing.Event to signal stopping.
    :param log_file_path: The path to the CSV file for saving data.
    :param interval: The time interval between each data reading (in seconds).
    """
    print(f"[{os.getpid()}] GPU power monitoring started. Data will be saved to '{log_file_path}'.")
    with open(log_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'GPU Power (W)']) # Only power consumption

        model_file_to_watch = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_to_watch)

        while not os.path.exists(model_file_to_watch):
            try:
                # Execute nvidia-smi to get power draw
                # --format=csv,noheader,nounits ensures the output is just a number
                command = "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
                result = subprocess.run(command.split(), capture_output=True, text=True, check=True)
                power_draw = float(result.stdout.strip()) # Power in Watts

                current_time = time.time()
                writer.writerow([current_time, power_draw])
                csvfile.flush() # Ensure data is written immediately to file

            except FileNotFoundError:
                print(f"[{os.getpid()}] Error: nvidia-smi not found. Please ensure it's in your system's PATH.")
            except subprocess.CalledProcessError as e:
                print(f"[{os.getpid()}] Error executing nvidia-smi: {e}")
            except ValueError:
                print(f"[{os.getpid()}] Error: nvidia-smi output is not parseable. Perhaps GPU is not in use or another issue.")
            except Exception as e:
                print(f"[{os.getpid()}] Unknown error during GPU monitoring: {e}")

            time.sleep(interval)
    print(f"[{os.getpid()}] GPU power monitoring stopped.")

# Main function to coordinate processes and calculate results
def main():
    dataset_dir_name = "yolo_dataset"
    log_file_name = "gpu_power_log.csv"
    model_file_to_watch = "yolov11_seg_tumor.pt"
    carbon_intensity_factor = 0.742 # kg CO2/kWh in Iran

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_script_dir, dataset_dir_name)

    print(f"Monitoring script started. Looking for directory: '{dataset_path}'...")
    while not os.path.isdir(dataset_path):
        time.sleep(3) # Wait 3 seconds before checking again
    print(f"Directory '{dataset_dir_name}' found. Proceeding...")

    monitor_gpu_power(log_file_name)
    
    try:
        with open(log_file_name, 'r') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) # Skip header row

            # Check if there's any data beyond the header
            rows = list(reader) # Read all rows into a list to check emptiness
            if not rows:
                print("Log file contains no data. Ensure nvidia-smi worked correctly and monitoring ran.")
                return

            previous_timestamp = None
            first_timestamp = None
            last_timestamp = None
            total_energy_wh = 0.0
            data_points = 0

            # Reset reader for actual processing
            csvfile.seek(0)
            next(reader) # Skip header again

            for row in reader:
                try:
                    current_timestamp = float(row[0])
                    power_watts = float(row[1])

                    if first_timestamp is None:
                        first_timestamp = current_timestamp
                    last_timestamp = current_timestamp

                    if previous_timestamp is not None:
                        # Calculate energy for the time interval between two data points
                        time_diff_seconds = current_timestamp - previous_timestamp
                        # Energy = Power (Watts) * Time (hours)
                        energy_in_interval_wh = power_watts * (time_diff_seconds / 3600)
                        total_energy_wh += energy_in_interval_wh
                        data_points += 1
                    previous_timestamp = current_timestamp
                except (ValueError, IndexError) as e:
                    print(f"Error reading row '{row}': {e}")
                    continue

        if data_points == 0:
            print("Not enough data collected to calculate energy consumption.")
            return

        total_energy_kwh = total_energy_wh / 1000 # Convert Watt-hours to Kilowatt-hours
        co2_emissions_kg = total_energy_kwh * carbon_intensity_factor

        # Calculate total duration of monitoring
        total_duration_seconds = last_timestamp - first_timestamp if first_timestamp is not None else 0
        total_duration_minutes = total_duration_seconds / 60

        print("\n--- Energy Consumption and CO2 Emissions Results ---")
        print(f"Total monitoring duration: {total_duration_minutes:.2f} minutes ({total_duration_seconds:.2f} seconds)")
        print(f"Total GPU energy consumed during training: {total_energy_kwh:.4f} kWh")
        print(f"Estimated Carbon Dioxide (CO2) emissions: {co2_emissions_kg:.4f} kg")
        print(f"(Based on carbon intensity factor of {carbon_intensity_factor} kg CO2/kWh)")

    except Exception as e:
        print(f"An error occurred during energy and CO2 calculation: {e}")
    finally:
        # Optional: Clean up the log file after calculation
        # if os.path.exists(log_file_name):
        #     os.remove(log_file_name)
        #     print(f"Log file '{log_file_name}' deleted.")
        pass

if __name__ == "__main__":
    main()