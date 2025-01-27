import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import joblib
import threading
import time
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from itertools import cycle
import shutil  # For copying files
import tifffile  # For saving .tif images

# Import Picamera2 module
try:
    from picamera2 import Picamera2
except ImportError:
    # Initialize Tkinter root to show messagebox before exiting
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("Import Error", "The Picamera2 module is not installed. Please install it to use this application.")
    exit()

class SfdiApp:
    """
    Main class for the SFDI application.
    """
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.continuous_measurement_running = False

        # Initialize Picamera2
        self.init_camera()

        # Default directories
        self.save_directory = os.path.join(os.getcwd(), "saved_pictures")
        os.makedirs(self.save_directory, exist_ok=True)  # Ensure save_directory exists
        self.reference_phantom_folder = os.path.join(os.getcwd(), "reference_images")
        os.makedirs(self.reference_phantom_folder, exist_ok=True)

        # Load pattern image paths
        self.projections_base_folder = os.path.join("projections")
        if not os.path.exists(self.projections_base_folder):
            messagebox.showerror("Error", f"Projections folder not found: {self.projections_base_folder}")
            exit()

        # Get image paths directly from projections folder
        self.image_paths = []  # List of image paths
        image_files = [f for f in sorted(os.listdir(self.projections_base_folder)) if f.endswith(('.png', '.jpg'))]
        if not image_files:
            messagebox.showerror("Error", f"No image files found in projections folder: {self.projections_base_folder}")
            exit()
        for image_file in image_files:
            image_path = os.path.join(self.projections_base_folder, image_file)
            self.image_paths.append(image_path)
        self.current_image_index = 0

        # Camera matrix and distortion coefficients
        self.camera_matrix = np.array([[1350, 0, 960],
                                       [0, 1350, 540],
                                       [0, 0, 1]], dtype=np.float32)

        self.dist_coeffs = np.array([-0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float32)

        # Load pre-trained models for Rd prediction
        self.load_rd_models()

        # Parameters for the rational function
        self.params_mu_a = {
            'a0': 2.093868,
            'a1': 642.842312,
            'a2': 327.098248,
            'a3': -325.076674,
            'b0': 1107.943786,
            'b1': -814.774199,
            'b2': 28864.351702,
            'b3': -28472.403839
        }

        self.params_mu_s = {
            'a0': 0.101316,
            'a1': 952.290670,
            'a2': -812.749684,
            'a3': 3253.732536,
            'b0': 1574.365393,
            'b1': -1558.549226,
            'b2': -564.330386,
            'b3': 836.510533
        }

        # Initialize reference images variables
        self.iac_reference_phantom = None
        self.iac_reference_phantom_0 = None

        # Set up the GUI
        self.setup_gui()

    def init_camera(self):
        """
        Initializes the Picamera2 with updated resolution settings.
        """
        self.vid = Picamera2()
        self.configurations = {
            "RAW 10bits 2592x1944": {"format": "SRGGB10", "size": (2592, 1944)},
            "RAW 10bits 1280x720": {"format": "SRGGB10", "size": (1280, 720)},
            "RAW 10bits 1920x1080": {"format": "SRGGB10", "size": (1920, 1080)},
            "RAW 10bits 640x480": {"format": "SRGGB10", "size": (640, 480)},
            "RAW 12bits 1280x720": {"format": "SRGGB12", "size": (1280, 720)},
            "RAW 12bits 1920x1080": {"format": "SRGGB12", "size": (1920, 1080)},
        }
        self.current_config_name = "RAW 10bits 2592x1944"  # Updated resolution
        self.current_config = self.configurations[self.current_config_name]

        # Create a configuration that supports preview and capture
        self.capture_config = self.vid.create_still_configuration(
            main={"size": self.current_config['size'], "format": "RGB888"},  # Main stream for preview
            raw=self.current_config,  # RAW stream for high-resolution capture
            display="main"  # Use 'main' stream for display
        )
        self.vid.configure(self.capture_config)
        self.vid.start()

        # Store the current image dimensions
        self.image_width = self.current_config['size'][0]
        self.image_height = self.current_config['size'][1]

    def load_rd_models(self):
        """
        Loads the pre-trained models for predicting Rd_Phantom_DC and Rd_Phantom_AC.
        """
        try:
            model1_path = os.path.join("utils", "best_mlp_model_M1_2.pkl")
            model2_path = os.path.join("utils", "best_mlp_model_M2_2.pkl")

            # Print absolute paths for debugging
            abs_model1_path = os.path.abspath(model1_path)
            abs_model2_path = os.path.abspath(model2_path)
            print(f"Loading Model M1 from: {abs_model1_path}")
            print(f"Loading Model M2 from: {abs_model2_path}")

            self.model_M1 = joblib.load(abs_model1_path)
            print("Model M1 loaded successfully.")

            self.model_M2 = joblib.load(abs_model2_path)
            print("Model M2 loaded successfully.")

        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Model file not found: {e}")
            self.model_M1 = None
            self.model_M2 = None
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading models: {e}")
            self.model_M1 = None
            self.model_M2 = None

    def setup_gui(self):
        """
        Sets up the GUI elements with an improved, organized layout.
        """
        # Create a main frame to hold left and right frames
        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Left Frame for Controls ---
        self.left_frame = tk.Frame(self.main_frame, padx=5, pady=5)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Adjust the left frame width to be smaller
        self.left_frame.config(width=300)
        self.left_frame.pack_propagate(False)  # Prevent frame from resizing to fit contents

        # --- Capture Controls Frame ---
        capture_frame = tk.LabelFrame(self.left_frame, text="Capture Controls", padx=5, pady=5)
        capture_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        # Arrange buttons in a two-column grid with reduced padding and spacing
        self.btn_select_directory = tk.Button(capture_frame, text="Select Folder", width=14, command=self.browse_directory)
        self.btn_select_directory.grid(row=0, column=0, padx=2, pady=2)

        self.btn_auto_capture = tk.Button(capture_frame, text="Auto Capture", width=14, command=self.auto_capture_thread)
        self.btn_auto_capture.grid(row=0, column=1, padx=2, pady=2)

        self.btn_capture_reference = tk.Button(capture_frame, text="Capture Ref", width=14, command=self.capture_reference_thread)
        self.btn_capture_reference.grid(row=1, column=0, padx=2, pady=2)

        self.btn_continuous_measurement = tk.Button(capture_frame, text="cont. measure", width=14, command=self.continuous_measurement_thread)
        self.btn_continuous_measurement.grid(row=1, column=1, padx=2, pady=2)

        self.btn_project_and_capture = tk.Button(capture_frame, text="Project and Capture", width=14, command=self.project_and_capture_thread)
        self.btn_project_and_capture.grid(row=2, column=0, columnspan=2, padx=2, pady=2)

        self.btn_close = tk.Button(capture_frame, text="Close", width=14, command=self.close_app)
        self.btn_close.grid(row=3, column=0, columnspan=2, padx=2, pady=2)

        # --- Measurement Parameters Frame ---
        measurement_frame = tk.LabelFrame(self.left_frame, text="Measurement Parameters", padx=5, pady=5)
        measurement_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        tk.Label(measurement_frame, text="μ_a:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.mu_a_value_entry = tk.Entry(measurement_frame, width=12)
        self.mu_a_value_entry.grid(row=0, column=1, padx=2, pady=2)
        self.mu_a_value_entry.insert(0, "0.0059")  # Default value

        tk.Label(measurement_frame, text="μ_s:").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        self.mu_s_value_entry = tk.Entry(measurement_frame, width=12)
        self.mu_s_value_entry.grid(row=1, column=1, padx=2, pady=2)
        self.mu_s_value_entry.insert(0, "0.9748")  # Default value

        self.btn_calculate_maps = tk.Button(measurement_frame, text="Calculate μ_a/μ_s", width=16, command=self.calculate_mu_maps_thread)
        self.btn_calculate_maps.grid(row=0, column=2, rowspan=2, padx=2, pady=2)

        # --- Results Folder Frame ---
        results_frame = tk.LabelFrame(self.left_frame, text="Results Folder", padx=5, pady=5)
        results_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        tk.Label(results_frame, text="Folder Name:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.results_folder_entry = tk.Entry(results_frame, width=18)
        self.results_folder_entry.grid(row=0, column=1, padx=2, pady=2)
        self.results_folder_entry.insert(0, "experiment_01")  # Default folder name

        # --- ROI Settings Frame ---
        roi_frame = tk.LabelFrame(self.left_frame, text="ROI Settings", padx=5, pady=5)
        roi_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        tk.Label(roi_frame, text="ROI_x:").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.roi_x_entry = tk.Entry(roi_frame, width=8)
        self.roi_x_entry.grid(row=0, column=1, padx=2, pady=2)
        self.roi_x_entry.insert(0, "0")  # Default x

        tk.Label(roi_frame, text="ROI_y:").grid(row=0, column=2, sticky="e", padx=2, pady=2)
        self.roi_y_entry = tk.Entry(roi_frame, width=8)
        self.roi_y_entry.grid(row=0, column=3, padx=2, pady=2)
        self.roi_y_entry.insert(0, "0")  # Default y

        tk.Label(roi_frame, text="ROI_dx:").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        self.roi_dx_entry = tk.Entry(roi_frame, width=8)
        self.roi_dx_entry.grid(row=1, column=1, padx=2, pady=2)
        self.roi_dx_entry.insert(0, str(self.image_width))  # Default dx set to 2592

        tk.Label(roi_frame, text="ROI_dy:").grid(row=1, column=2, sticky="e", padx=2, pady=2)
        self.roi_dy_entry = tk.Entry(roi_frame, width=8)
        self.roi_dy_entry.grid(row=1, column=3, padx=2, pady=2)
        self.roi_dy_entry.insert(0, str(self.image_height))  # Default dy set to 1944

        # --- Camera Settings Frame ---
        camera_frame = tk.LabelFrame(self.left_frame, text="Camera Settings", padx=5, pady=5)
        camera_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=2, pady=2)

        tk.Label(camera_frame, text="Framerate (fps):").grid(row=0, column=0, sticky="e", padx=2, pady=2)
        self.framerate_entry = tk.Entry(camera_frame, width=8)
        self.framerate_entry.grid(row=0, column=1, padx=2, pady=2)
        self.framerate_entry.insert(0, "30")  # Default value

        tk.Label(camera_frame, text="Exposure Time (μs):").grid(row=1, column=0, sticky="e", padx=2, pady=2)
        self.exposure_entry = tk.Entry(camera_frame, width=8)
        self.exposure_entry.grid(row=1, column=1, padx=2, pady=2)
        self.exposure_entry.insert(0, "10000")  # Default value

        tk.Label(camera_frame, text="ISO (Gain):").grid(row=2, column=0, sticky="e", padx=2, pady=2)
        self.iso_entry = tk.Entry(camera_frame, width=8)
        self.iso_entry.grid(row=2, column=1, padx=2, pady=2)
        self.iso_entry.insert(0, "1")  # Default value

        self.btn_apply_settings = tk.Button(camera_frame, text="Apply Settings", width=12, command=self.apply_camera_settings)
        self.btn_apply_settings.grid(row=2, column=2, padx=2, pady=2)

        # Configure grid weights for responsiveness within left_frame
        for i in range(2):
            self.left_frame.columnconfigure(i, weight=1)

        # --- Right Frame for Camera Preview ---
        self.right_frame = tk.Frame(self.main_frame, width=512, padx=5, pady=5)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.right_frame.pack_propagate(False)  # Prevent frame from resizing to fit contents

        # Center the camera_label within the right_frame
        self.camera_label = tk.Label(self.right_frame)
        self.camera_label.pack(expand=True)

        # Frame to display frequency images in a separate window
        self.frequency_window = tk.Toplevel(self.window)
        self.frequency_window.title("Projection Image")
        self.frequency_window.resizable(True, True)

        # Make the window fill the screen and allow resizing
        self.frequency_window.geometry("800x600")  # Initial size
        self.frequency_canvas = tk.Canvas(self.frequency_window)
        self.frequency_canvas.pack(fill=tk.BOTH, expand=True)
        self.frequency_window.bind("<Configure>", self.on_frequency_canvas_resize)
        self.update_frequency_image()

        # Start updating the camera feed
        self.update_camera_feed()

    def on_frequency_canvas_resize(self, event):
        """
        Resizes the frequency image when the canvas is resized.
        """
        self.update_frequency_image()

    def apply_camera_settings(self):
        """
        Applies the camera settings based on user input.
        """
        try:
            framerate = float(self.framerate_entry.get())
            exposure_time = int(float(self.exposure_entry.get()))
            iso = float(self.iso_entry.get())
            frame_duration = int(1e6 / framerate)
            # Check if exposure time is less than or equal to frame duration
            if exposure_time > frame_duration:
                messagebox.showerror("Configuration Error", "Exposure time must be less than or equal to frame duration.")
                return
            # Update camera settings
            self.vid.set_controls({
                "FrameDurationLimits": (frame_duration, frame_duration),
                "ExposureTime": exposure_time,
                "AnalogueGain": iso
            })
            print(f"Settings applied: Framerate={framerate} fps, Exposure Time={exposure_time} μs, ISO={iso}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for camera settings.")

    def update_camera_feed(self):
        """
        Captures a frame from the camera and updates the image in the GUI.
        """
        frame = self.vid.capture_array("main")
        if frame is not None:
            # Convert to grayscale PIL image
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Draw ROI rectangle on the frame
            roi_coords = self.get_roi_coordinates()
            if roi_coords:
                x, y, dx, dy = roi_coords
                # Ensure ROI is within image bounds
                x = max(0, min(x, self.image_width - 1))
                y = max(0, min(y, self.image_height - 1))
                dx = max(1, min(dx, self.image_width - x))
                dy = max(1, min(dy, self.image_height - y))
                cv2.rectangle(gray_frame, (x, y), (x + dx, y + dy), (255, 0, 0), 2)  # Red rectangle

            image = Image.fromarray(gray_frame)
            # Resize image to have width 512 pixels while maintaining aspect ratio
            new_width = 512
            aspect_ratio = self.image_height / self.image_width
            new_height = int(new_width * aspect_ratio)
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            photo = ImageTk.PhotoImage(image)
            # Update the Label
            self.camera_label.config(image=photo)
            self.camera_label.image = photo
        # Schedule the next camera feed update
        self.window.after(30, self.update_camera_feed)

    def get_roi_coordinates(self):
        """
        Retrieves and validates the ROI coordinates from the input fields.
        Automatically adjusts the ROI if it exceeds image boundaries without displaying error messages.
        Returns a tuple (x, y, dx, dy).
        """
        try:
            # Retrieve ROI values from entry fields
            x = int(self.roi_x_entry.get())
            y = int(self.roi_y_entry.get())
            dx = int(self.roi_dx_entry.get())
            dy = int(self.roi_dy_entry.get())

            # **Automatic Adjustment: Reset x and y to 0 if they are out of bounds**
            if x >= self.image_width:
                x = 0
            elif x < 0:
                x = 0  # Optionally, you can set to the nearest valid value

            if y >= self.image_height:
                y = 0
            elif y < 0:
                y = 0  # Optionally, you can set to the nearest valid value

            # **Ensure dx and dy are at least 1**
            dx = max(1, dx)
            dy = max(1, dy)

            # **Adjust dx and dy if ROI exceeds image boundaries**
            if x + dx > self.image_width:
                dx = self.image_width - x
            if y + dy > self.image_height:
                dy = self.image_height - y

            # **Return the adjusted ROI coordinates**
            return (x, y, dx, dy)
        except ValueError:
            # **Handle invalid integer inputs**
            messagebox.showerror("Input Error", "Please enter valid integer values for ROI (x, y, dx, dy).")
            return None

    def browse_directory(self):
        """
        Opens a dialog for the user to select the save directory.
        """
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.save_directory = directory_path
            print(f"Selected directory: {self.save_directory}")

    def auto_capture_thread(self):
        """
        Starts automatic capture in a separate thread.
        """
        threading.Thread(target=self.auto_capture, daemon=True).start()

    def capture_reference_thread(self):
        """
        Starts reference capture in a separate thread.
        """
        threading.Thread(target=self.capture_reference, daemon=True).start()

    def calculate_mu_maps_thread(self):
        """
        Starts the calculation of mu_a and mu_s maps in a separate thread.
        """
        threading.Thread(target=self.calculate_mu_maps, daemon=True).start()

    def continuous_measurement_thread(self):
        """
        Starts continuous measurement in a separate thread.
        """
        threading.Thread(target=self.continuous_measurement, daemon=True).start()

    def project_and_capture_thread(self):
        """
        Starts project and capture in a separate thread.
        """
        threading.Thread(target=self.project_and_capture, daemon=True).start()

    def auto_capture(self):
        """
        Automatically captures images of the projections.
        """
        print("Automatically capturing images...")
        for image_path in self.image_paths:
            self.update_frequency_image(image_path)
            self.window.update()
            time.sleep(0.5)  # Reduce wait time
            # Capture frame from 'raw' stream
            frame = self.vid.capture_array("raw")
            if frame is not None:
                filename = os.path.basename(image_path)
                output_filename = filename.replace('.png', '.raw').replace('.jpg', '.raw')
                output_path = os.path.join(self.save_directory, output_filename)
                # Save frame as RAW image
                frame.tofile(output_path)
                print(f"Captured image saved: {output_path}")
            else:
                print("Error capturing image.")

    def capture_reference(self):
        """
        Captures reference images of the projections.
        """
        print("Capturing reference images...")
        for image_path in self.image_paths:
            self.update_frequency_image(image_path)
            self.window.update()
            time.sleep(0.5)  # Reduce wait time
            # Capture frame from 'raw' stream
            frame = self.vid.capture_array("raw")
            if frame is not None:
                filename = os.path.basename(image_path)
                output_filename = filename.replace('.png', '.raw').replace('.jpg', '.raw')
                output_path = os.path.join(self.reference_phantom_folder, output_filename)
                # Save frame as RAW image
                frame.tofile(output_path)
                print(f"Reference image saved: {output_path}")
            else:
                print("Error capturing image.")
        # Reload the reference images
        self.load_reference_images()

    def project_and_capture(self):
        """
        Projects images in the projections folder and captures the images, saving them to 'captured_images' folder.
        """
        print("Projecting images and capturing...")
        # Ensure the captured_images folder exists
        captured_images_folder = os.path.join(os.getcwd(), "captured_images")
        os.makedirs(captured_images_folder, exist_ok=True)
        for image_path in self.image_paths:
            # Update frequency image
            self.update_frequency_image(image_path)
            self.window.update()
            time.sleep(0.5)  # Adjust the wait time as needed
            # Capture frame from 'raw' stream
            frame = self.vid.capture_array("raw")
            if frame is not None:
                print(f"Captured frame dtype: {frame.dtype}")
                print(f"Captured frame shape: {frame.shape}")
                filename = os.path.basename(image_path)
                output_filename = filename.replace('.png', '.raw').replace('.jpg', '.raw')
                output_path = os.path.join(captured_images_folder, output_filename)
                # Save frame as RAW image
                frame.tofile(output_path)
                print(f"Captured image saved: {output_path}")
            else:
                print("Error capturing image.")

    def update_frequency_image(self, image_path=None):
        """
        Updates the frequency image displayed in the window.
        """
        if image_path:
            try:
                self.current_image_index = self.image_paths.index(image_path)
            except ValueError:
                # image_path not found in self.image_paths
                pass
        else:
            image_path = self.image_paths[self.current_image_index]

        try:
            self.frequency_image = Image.open(image_path)
            # Keep a reference to the original image
            self.original_frequency_image = self.frequency_image.copy()

            # Resize the image to fit the current canvas size
            canvas_width = self.frequency_canvas.winfo_width()
            canvas_height = self.frequency_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Resize the image to fill the canvas while maintaining aspect ratio
                self.frequency_image = self.original_frequency_image.resize((canvas_width, canvas_height), Image.ANTIALIAS)
                self.frequency_photo = ImageTk.PhotoImage(self.frequency_image)
                self.frequency_canvas.delete("all")
                self.frequency_canvas.create_image(0, 0, anchor=tk.NW, image=self.frequency_photo)
                self.frequency_canvas.image = self.frequency_photo  # Keep reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def load_reference_images(self):
        """
        Loads the reference images and calculates IAC and IDC.
        """
        try:
            reference_phantom_images = self.load_images_from_folder(self.reference_phantom_folder)
            self.iac_reference_phantom = self.calculate_iac(reference_phantom_images, 2)
            self.iac_reference_phantom_0 = self.calculate_iac(reference_phantom_images, 1)

            # Avoid division by zero
            epsilon = 1e-8
            self.iac_reference_phantom = np.where(self.iac_reference_phantom == 0, epsilon, self.iac_reference_phantom)
            self.iac_reference_phantom_0 = np.where(self.iac_reference_phantom_0 == 0, epsilon, self.iac_reference_phantom_0)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load reference images: {e}")
            self.iac_reference_phantom = None
            self.iac_reference_phantom_0 = None

    def calculate_mu_maps(self):
        """
        Calculates mu_a and mu_s maps using the rational function and Rd models.
        """
        print("Calculating μ_a and μ_s maps...")

        # Get mu_a and mu_s values
        try:
            mu_a_value = float(self.mu_a_value_entry.get())
            mu_s_value = float(self.mu_s_value_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for μ_a and μ_s.")
            return

        # Get Results Folder Name
        results_folder_name = self.results_folder_entry.get().strip()
        if not results_folder_name:
            messagebox.showerror("Input Error", "Please enter a valid Results Folder Name.")
            return

        # Define the results directory
        results_base_dir = os.path.join("results", results_folder_name)
        os.makedirs(results_base_dir, exist_ok=True)

        # Create subfolders for reference_phantom and saved_pictures
        reference_phantom_results_dir = os.path.join(results_base_dir, "reference_phantom")
        saved_pictures_results_dir = os.path.join(results_base_dir, "saved_pictures")
        os.makedirs(reference_phantom_results_dir, exist_ok=True)
        os.makedirs(saved_pictures_results_dir, exist_ok=True)

        # Check if Rd models are available
        if self.model_M1 is None or self.model_M2 is None:
            messagebox.showerror("Error", "Rd prediction models are not loaded. Please ensure the models are available.")
            return

        # Load images
        try:
            saved_pictures_images = self.load_images_from_folder(self.save_directory)
            # Reference images have already been loaded in self.load_reference_images()
            if self.iac_reference_phantom is None or self.iac_reference_phantom_0 is None:
                messagebox.showerror("Error", "Reference images are not loaded. Please capture the reference images first.")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load images: {e}")
            return

        # Get ROI coordinates
        roi_coords = self.get_roi_coordinates()
        if not roi_coords:
            # ROI selection failed
            return
        x, y, dx, dy = roi_coords

        # Crop images to ROI
        saved_pictures_images_roi = [img[y:y+dy, x:x+dx] for img in saved_pictures_images]

        # Calculate IAC and IDC within ROI
        iac_saved_pictures = self.calculate_iac(saved_pictures_images_roi, 2)
        iac_saved_pictures_0 = self.calculate_iac(saved_pictures_images_roi, 1)

        # Check if IAC calculations are successful
        if iac_saved_pictures is None or iac_saved_pictures_0 is None:
            messagebox.showerror("Processing Error", "Failed to calculate IAC.")
            return

        # Calculate Rd for the phantom using the models
        input_data = np.array([[mu_a_value, mu_s_value]])
        self.Rd_Phantom_DC = self.model_M1.predict(input_data)[0]
        self.Rd_Phantom_AC = self.model_M2.predict(input_data)[0]

        # Calculate Rd for the samples
        Rd_sample_ac = np.array((iac_saved_pictures / self.iac_reference_phantom[y:y+dy, x:x+dx]) * self.Rd_Phantom_AC)
        Rd_sample_dc = np.array((iac_saved_pictures_0 / self.iac_reference_phantom_0[y:y+dy, x:x+dx]) * self.Rd_Phantom_DC)

        # Process images to obtain mu_a and mu_s maps
        mu_a_map, mu_s_map = self.process_images(Rd_sample_dc, Rd_sample_ac)

        # Apply median filter to remove outliers
        mu_a_map_filtered = self.remove_outliers_with_median_filter(mu_a_map, kernel_size=5)
        mu_s_map_filtered = self.remove_outliers_with_median_filter(mu_s_map, kernel_size=5)

        # Save the results in the specified results directory
        self.save_results(mu_a_map, mu_s_map, mu_a_map_filtered, mu_s_map_filtered, results_base_dir)

        # Copy reference images and saved pictures to results directory
        try:
            # Copy reference images
            for filename in os.listdir(self.reference_phantom_folder):
                src_path = os.path.join(self.reference_phantom_folder, filename)
                dest_path = os.path.join(reference_phantom_results_dir, filename)
                shutil.copy(src_path, dest_path)

            # Copy saved pictures
            for filename in os.listdir(self.save_directory):
                src_path = os.path.join(self.save_directory, filename)
                dest_path = os.path.join(saved_pictures_results_dir, filename)
                shutil.copy(src_path, dest_path)

            print(f"Copied reference images and saved pictures to {results_base_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy images to results directory: {e}")

        # Show toast notification
        toast_message = "μₐ calculation has finished.\nYou can now upscale the image resolution to 2592x1944."
        self.show_toast(toast_message)

        print("Processing complete.")

    def load_images_from_folder(self, folder):
        """
        Loads RAW images from a directory and converts to grayscale.
        """
        images = []
        for root, dirs, files in os.walk(folder):
            raw_files = [f for f in files if f.endswith('.raw')]
            for filename in sorted(raw_files):
                raw_path = os.path.join(root, filename)
                # Load the RAW image
                raw_image = np.fromfile(raw_path, dtype=np.uint16)
                # Assume dimensions are known or inferred
                height, width = self.image_height, self.image_width
                try:
                    raw_image = raw_image.reshape(height, width)
                except ValueError:
                    messagebox.showerror("Error", f"Image {filename} has incorrect dimensions.")
                    continue

                # Normalize and convert to grayscale
                max_value = (2 ** 10) - 1  # For 10 bits
                image_16bit = (raw_image / max_value * 65535).astype(np.uint16)
                try:
                    image_gray = cv2.cvtColor(image_16bit, cv2.COLOR_BAYER_RG2GRAY)
                except cv2.error as e:
                    messagebox.showerror("Error", f"Color conversion failed for {filename}: {e}")
                    continue

                images.append(image_gray.astype(np.float32))

        if not images:
            raise FileNotFoundError("No RAW images found in the directory.")

        return images

    def calculate_iac(self, image_list, number):
        """
        Calculates the alternating current intensity (IAC) from the images.
        """
        start_index = (number - 1) * 3
        image_array = np.array(image_list[start_index:start_index + 3])
        if image_array.shape[0] < 3:
            messagebox.showerror("Error", f"Not enough images to calculate IAC for set {number}.")
            return None
        img1, img2, img3 = image_array
        sqrt2_over_3 = (2 ** 0.5) / 3
        term1 = (img1 - img2) ** 2
        term2 = (img2 - img3) ** 2
        term3 = (img3 - img1) ** 2
        return sqrt2_over_3 * np.sqrt(term1 + term2 + term3)

    def rational_function_mu_a(self, DC_flat, AC_flat):
        """
        Calculates mu_a using the rational function.
        """
        a0, a1, a2, a3 = self.params_mu_a['a0'], self.params_mu_a['a1'], self.params_mu_a['a2'], self.params_mu_a['a3']
        b0, b1, b2, b3 = self.params_mu_a['b0'], self.params_mu_a['b1'], self.params_mu_a['b2'], self.params_mu_a['b3']
        numerator = a0 + a1 * DC_flat + a2 * AC_flat + a3 * DC_flat * AC_flat
        denominator = 1 + b0 * DC_flat + b1 * AC_flat + b2 * DC_flat ** 2 + b3 * AC_flat ** 2
        return numerator / denominator

    def rational_function_mu_s(self, DC_flat, AC_flat):
        """
        Calculates mu_s using the rational function.
        """
        a0, a1, a2, a3 = self.params_mu_s['a0'], self.params_mu_s['a1'], self.params_mu_s['a2'], self.params_mu_s['a3']
        b0, b1, b2, b3 = self.params_mu_s['b0'], self.params_mu_s['b1'], self.params_mu_s['b2'], self.params_mu_s['b3']
        numerator = a0 + a1 * DC_flat + a2 * AC_flat + a3 * DC_flat * AC_flat
        denominator = 1 + b0 * DC_flat + b1 * AC_flat + b2 * DC_flat ** 2 + b3 * AC_flat ** 2
        return numerator / denominator

    def process_images(self, Rd_sample_dc, Rd_sample_ac):
        """
        Processes the images to obtain mu_a and mu_s maps using the rational function.
        """
        DC_flat = Rd_sample_dc.flatten()
        AC_flat = Rd_sample_ac.flatten()

        mu_a_flat = self.rational_function_mu_a(DC_flat, AC_flat)
        mu_s_flat = self.rational_function_mu_s(DC_flat, AC_flat)

        # Ensure non-negative values
        mu_a_flat = np.maximum(0, mu_a_flat)
        mu_s_flat = np.maximum(0, mu_s_flat)

        mu_a_map = mu_a_flat.reshape(Rd_sample_dc.shape)
        mu_s_map = mu_s_flat.reshape(Rd_sample_dc.shape)
        return mu_a_map, mu_s_map

    def save_results(self, mu_a_map, mu_s_map, mu_a_map_filtered, mu_s_map_filtered, results_base_dir):
        """
        Saves the mu_a and mu_s maps as .tif images in the specified results directory.
        """
        # Define filenames
        filenames = {
            'mu_a_original_tif': os.path.join(results_base_dir, "mu_a_map.tif"),
            'mu_s_original_tif': os.path.join(results_base_dir, "mu_s_map.tif"),
            'mu_a_filtered_tif': os.path.join(results_base_dir, "mu_a_map_filtered.tif"),
            'mu_s_filtered_tif': os.path.join(results_base_dir, "mu_s_map_filtered.tif"),
        }

        # Save .tif images
        tifffile.imwrite(filenames['mu_a_original_tif'], mu_a_map.astype(np.float32))
        tifffile.imwrite(filenames['mu_s_original_tif'], mu_s_map.astype(np.float32))
        tifffile.imwrite(filenames['mu_a_filtered_tif'], mu_a_map_filtered.astype(np.float32))
        tifffile.imwrite(filenames['mu_s_filtered_tif'], mu_s_map_filtered.astype(np.float32))

        print(f"Results saved in {results_base_dir}")

    def remove_outliers_with_median_filter(self, image, kernel_size=3):
        """
        Applies a median filter to the image to remove outliers.

        Parameters:
        - image: 2D NumPy array representing the image.
        - kernel_size: Size of the median filter kernel (must be odd).

        Returns:
        - Filtered image as a 2D NumPy array.
        """
        filtered_image = cv2.medianBlur(image.astype(np.float32), kernel_size)
        return filtered_image

    def continuous_measurement(self):
        """
        Captures images continuously for 15 seconds and processes them after.
        """
        print("Starting continuous measurement for 15 seconds...")

        # Get mu_a and mu_s values for the phantom
        try:
            mu_a_value = float(self.mu_a_value_entry.get())
            mu_s_value = float(self.mu_s_value_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for μ_a and μ_s.")
            return

        # Get Results Folder Name
        results_folder_name = self.results_folder_entry.get().strip()
        if not results_folder_name:
            messagebox.showerror("Input Error", "Please enter a valid Results Folder Name.")
            return

        # Define the results directory
        results_base_dir = os.path.join("results", results_folder_name)
        os.makedirs(results_base_dir, exist_ok=True)

        # Create subfolders for reference_phantom and saved_pictures
        reference_phantom_results_dir = os.path.join(results_base_dir, "reference_phantom")
        saved_pictures_results_dir = os.path.join(results_base_dir, "saved_pictures")
        os.makedirs(reference_phantom_results_dir, exist_ok=True)
        os.makedirs(saved_pictures_results_dir, exist_ok=True)

        # Calculate Rd for the phantom using the models
        input_data = np.array([[mu_a_value, mu_s_value]])
        self.Rd_Phantom_DC = self.model_M1.predict(input_data)[0]
        self.Rd_Phantom_AC = self.model_M2.predict(input_data)[0]

        # Initialize variables
        image_paths_cycle = cycle(self.image_paths)
        start_time = time.time()
        images_captured = []

        while time.time() - start_time < 15:
            # Capture images for one frequency
            for _ in range(6):  # Assuming 6 images (3 per frequency)
                image_path = next(image_paths_cycle)
                self.update_frequency_image(image_path)
                self.window.update()
                time.sleep(0.05)  # Small delay to ensure update
                # Capture frame from 'raw' stream
                frame = self.vid.capture_array("raw")
                if frame is not None:
                    images_captured.append(frame)
                else:
                    print("Error capturing frame.")

        print("Image capture complete. Processing images...")

        # Process the captured images
        processed_images = self.process_captured_images(images_captured)

        # Since we have multiple images captured over time, we need to process them accordingly
        # We can average the images for each pattern

        num_patterns = 6  # Number of patterns (3 per frequency)
        num_sets = len(processed_images) // num_patterns

        # Create lists to hold images for each pattern
        pattern_images = [[] for _ in range(num_patterns)]

        for i in range(len(processed_images)):
            pattern_index = i % num_patterns
            pattern_images[pattern_index].append(processed_images[i])

        # Average images for each pattern
        averaged_images = []
        for imgs in pattern_images:
            if imgs:
                averaged_image = np.mean(imgs, axis=0)
                averaged_images.append(averaged_image)
            else:
                print("No images for a pattern.")

        # Now proceed with the rest of the processing using the averaged images

        if len(averaged_images) != num_patterns:
            messagebox.showerror("Processing Error", "Incorrect number of averaged images.")
            return

        # Get ROI coordinates
        roi_coords = self.get_roi_coordinates()
        if not roi_coords:
            # ROI selection failed
            return
        x, y, dx, dy = roi_coords

        # Crop images to ROI
        averaged_images_roi = [img[y:y+dy, x:x+dx] for img in averaged_images]

        # Calculate IAC and IDC within ROI
        iac_images = self.calculate_iac(averaged_images_roi, 2)
        iac_images_0 = self.calculate_iac(averaged_images_roi, 1)

        # Check if IAC calculations are successful
        if iac_images is None or iac_images_0 is None:
            messagebox.showerror("Processing Error", "Failed to calculate IAC.")
            return

        # Load reference images
        if self.iac_reference_phantom is None or self.iac_reference_phantom_0 is None:
            messagebox.showerror("Error", "Reference images are not loaded. Please capture the reference images first.")
            return

        # Calculate Rd for the samples
        Rd_sample_ac = np.array((iac_images / self.iac_reference_phantom[y:y+dy, x:x+dx]) * self.Rd_Phantom_AC)
        Rd_sample_dc = np.array((iac_images_0 / self.iac_reference_phantom_0[y:y+dy, x:x+dx]) * self.Rd_Phantom_DC)

        # Process images to obtain mu_a and mu_s maps
        mu_a_map, mu_s_map = self.process_images(Rd_sample_dc, Rd_sample_ac)

        # Apply median filter to remove outliers
        mu_a_map_filtered = self.remove_outliers_with_median_filter(mu_a_map, kernel_size=5)
        mu_s_map_filtered = self.remove_outliers_with_median_filter(mu_s_map, kernel_size=5)

        # Save the results in the specified results directory
        self.save_results(mu_a_map, mu_s_map, mu_a_map_filtered, mu_s_map_filtered, results_base_dir)

        # Copy reference images and saved pictures to results directory
        try:
            # Copy reference images
            for filename in os.listdir(self.reference_phantom_folder):
                src_path = os.path.join(self.reference_phantom_folder, filename)
                dest_path = os.path.join(reference_phantom_results_dir, filename)
                shutil.copy(src_path, dest_path)

            # Copy saved pictures
            for filename in os.listdir(self.save_directory):
                src_path = os.path.join(self.save_directory, filename)
                dest_path = os.path.join(saved_pictures_results_dir, filename)
                shutil.copy(src_path, dest_path)

            print(f"Copied reference images and saved pictures to {results_base_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy images to results directory: {e}")

        # Show toast notification
        toast_message = "μₐ calculation has finished.\nYou can now upscale the image resolution to 2592x1944."
        self.show_toast(toast_message)

        print("Processing complete.")

    def process_captured_images(self, images_captured):
        """
        Processes captured RAW images.
        """
        processed_images = []
        for raw_image in images_captured:
            # Reshape raw_image to match the expected dimensions
            try:
                raw_image = raw_image.reshape(self.image_height, self.image_width)
            except ValueError:
                print("Error: Captured image has incorrect dimensions.")
                continue

            # Normalize and convert to grayscale
            max_value = (2 ** 10) - 1  # For 10 bits
            image_16bit = (raw_image / max_value * 65535).astype(np.uint16)
            try:
                image_gray = cv2.cvtColor(image_16bit, cv2.COLOR_BAYER_RG2GRAY)
            except cv2.error as e:
                print(f"Color conversion failed: {e}")
                continue

            processed_images.append(image_gray.astype(np.float32))
        return processed_images

    def close_app(self):
        """
        Closes the application.
        """
        if messagebox.askokcancel("Close", "Are you sure you want to close the application?"):
            self.continuous_measurement_running = False
            self.vid.stop()
            self.window.quit()

    def show_toast(self, message, duration=3000):
        """
        Displays a transient toast notification with the given message.

        Parameters:
        - message: The message to display in the toast.
        - duration: Duration in milliseconds before the toast disappears. Default is 3000 ms.
        """
        toast = tk.Toplevel(self.window)
        toast.overrideredirect(True)  # Remove window decorations
        toast.attributes("-topmost", True)  # Keep the toast on top

        # Calculate the position for the toast
        self.window.update_idletasks()  # Ensure accurate window size
        window_x = self.window.winfo_rootx()
        window_y = self.window.winfo_rooty()
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()

        # Set the toast position to bottom-right corner of the main window
        toast_width = 300
        toast_height = 80
        x = window_x + window_width - toast_width - 20  # 20 px padding from the edge
        y = window_y + window_height - toast_height - 20  # 20 px padding from the edge
        toast.geometry(f"{toast_width}x{toast_height}+{x}+{y}")

        # Create a frame to hold the message
        frame = tk.Frame(toast, bg="#444444")
        frame.pack(fill=tk.BOTH, expand=True)

        # Add the message label
        label = tk.Label(frame, text=message, fg="white", bg="#444444", justify=tk.LEFT, padx=10, pady=10, wraplength=280)
        label.pack(fill=tk.BOTH, expand=True)

        # Schedule the toast to close after the specified duration
        toast.after(duration, toast.destroy)

def main():
    root = tk.Tk()
    root.geometry("850x600")  # Adjusted window size for better layout
    sfdi_app = SfdiApp(root, "SFDI App")
    root.mainloop()

if __name__ == "__main__":
    main()
