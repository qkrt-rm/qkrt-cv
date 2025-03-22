import pyrealsense2 as rs
import cv2
import numpy as np
from datetime import datetime
import os


def plot_points(event, x, y, flags, param):
    """Function to plot 4 points for annotation."""
    global points, img_display, current_class
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        color = (0, 0, 255) if current_class == 1 else (255, 0, 0)
        cv2.circle(img_display, (x, y), 5, color, -1)
        cv2.imshow('Captured Image', img_display)
        if len(points) == 4:
            annotations.append((points.copy(), current_class))
            # Draw the polygon on the image display
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_display, [pts], isClosed=True, color=color, thickness=2)
            cv2.imshow('Captured Image', img_display)
            points.clear()

def redraw_annotations():
    """Redraw all annotations on the display image."""
    global img_display
    img_display = image.copy()
    for annotation in annotations:
        points, label = annotation
        color = (0, 0, 255) if label == 1 else (255, 0, 0)
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_display, [pts], isClosed=True, color=color, thickness=2)
        for x, y in points:
            cv2.circle(img_display, (x, y), 5, color, -1)
    cv2.imshow('Captured Image', img_display)

# Initialize global variables
points = []
annotations = []
image = None
image_name = ""
current_class = 1  # Default class is "red" (1)

# Create dataset folder structure
dataset_dir = "dataset"
os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, "labels"), exist_ok=True)

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    print("\n------Dataset Creator Controls------------\n Press 'c' to capture an image. Press 'r' for class red, 'b' for class blue. Press 'd' to delete the last annotation. Press 'q' to close the application.\n\n\n")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())
        cv2.imshow('Camera Feed', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Reset variables for the new image
            points.clear()
            annotations.clear()
            image = frame.copy()
            img_display = image.copy()
            # Generate image name based on current date and time
            image_name = datetime.now().strftime("%Y%m%d_%H%M%S.png")
            image_path = os.path.join(dataset_dir, "images", image_name)
            cv2.imshow('Captured Image', image)
            print("Image captured. Plot 4 points.")
            cv2.setMouseCallback('Captured Image', plot_points)

        elif key == ord('r'):
            current_class = 1  # Red class
            print("Class set to Red (1)")

        elif key == ord('b'):
            current_class = 2  # Blue class
            print("Class set to Blue (2)")

        elif key == ord('d'):
            if annotations:
                annotations.pop()
                print("Deleted the last annotation.")
                redraw_annotations()
            else:
                print("No annotations to delete.")

        elif key == ord('s'):
            # Save the image without annotations
            cv2.imwrite(image_path, image)
            print(f"Saved {image_path}.")
            # Write annotations info to a text file
            labels_path = os.path.join(dataset_dir, "labels", f"{image_name.split('.')[0]}.txt")
            with open(labels_path, "w") as file:
                for annotation in annotations:
                    points, label = annotation
                    points_str = " ".join([f"{x} {y}" for x, y in points])
                    file.write(f"{label} {points_str}\n")
            print(f"Annotations saved to {labels_path}.")
            annotations.clear()  # Clear annotations after saving

        elif key == ord('q'):
            print("Exiting...")
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()