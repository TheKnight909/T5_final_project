import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from huggingface_hub import hf_hub_download


# Color mapping for different classes
class_colors = {
    0: (0, 255, 0),    # Green (Helmet)
    1: (255, 0, 0),    # Blue (License Plate)
    2: (0, 0, 255),    # Red (MotorbikeDelivery)
    3: (255, 255, 0),  # Cyan (MotorbikeSport)
    4: (255, 0, 255),  # Magenta (No Helmet)
    5: (0, 255, 255),  # Yellow (Person)
}


def run_yolo(image):
    # Run the model on the image and get results
    results = model(image)
    return results

def process_results(results, image):
    # Draw bounding boxes and labels on the image
    boxes = results[0].boxes  # Get boxes from results
    for box in boxes:
        # Get the box coordinates and label
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
        conf = box.conf[0]  # Confidence score
        cls = int(box.cls[0])  # Class index
        label = model.names[cls]  # Get class name from index
        color = class_colors.get(cls, (255, 255, 255))  # Get color for class

        # Draw rectangle and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Draw colored box
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def process_image(uploaded_file):
    # Read the image file
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1))

    # Run YOLO model on the image
    results = run_yolo(image)

    # Process the results and draw boxes on the image
    processed_image = process_results(results, image)

    # Convert the image from BGR to RGB before displaying it
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display the processed image in Streamlit
    st.image(processed_image_rgb, caption='Detected Image', use_column_width=True)

# Cache the video processing to prevent reprocessing on reruns
@st.cache_data
def process_video_and_save(uploaded_file):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Get the path of the temporary file

    # Read the video file
    video = cv2.VideoCapture(temp_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    frames = []

    current_frame = 0
    start_time = time.time()  # Start the timer

    while True:
        ret, frame = video.read()
        if not ret:
            break  # Break the loop if there are no frames left

        # Run YOLO model on the current frame
        results = run_yolo(frame)

        # Process the results and draw boxes on the current frame
        processed_frame = process_results(results, frame)

        # Convert the frame from BGR to RGB before displaying
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        frames.append(processed_frame_rgb)  # Save the processed frame

        current_frame += 1

    video.release()

    # Create a video writer to save the processed frames
    height, width, _ = frames[0].shape
    output_path = 'processed_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame in frames:
        # Convert back to BGR for saving the video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)  # Write each processed frame to the video

    out.release()

    # Return the path of the processed video
    return output_path


def live_video_feed():
    stframe = st.empty()  # Placeholder for the video stream in Streamlit
    video = cv2.VideoCapture(0)  # Capture live video from the webcam

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Run YOLO model on the current frame
        results = run_yolo(frame)

        # Process the results and draw boxes on the current frame
        processed_frame = process_results(results, frame)

        # Convert the frame from BGR to RGB before displaying
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame in the Streamlit app
        stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)

        # Stop the live feed when the user clicks the "Stop" button
        if st.button("Stop"):
            break

    video.release()


def main():
    model_file = hf_hub_download(repo_id="TheKnight115/Yolov8m", filename="yolov8_Medium.pt")

    global model
    model = YOLO(model_file)

    st.title("Motorbike Violation Detection")

    # Create a selection box for input type
    input_type = st.selectbox("Select Input Type", ("Image", "Video", "Live Feed"))

    # Image or video file uploader
    if input_type == "Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Process the image
            process_image(uploaded_file)

    elif input_type == "Video":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov"])
        if uploaded_file is not None:
            # Process and save the video
            output_path = process_video_and_save(uploaded_file)

            # Display the processed video
            st.video(output_path)

            # Provide a download button for the processed video
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            st.download_button(label='Download Processed Video',
                               data=video_bytes, file_name='processed_video.mp4', mime='video/mp4')

    elif input_type == "Live Feed":
        st.write("Live video feed from webcam. Press 'Stop' to stop the feed.")
        live_video_feed()


if __name__ == "__main__":
    main()
