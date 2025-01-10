# -Eye-Contact-Analysis-Video-Based-analysis-
Objective: Develop a script using Python and OpenCV to calculate the percentage of time a candidate maintains eye contact during an interview.
# Import the OpenCV library
import cv2

# Define the function to calculate eye contact
def calculate_eye_contact():
    # Open a connection to the default webcam (0)
    cap = cv2.VideoCapture(0)
    
    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return None

    # Set a fixed frame rate for accurate time calculation
    fixed_fps = 30  
    
    # Initialize variables to keep track of time, frames, eye contact frequency, and duration
    total_time_video = 0
    eye_contact_frames = 0
    total_frames = 0
    eye_contact_duration = 0

    # Load pre-trained Haar cascade classifiers for face and eye detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Start capturing video from the webcam
    while cap.isOpened():
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        # Check if the frame was read successfully
        if not ret:
            print("Error: Unable to read frame from camera.")
            break

        # Increment the total frames counter
        total_frames += 1

        # Convert the frame to grayscale for face and eye detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) corresponding to the face
            roi_gray = gray[y:y + h, x:x + w]

            # Detect eyes within the face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            
            # Check if both eyes are detected (assuming both eyes indicate eye contact)
            if len(eyes) >= 2:
                # Increment eye contact frames and update eye contact duration
                eye_contact_frames += 1
                eye_contact_duration += 1 / fixed_fps

        # Display the frame in a window named 'Frame'
        cv2.imshow('Frame', frame)

        # Calculate total time based on the frame rate
        total_time_video = total_frames / fixed_fps

        # Check for key press to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Check if any frames were captured
    if total_time_video == 0:
        print("Error: No frames captured from the camera.")
        return None

    # Calculate eye contact percentage
    eye_contact_percentage = (eye_contact_frames / total_frames) * 100

    # Print the results
    print(f"Total time of the video: {total_time_video:.2f} seconds")
    print(f"Eye contact frequency: {eye_contact_frames} frames")
    print(f"Eye contact duration: {eye_contact_duration:.2f} seconds")
    
    # Return the calculated eye contact percentage
    return eye_contact_percentage

# Check if the script is being run as the main program
if __name__ == "__main__":
    # Call the calculate_eye_contact function and store the returned eye contact percentage
    eye_contact_percentage = calculate_eye_contact()
    
    # Print the eye contact percentage if it is not None
    if eye_contact_percentage is not None:
        print(f"Eye contact percentage: {eye_contact_percentage:.2f}%")

