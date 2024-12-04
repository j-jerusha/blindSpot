import cv2
import numpy as np
import pygame
from utils import load_model

# Initialize pygame for sound
pygame.mixer.init()

# Load the trained model
model = load_model('C:/Desktop/blindSpot/models/blind_spot_model.keras')

cap = cv2.VideoCapture(0) 

# Define the class names
class_names = ["Non-Vehicle", "Vehicle"]

# Initialize sound flag
sound_playing = False
alert_sound = pygame.mixer.Sound('C:/Desktop/blindSpot/bleep-censorship-sound-wav-74691.mp3')  # Path to your alert sound file

def load_and_preprocess_image(image):
    """Preprocess the input image."""
    img = cv2.resize(image, (128, 128))  # Resize to match model input shape
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def play_alert():
    """Play an alert sound."""
    global sound_playing
    if not sound_playing:  
        alert_sound.play(loops=-1)  
        sound_playing = True

def stop_alert():
    """Stop the alert sound."""
    global sound_playing
    if sound_playing:  # Stop sound only if playing
        alert_sound.stop()
        sound_playing = False

# Run the webcam feed
while True:
    ret, frame = cap.read()

    if not ret:
        break

   
    processed_image = load_and_preprocess_image(frame)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Display prediction results
    label = class_names[predicted_class]
    confidence = prediction[0][predicted_class]

    # Add text to the frame
    cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate proximity (size of detected object)
    height, width, _ = frame.shape
    proximity = height * width * (0.5 if predicted_class == 1 else 0.3)  # Placeholder for detecting proximity based on object size

    # Check conditions for playing the alert sound
    if confidence > 0.9 and proximity > 100000:  # Adjust the threshold for proximity as needed
        play_alert()
        if predicted_class == 1:
            cv2.putText(frame, "Alert: Vehicle Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Alert: Non-Vehicle Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        stop_alert()  # Stop alert if conditions are not met

    # Show the frame
    cv2.imshow('Blind Spot Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
stop_alert()  # Ensure sound is stopped when exiting
