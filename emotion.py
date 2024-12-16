import cv2
import os
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Folder containing images
image_folder = "picture"

# List all images in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print(f"No images found in the folder: {image_folder}")
    exit()

for image_filename in image_files:
    image_path = os.path.join(image_folder, image_filename)
    
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image file '{image_filename}'. Skipping...")
        continue

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print(f"No faces detected in {image_filename}.")
    else:
        for (x, y, w, h) in faces:
            # Analyze the face using DeepFace
            try:
                analysis = DeepFace.analyze(image[y:y+h, x:x+w], actions=['emotion'], enforce_detection=False)

                # Access `dominant_emotion` safely
                if isinstance(analysis, list):
                    emotion = analysis[0]['dominant_emotion']
                else:
                    emotion = analysis['dominant_emotion']

                # Draw rectangle and emotion label
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error analyzing face in {image_filename}: {e}")

    # Resize the image to fit the screen
    screen_width, screen_height = 1366, 768  # Set your display resolution here
    image_resized = cv2.resize(image, (screen_width, screen_height), interpolation=cv2.INTER_AREA)

    # Display the resized image with emotion labels
    cv2.imshow(f"Emotion Detection - {image_filename}", image_resized)

    # Wait for any key press before moving to the next image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
