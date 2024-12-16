This project showcases the real-time facial emotion recognition implementation using the deepface library and OpenCV. The goal is to capture live video from a webcam, detect faces in the video stream, and predict the emotions for each recognized face. These predicted emotions are displayed in real-time on the video frames.

To simplify the process, we use the deepface library, a deep learning tool for facial analysis that utilizes pre-trained models for precise emotion detection. The underlying deep learning operations are powered by TensorFlow, while OpenCV, an open-source computer vision library, is used for image and video processing.
Initial Setup:
Clone the repository: Execute git clone
 https://github.com/adityarv09/Facial-Emotion-Recognition-with-OpenCV-and-DeepFace.git .

Navigate to the project directory: Run cd Facial-Emotion-Recognition-using-OpenCV-and-DeepFace.

Install required dependencies:

Option 1: Use pip install -r requirements.txt.

Option 2: Install dependencies individually:
pip install deepface
pip install opencv-python
Obtain the Haar cascade XML file for face detection:

Download the haarcascade_frontalface_default.xml file from the OpenCV GitHub repository.
Execute the code:

Run the Python script.-->python emotion.py
The webcam will activate, initiating real-time facial emotion detection.
Emotion labels will be superimposed onto the frames containing recognized faces.
