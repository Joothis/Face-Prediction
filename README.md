# Face Detection and Recognition System

## Overview

This project is a Python-based system for detecting and recognizing faces using **OpenCV**. It can capture face images, train a face recognition model, and recognize individuals in real-time or from saved images. The system is designed for simplicity and accuracy, leveraging powerful computer vision algorithms to identify human faces.

## Features

- **Face Detection**: Detects faces in images or live camera feeds using OpenCV's pre-trained **Haar Cascade Classifier**.
- **Face Data Collection**: Captures face images from a camera and stores them in a dataset for training.
- **Model Training**: Trains a face recognition model on the collected dataset using algorithms like **Local Binary Patterns Histograms (LBPH)**.
- **Real-Time Face Recognition**: Recognizes faces in real-time by comparing detected faces against the trained model.
- **Name Mapping**: Matches recognized faces to their corresponding names stored in a JSON file.



## Prerequisites

- Python 3.x
- OpenCV
- NumPy

To install the required libraries, run the following:

```bash
pip install opencv-python numpy
```
##Usage
1. Capture Face Images
To capture face images for training, run the capture_faces.py script. This will use your webcam to capture images and store them in the dataset/ folder.
```bash
python capture_faces.py
```

2. Train the Face Recognition Model
Once you have collected face images, train the recognition model by running the train.py script.
```bash
python train.py
```
This script processes the images in the dataset/ folder and saves the trained model to the trainer/ directory.

3. Recognize Faces
To recognize faces in real-time or from saved images, use the face_prediction.py script.
```
python face_prediction.py
```
The script will access your webcam, detect faces, and recognize any faces that match the trained model. The recognized faces will be labeled with the corresponding names from names.json.

Files
capture_faces.py: Captures images of faces from a webcam and stores them for training.
train.py: Trains a face recognition model using images from the dataset/ folder.
face_prediction.py: Recognizes faces in real-time or from saved images using the trained model.
haarcascade_frontalface_default.xml: A pre-trained model for face detection using OpenCV's Haar Cascade.
names.json: Stores mappings between recognized faces and their corresponding names.
Future Enhancements
Improve face recognition accuracy by experimenting with different algorithms.
Add support for recognizing multiple faces simultaneously.
Implement a GUI for easier interaction with the system.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

### Instructions:
1. Copy the entire text above.
2. Open a text editor.
3. Paste the text into a new file.
4. Save the file as `README.md` in your project directory.

This will give you a complete README file that explains your project, its usage, structure, and other details. Let me know if you need more help!
