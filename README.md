# Face and Hand Detection using MediaPipe & Streamlit

## Overview

This project is a real-time Face and Hand Landmark Detection application built using **Python**, **OpenCV**, **MediaPipe**, and **Streamlit**. The application captures live video from a webcam and detects facial landmarks, left-hand landmarks, and right-hand landmarks in real time.

The detected landmarks are displayed on the video feed along with the current FPS (Frames Per Second).

---

## Features

* Real-time webcam streaming
* Face landmark detection
* Left-hand landmark detection
* Right-hand landmark detection
* FPS monitoring
* Interactive Streamlit web interface
* Lightweight and fast execution

---

## Technologies Used

* Python
* OpenCV
* MediaPipe
* Streamlit

---

## Project Structure

```text
Face_Hand_Detection/
│
├── app.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## Installation

### Clone the Repository

```bash
git clone <your-github-repository-link>
cd Face_Hand_Detection
```

### Create Virtual Environment

```bash
python -m venv myenv
```

### Activate Virtual Environment

#### Windows

```bash
myenv\Scripts\activate
```

#### Linux / Mac

```bash
source myenv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
streamlit
opencv-python
mediapipe
```

---

## Running the Application

```bash
streamlit run app.py
```

After execution, Streamlit will automatically open the application in your browser.

---

## How It Works

1. Opens the webcam using OpenCV.
2. Captures video frames continuously.
3. Converts frames from BGR to RGB.
4. Processes frames using MediaPipe Holistic Model.
5. Detects:

   * Face Landmarks
   * Left Hand Landmarks
   * Right Hand Landmarks
6. Draws landmarks on the video feed.
7. Calculates and displays FPS.
8. Streams the output through Streamlit.

---

## Applications

* Gesture Recognition Systems
* Human-Computer Interaction
* Sign Language Recognition
* AR/VR Applications
* Face Tracking Systems
* AI-Powered Vision Applications

---

## Future Enhancements

* Hand Gesture Recognition
* Face Emotion Detection
* Face Identification System
* Screenshot Capture Feature
* Video Recording Support
* Multi-Person Tracking

---

## Author

**Vuyyuru Soma Sekhar**

Aspiring Data Analyst | Machine Learning Enthusiast | Computer Vision Learner


