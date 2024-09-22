# Rep Right: AI Fitness Trainer

This project provides an AI-powered fitness trainer that can analyze and classify various exercises such as squats, bicep curls, and shoulder raises in real-time using a webcam or pre-recorded videos.

## Features

- **Real-time Squats Analysis**: Detects and analyzes squats using MediaPipe for pose detection.
- **Exercise Classification**: Uses a machine learning model to classify different exercises such as bicep curls, shoulder lateral side raises, and squats.
- **Video Download**: Allows users to download the processed video with annotations.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/Sarvesh521/Rep-Right
    cd Rep-Right
    ```

2. Install the required Python packages(, preferably in a [virtual environment](https://docs.python.org/3/library/venv.html)):
    ```bash
    pip install -r requirements.txt
    ```



## Usage

### Real-time Exercise Analysis
You can run the real-time exercise analysis using the `📷️_Live_Stream.py` script.

```bash
streamlit run .\🏠️_Demo.py
```

This will open a Streamlit web app in your browser.
The app will process the live webcam feed, or an uploaded video, and allow you to analyse the workout.
### Exercise Classification
To classify different exercises (bicep curls, shoulder presses, and squats), you can either upload a video, or use live camera, and select auto-detect exercise from the dropdown, and the model will classify the exercise.
The app will then let you analyse your own form.

### Form analysis and Real Time Feedback
After auto-detection, you can select which exercise to analyse the form of.
The app will let you know how many correct and incorrect reps you have done, and it will also let you know what is going wrong, in form of live visual feedback

### User-Interface
The app has a simple user interface with the following features:
- **Upload Video**: Upload a video to analyze.
- **Live Stream**: Use the webcam to analyze the workout in real-time.
- **Dashboard**: Analyses data from your finished workouts, and shows simple statistics in visual format
