# Rep Right: AI Fitness Trainer

This project provides an AI-powered fitness trainer capable of analyzing and classifying various exercises such as squats, bicep curls, and shoulder raises in real-time using a webcam or pre-recorded videos. In addition to analyzing these exercises, the system also incorporates a model that classifies exercises into squats, bicep curls, push-ups, and shoulder presses, offering more comprehensive workout analysis and feedback.

## Key features:

1. Real-Time Exercise Analysis: The AI model analyzes each exercise rep, distinguishing between correct and incorrect form.
2. Rep Feedback: The system provides feedback on individual reps, helping users improve their form and avoid mistakes.
3. Rep Accuracy Tracking: A dashboard displays a line chart comparing the count of correct vs. incorrect reps for each exercise.
4. Progression Monitoring: Another dashboard tracks the progression of weights over time, allowing users to monitor their strength gains.

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
You can run the real-time exercise analysis using the `📷️_Live_Stream.py` script or pre-record your videos and upload using the `2_ ⬆️_Upload_Video.py`.

```bash
streamlit run .\🏠️_Demo.py
```

This will open a Streamlit web app in your browser.
The app will process the live webcam feed, or an uploaded video, and allow you to analyse the workout.
### Exercise Classification
To classify different exercises (bicep curls, shoulder presses, puhsups and squats), you can either upload a video, or use live camera, and select auto-detect exercise from the dropdown, and the model will classify the exercise.
The app will then let you analyse your own form. (Note:The system focuses on three key exercises: bicep curls, squats, and shoulder raises, ensuring accurate form analysis and providing valuable feedback for users to improve their technique.)

### Form analysis and Real Time Feedback
After auto-detection, you can select which exercise to analyse the form of.
The app will let you know how many correct and incorrect reps you have done, and it will also let you know what is going wrong, in form of live visual feedback and audio feedback

### User-Interface
The app has a simple user interface with the following features:
- **Upload Video**: Upload a video to analyze.
- **Live Stream**: Use the webcam to analyze the workout in real-time.
- **Dashboard**: Analyses data from your finished workouts, and shows simple statistics in visual format


## Tech Stack

| Technology    | Logo                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------- |
| **Streamlit** | <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="100"/>        |
| **Hugging Face** | <img src="https://huggingface.co/front/assets/huggingface_logo.svg" width="100"/>       |
| **MediaPipe** | <img src="https://github.com/Sarvesh521/Rep-Right/blob/79e4584c01ffaecfad59a995a50975b8a26734a2/MediaPipe.png" width="100"/>                       |
| **OpenCV**    | <img src="https://opencv.org/wp-content/uploads/2020/07/OpenCV_logo_white_600x.png" width="100"/> |
| **PyTorch**   | <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="100"/>                |
| **Python**    | <img src="https://www.python.org/static/community_logos/python-logo.png" width="100"/>     |



