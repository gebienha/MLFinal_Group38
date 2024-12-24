# Real-Time BISINDO Recognition with Dynamic Sentence Output

This project aims to be a stepping stone for increasing awareness of BISINDO among the broader society. By utilizing real-time BISINDO recognition with dynamic sentence output, this innovation seeks to promote inclusivity and bridge the communication gap between the deaf community and the general public.
## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all related packages.

```bash
pip install cv2
pip install numpy
pip install mediapipe
```
For any missing packages, install them using pip install (package name).
## Usage
1. First, run `keypoint_classification_EN.ipynb`. This file contains the training model for our hand landmark classification model. The dataset has already been provided in the `model` folder.
2. Then, run `app.py`. This file contains the main functions to run the sign language detection software. **Note: make sure to download OpenCV beforehand as it is needed to run this file**


## Output

An example output, if successful, should look like this:\
![classv](https://github.com/user-attachments/assets/baeb39ea-c3e6-4d17-8216-a6fb50f127b4)\
Fig. 1.	Real-time image of class “V” detected as class “V”

Our model has successfully recognized BISINDO alphabet gestures using hand landmark classification. The model achieved a high accuracy of 97–98% and showed reliable real-time performance even in noisy environments.\
![accuracy](https://github.com/user-attachments/assets/30b5c080-be6a-452d-92b5-73477f4d1468)\
Fig. 2.	Graph of average training and validation accuracy\
![confusionmatrix](https://github.com/user-attachments/assets/a66dee4b-90c5-4c50-9406-47e5bfd9ff81)\
Fig. 3.	Confusion matrix




