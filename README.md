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
<p align="center">
![Screenshot 2024-12-24 143312](https://github.com/user-attachments/assets/52b38c09-e7d7-4351-a221-77ba6a5df98f)

Fig. 1.	Real-time image of class “V” detected as class “V”\
</p>
Our model has successfully recognized BISINDO alphabet gestures using hand landmark classification. The model achieved a high accuracy of 97–98% and showed reliable real-time performance even in noisy environments.\
<p align="center">
![Screenshot 2024-12-24 143008](https://github.com/user-attachments/assets/6147254f-2137-4e15-9f7d-3324757fd506)

Fig. 2.	Graph of average training and validation accuracy\

![Screenshot 2024-12-24 143614](https://github.com/user-attachments/assets/a4578438-59c1-4ed5-b952-4e0ef9e33452)

Fig. 3.	Confusion matrix
</p>



