# Activity Recognition and Step Detection for Wearable Devices
## Project Description
This project uses real-user sensor data collected from wearable devices to classify different activities (e.g., walking, running, standing, cycling) and count the number of steps taken. The dataset consists of multi-sensor data (accelerometer, gyroscope, altitude) over time, and the solution applies signal processing and machine learning models to classify the activities and detect step counts.

## Contributors
Youmna Abboud, Cyrus Achtari, Jean-Claude Rihani

This project was completed at **ETH** as part of the **Mobile Health and Activity Monitoring** course.

## Data
The dataset consists of multiple .pkl files containing the following sensor data:
- Accelerometer (ax, ay, az): X, Y, and Z accelerometer data points.
- Gyroscope (gx, gy, gz): X, Y, and Z gyroscope data points.
- Altitude: Altitude data points.
The files are organized into directories and each file contains data for a specific participant.
Each .pkl file represents a set of sensor data that corresponds to a particular activity.
# Model Overview
The solution applies machine learning models and signal processing techniques to perform activity recognition and step detection:
## Preprocessing:
Bandpass filtering is applied to the sensor data to remove noise. The data is segmented into windows (e.g., 3000 samples per window) with feature extraction performed on each segment.
## Feature Extraction: 
The extracted features include statistical features (mean, variance, skewness, kurtosis), frequency-domain features (e.g., FFT), and time-domain features (e.g., energy, interquartile range).
## Activity Recognition:
A classifier (XGBoost in this case) is trained on the features from sensor data to predict the activity type (e.g., walking, running, standing).
## Step Detection:
The number of steps is detected by analyzing the filtered accelerometer data using peak detection methods.
## Feature Extraction
To extract meaningful features from the raw sensor data, the following techniques are used:
### Time-Domain Features:
Mean, variance, skewness, kurtosis, energy, interquartile range (IQR), and range (difference between max and min values).
### Frequency-Domain Features:
Fourier transform (FFT) is used to extract the frequency characteristics of the signal. Frequency-domain entropy is computed to capture the signal’s complexity.
### Other Features:
Inter-correlation between accelerometer and gyroscope signals.
Step count is detected by identifying peaks in the accelerometer signal after filtering.
# Results
The models achieve the following performance metrics on the validation set:
## Location-based activity classifier:
- Accuracy: 95%
- Balanced Accuracy: 94.88%

## Path-based activity classifier:
- Accuracy: 80%
- Balanced Accuracy: 78.81%

## Activity-based classifier:
- Accuracy: 94.11%
- Balanced Accuracy: 93.75%
# Data Source
Kaggle Activity Recognition Dataset

# Libraries
NumPy, SciPy, XGBoost, scikit-learn, and others for their excellent implementations of machine learning and signal processing algorithms.

