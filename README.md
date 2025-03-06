# NeuralNetwork_Hurricanes
A Neural Network simulating the aggregated measurements from various hurricanes, specifically for dropsondes, saildrones, and NOAA best track data.

## Semester project Part 1: Conceptual design
## Project Name: Neural Networks Characterizing Hurricanes

The goal of this project is to use the tools of Deep Neural Networks in order to fit and predict certain measured values taken from hurricane measurement instruments. The project will use data taken from three open data sources: the HURDAT 2 Hurricane Best Track Database (https://www.aoml.noaa.gov/hrd/hurdat/Data_Storm.html),  the Hurricane Research Division Dropsonde Database (https://www.aoml.noaa.gov/ftp/pub/hrd/data/dropsonde/), the and the (https://data.pmel.noaa.gov/generic/erddap/) The Pacific Marine Environmental Laboratory's ERDDAP Data of Saildrones located near hurricanes, (hereafter named HURDAT2, Dropsondes, and Saildrones respectively). The relevant data have been put into a Structured Query Language (SQL) database – specifically a SQLite3 format – so that it can be pulled into Pytorch easily. 

### Initiative 1
**Neural Network Characterizing Hurricanes Measurements**

Using the three major datasets (HURDAT2, Dropsondes, Saildrones), I will build a neural network that can predict measurements from sensors given various parameters throughout a hurricane’s life cycle. I will start with a Long short-term memory (LSTM) model followed by a Gated Recurrent Unit (GRU) as the data used comes in infrequent intervals and there are varying parameters that are measured.  I will test different layer widths and depths to see what the accuracy of the different approaches are.  I can use various sampling methods to determine my training data, validation data, and final testing data.

### Initiative 2
**Computer Vision Analysis of Saildrone Video**

Using computer vision techniques, I will analyze and gather quantitative measurements from saildrone videos (including significant wave height, turbulence, and windspeed/water current alignment). I will start by using a Convolutional Neural Network to see what general features are present in the video and an interpretation of the visual aspects. Then, I will use the results to feed into a RNN which will help me to see how the video can be analyzed in time and what insights can be taken from the video. There is a particularly interesting video that is taken through the eye of the hurricane that has wave direction and possibly wind direction information.  Using computer vision techniques, gathering correlations from the videos would be helpful and insightful to better predict this important measurement of sea surface – air correlation. For this, I may need to detect where the waves are going and which direction the wind is blowing, either through the video alone or in tandem with other measurements.

### Initiative 3
**Physics Informed Neural Networks (PINNS) for Large Eddy Simulations (LES)**

For possible future research, this project may investigate using PINNs, finding solutions and calculations for the Navier Stokes Equations in Hurricane Large Eddy Simulations which are computationally very expensive.

## Semester project Part 2: Datasets
## Initiative 1
### Source
A large number of dropsonde measurements were taken from the National Oceanic and Atmospheric Administration's (NOAA) “Dropsonde Data Archive.”  at https://www.aoml.noaa.gov/hrd/data_sub/dropsonde.html. 
### Number of distinct objects/subjects 
There are approximately 22000 drops that each have approximately 1000 measurements (taken at 2 to 4 Hz) of height, windspeed, relative humidity, pressure, among other variables. There is also a variety of metadata about the context of the drop (which hurricane it was dropped, the intensity at that time, the distance to the center of the hurricane).
### The Train and Validation Subsets
I will likely take a 60% subset for the train from each of category 3, 4, and 5 hurricanes as this is a major aspect of the hurricane intensity and its correlation to various other aspects. Similarly, the 20% for validation will be taken from these categories.
### Characterization of samples
Each measurement has a respective accuracy (height down to the meter, pressure to the tenth of a milibar, windspeed to the hundreth of meters per second). Note that if a measurement was not taken at that time, it is represented with a -999 which is converted to a NULL value in post processing.
### Sample of data
IX      t (s)  P (mb)    T (C)    RH (%)  Z (m)   WD    WS (m/s)   U (m/s)   V (m/s)  NS  WZ (m/s)  ZW (m)   FP  FT  FH  FW    LAT (N)   LON (E)
0072   17.75  -999.0   -999.00   -999.0    -999    55     8.79     -7.20     -5.04    10  -999.0    -999     0   0   0   0    -999.000  -999.000
0073   18.00   502.7     -3.87     57.8    5843    55     8.75     -7.20     -4.97    10    -0.9    5843     0   0   0   0      16.073   -55.990
0074   18.25  -999.0   -999.00   -999.0    -999    56     8.72     -7.21     -4.91    10  -999.0    -999     0   0   0   0    -999.000  -999.000
0075   18.50   503.2     -3.84     58.9    5836    56     8.70     -7.23     -4.84    10    -0.9    5836     0   0   0   0      16.073   -55.990
0076   18.75  -999.0   -999.00   -999.0    -999  -999  -999.00   -999.00   -999.00    10  -999.0    -999     0   0   0   0    -999.000  -999.000
0077   19.00   503.6     -3.82     60.0    5829  -999  -999.00   -999.00   -999.00    10    -0.9    5829     0   0   0   0      16.073   -55.990
0078   19.25  -999.0   -999.00   -999.0    -999    58     8.64     -7.29     -4.63    10  -999.0    -999     0   0   0   0    -999.000  -999.000
0079   19.50   504.1     -3.80     61.2    5822  -999  -999.00   -999.00   -999.00    10    -0.9    5822     0   0   0   0      16.073   -55.991

## Semester project Part 3: First update

Writen by ChatGPT to describe how the LSTM was created:
This script builds and trains an LSTM (Long Short-Term Memory) neural network using PyTorch to predict atmospheric pressure (p_mb) based on sequential weather data from a CSV file (data_try_1.csv). It first loads the data, fills missing values, and normalizes key features (p_mb, t_c, rh_percent, z_m, ws_m_s) using MinMax scaling. The script then groups data by header_id, extracts time-series sequences, and prepares them as input for the LSTM. The model consists of two LSTM layers and a fully connected layer to predict the next pressure value in the sequence. It trains the model using Mean Squared Error (MSE) loss and the Adam optimizer, logging progress every 10 epochs. After training, the script makes a sample prediction and converts it back to the original scale. Throughout execution, key steps and results are logged to both the console and a log file (training_log.txt) for easy debugging and monitoring.
