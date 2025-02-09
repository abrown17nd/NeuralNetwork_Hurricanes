# NeuralNetwork_Hurricanes
A Neural Network simulating the aggregated measurements from various hurricanes, specifically for dropsondes, saildrones, and NOAA best track data.

## Semester project Part 1 Conceptual design
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


