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

Semester project Part 3 First update
This has been an interesting foray into formal machine learning. The bulk of my time so far has been getting familiar with the workflow and working with PyTorch itself, troubleshooting the issues getting started and loading the data. As stated above, I took the data that I am using for this from a SQLite database. Using SQLiteStudio, I exported the data into a CSV.  I then uploaded it to the Center for Research Computing (CRC). I talked with Adam about how the structure should look and am learning how to construct a Recurrent Neural Network, especially a Long Short-Term Memory model specifically. To get the ball rolling so to speak, I had ChatGPT build a simple LSTM model to ensure that the basics were working in submitting a job to the GPUs available on the CRC. Adam also sent the code for our fourth Practical which covers RNN and LSTM. I will be working through that and building my own, specifically fine tuning the various parameters and better understanding how to predict windspeed. The initial version of the code produced did the following

Writen by ChatGPT to describe how the LSTM was created:
This script builds and trains an LSTM (Long Short-Term Memory) neural network using PyTorch to predict atmospheric pressure (p_mb) based on sequential weather data from a CSV file (data_try_1.csv). It first loads the data, fills missing values, and normalizes key features (p_mb, t_c, rh_percent, z_m, ws_m_s) using MinMax scaling. The script then groups data by header_id, extracts time-series sequences, and prepares them as input for the LSTM. The model consists of two LSTM layers and a fully connected layer to predict the next pressure value in the sequence. It trains the model using Mean Squared Error (MSE) loss and the Adam optimizer, logging progress every 10 epochs. After training, the script makes a sample prediction and converts it back to the original scale. Throughout execution, key steps and results are logged to both the console and a log file (training_log.txt) for easy debugging and monitoring.

I noticed, however, that it did not take a validation set. As I had already uploaded the entire set of dropsondes to the system, I had the transformer change to automatically pick a random set for validation versus training. Also, this model predicts the pressure, which roughly correlates to the height and so may be a trivial exercise to start.  This is different than my original plan to more intentionally select the data set to help predict certain values, especially windspeed, say for example, in the case when hurricanes are intensifying.  That meta data is not directly in the dataset and will be added as a continuation of the work done as I continue to develop the system.  I did train the model for X epochs and got an accuracy of Y, which, again, for proof of concept to make sure that the “hello world” of neural network architecture for this project is somewhat successful. I will continue to work through and more directly write the LSTM as I develop the model, specifically training it to calculate windspeeds in different environments.  
Here are the results of the first run on the CRC
2025-03-06 01:48:20,290 - Epoch 1, Train Loss: 0.000000, Val Loss: 0.000000
2025-03-07 06:17:12,155 - Epoch 48, Train Loss: 0.000000, Val Loss: 0.000000
Clearly, I need to debug and see why the model is not providing any output.  This will be the start of next phase of this project.

## Semester project Part 4: Second update

### Instructions for running trained NN
The following is the results of training a Long-Short-Term-Memory Recurrent Neural Network (LSTM RNN) on the dropsonde data described above. 
In general, the model takes in a sample sequence start (defined by the trained sequence length) and loops through generating the next values to “predict” the wind speed values of the simulated drop. This is to create a view of a simulated drop that is predicted by the model. 
Since the last version, the dataset was better curated to fit the needs of the problem. Each actual ground truth dropsonde was taken and normalized as such. The height values z_m were put into 10 meter bins as another feature for bins at 0 meters to 200 meters.  Certain edge cases were handled as such. If any dropsonde was missing a single value in a bin, the average of the bins around it were taken.  If the dropsonde had two values in a bin, the average of those two values were used for that bin. If the dropsonde had more than two consecutive missing bin values, the dropsonde was removed. The data are located in the data_try_3.csv file.

To view the results of this training, run the LSTM_Model_dropsondes_viewer.py file which will create output from a random dropsonde header using the four models. 
The current setup was trained for 50 epochs for each of the four chosen sequence lengths [5, 10, 20, 25], using a hidden layer size of 64 and a learning rate of 0.001, giving 4 models to compare. These are plotted simultaneously on the graph saved ain the folder sequence_prediction_graphs with the name sequence_prediction_header_{random_header_id}.jpg. There are a few example graphs currently in the folder. 
As a note, future training will include varying the hidden layer size as well as the learning rates, while saving the best models for each run defined by having a minimum training loss. Also, graphs comparing the learning of the various model runs will be displayed.
### Classification and simulation accuracy computation
For assessing the model accuracy, I used a training set and a validation set split of 80% to 20% and a Mean Squared Error (MSE) comparison. This is a reasonable computation as the model should be predicting close to the next sequence values and calculating the MSE for this data set makes sense in terms of how close the predicted values  are to the actual values. The results, in text, of the different sequence length models are as follows:

--- Sequence length: 5 ---
Train loss: 0.004653
Validation loss: 0.005211

--- Sequence length: 10 ---
Train loss: 0.004697
Validation loss: 0.005363

--- Sequence length: 20 ---
Train loss: 0.004931
Validation loss: 0.005494

--- Sequence length: 25 ---
Train loss: 0.005258
Validation loss: 0.005641

These values are all on the same order of magnitude and suggest a good fitting of the model. However, as discussed at the end of this section, there may be a better way to define the cost function that helps train the network more accurately given the desired output.
The results can also be seen graphically in the file epoch_50_loss_comparison.jpg. 

### Classification and simulation accuracy computation
First, I will describe the general qualitative results of the model output as the project developed. A variety of setbacks and learning moments occurred that helped to improve the accuracy of the model. The first was the times that various normalizing and de-normalizing the dataset were needed. A few of the original graphs were comparing the actual data values with the model output which was still in a normed mode.  Next, since for each given sequence length the model uses the actual starting sequence, the model needs to use those values as the starting values when plotted.  After these changes were made, the simulated dropsonde windspeed measurements much more closely resembled the given dropsonde windspeed profiles in the windspeed variation profiles.  

Currently, however, there are two major issues that should be resolved before the final project update.  The first is that the model appears to have been trained “bottom up”, that is, the starting sequences are at the lowest height bins, starting from a z_m of 0.  In reality, the dropsondes are falling from the sky, and so the bins from the top should be the “starting sequences”.  Next, for the output dropsonde measurement predictions, the subsequent model values appear to be shifted from the starting values by a fairly large factor instead of roughly staying along with the actual sequence.  

Looking at the loss functions, the results seem to be fairly good by that measurement alone for all of the sequence lengths. This is confirmed intuitively by looking at the sample outputs and how they generally capture the shape of the actual data.

There are still a few things that I am not sure about and that I want to resolve before the final submission of this project. 

The first will be as stated above in regards to which direction the sequences are generated.  This may make a large difference in how closely the data are aligned and relate to the real world situation. 

Also, I am curious if using more values from above 200 meters would be useful to better capture variability. In the real use of the data in my research, we take an average of the first 500 meters in order to split the dropsondes into bins which we then use to calculate correlation coefficients. We calculate these coefficients using only the wind speeds from 200 meters and below which is why I for the purposes of proof of concept, I trained the model using this data. 
Next, I am not sure if the model is accurately generating the next sequences and that I am appending them correctly with the seed values from the actual dropsonde measurements.  This should be a relatively simple debugging issue that I should be able to ensure before the final tests (this may explain the significant difference after the seed values if the model output is just being started in the wrong place – a couple of my earlier results had this problem because I was appending the first five values for each regardless of the actual sequence length).

Finally, I am interested in seeing if modifying the loss function to more accurately calculate the cost of running the sequence all the way through would give better results. Right now, the loss function is only generated on how well it predicts the next value. If there is a way I could incorporate a cost related to the entire resulting predicted sequence, then the RNN might have to learn better the overall shape and not just the next value.

Overall, significant progress in accomplishing my task was done – starting from a model that did not give any output and had exploding gradients to now one that gives simulated dropsondes that look fairly realistic, I would say that this project is on the right track of achieving the goal stated at the outset.

## Semester project Part 5: Final update
### Instructions for running trained NN
The following is the (currently) final results of training a Long-Short-Term-Memory Recurrent Neural Network (LSTM RNN) on the dropsonde data described above. 
In general, the model takes in a sample sequence start (defined by the trained sequence length) and loops through generating the next values to “predict” the wind speed values of the simulated drop. This is to create a view of a simulated drop that is predicted by the model.  In the previous version, the model was trained “bottom up” (because the data was fed into the network in an ascending manner). In this attempt, the data fed into the network was sorted descending in order to simulate the starting conditions of the physical situation being emulated – the dropsonde is released from a plane and falls to the sea surface taking measurements. I also normalized with a Standard Normalization instead of a MinMax Scaler. Adam and I thought this would help prevent large outliers from skewing the data as many of the measurements are already normally distributed.
Improvements to this training set of the model include using a variety of hidden layer sizes: hidden_sizes = [8, 16, 32, 64] in addition to the sequence lengths: sequence_lengths = [5, 10, 20, 25], giving 16 different best models.
To view the results of this training, run the LSTM_Model_dropsondes_viewer.py file which will create output from a random dropsonde header using the 16 models. 
A few are already run in the sequence_prediction_graphs folder for attempt 6 at training the model. One (labeled good) is especially close at predicting the values.
### Test data base and results
After discussion with Adam, I decided to take the 10th epoch for each model as the models may have been overfitting after that. The best model by loss function amount is saved in the original file. This was chosen based on when the models more or less level out.
To assess the functioning of the model, due to project organization limitations at this point, I trained the model as described in class, first pulling aside 20% of the data to a side space, and then doing 80% of the remaining for the training and 20% for the validation.  Since this is just a random percentage of the data being chosen, it is reasonable that the models will work similarly on them.
### Classification accuracy
I then trained the models and (for ease of modifying the code), ran the resulting model on the test data. The results  are plotted in test_losses_at_each_step.png in the result_analyis. As can be seen there, the losses increase over time, to somewhere around 0.55 to 0.7 depending on the model. At epoch 10, the following values were achieved
| Hidden Size | Sequence Length | Val Loss |
|-------------|------------------|----------|
|      8      |        5         | 0.511715 |
|      8      |       10         | 0.514567 |
|      8      |       20         | 0.501905 |
|      8      |       25         | 0.497022 |
|     16      |        5         | 0.515176 |
|     16      |       10         | 0.507705 |
|     16      |       20         | 0.508266 |
|     16      |       25         | 0.522781 |
|     32      |        5         | 0.554142 |
|     32      |       10         | 0.535757 |
|     32      |       20         | 0.543974 |
|     32      |       25         | 0.524005 |
|     64      |        5         | 0.574050 |

| Hidden Size | Sequence Length | Test Loss |
|-------------|------------------|-----------|
|      8      |        5         | 0.511715  |
|      8      |       10         | 0.514567  |
|      8      |       20         | 0.501905  |
|      8      |       25         | 0.497022  |
|     16      |        5         | 0.515176  |
|     16      |       10         | 0.507705  |
|     16      |       20         | 0.508266  |
|     16      |       25         | 0.522781  |
|     32      |        5         | 0.554142  |
|     32      |       10         | 0.535757  |
|     32      |       20         | 0.543974  |
|     32      |       25         | 0.524005  |
|     64      |        5         | 0.574050  |
|     64      |       10         | 0.541613  |
|     64      |       10         | 0.562972  |
|     64      |       20         | 0.554225  |
|     64      |       25         | 0.515376  |

It was also interesting how much the model error blew up over time (suggesting major overfitting).
### Analysis of data and future implications
 While the validation and test data losses were similar, the actual outputs can be quite incorrect as seen in some of the examples that are printed and stored in the sequence prediction graphs.  Some matched quite well, and some were significantly different with a large jump from the first few starting values.  This is reasonable given how the data could fall in line with the other samples. In future work, I could separate out and normalize differently based on when the data is being taken. Also, I did not have time to implement a loss function that tracked the performance of the whole model compared to the original, and not just the next step. This could have helped my model more accurately adapt to the data and make better predictions on the test data.
Overall this was an excellent learning experience! I definitely learned a lot about how to construct data in a meaningful way with the different ways that I had to organize and separate and clean it. Then, thinking through the implementation of the recurrent neural network, took some time, but it was helpful to think through how this model that we studied in class could be used in application to my data.  Next, being able to run through different models and seeing how they worked and being able to compare the results was very helpful as well because now I feel like I could attempt much larger projects and compare and contrast the effects of different hyper parameters and network structure choices.  I am very appreciative for all of the help that was given to me throughout the process, and I know that I could not have gotten this far without all of it. Thanks for a great experience on this project!
