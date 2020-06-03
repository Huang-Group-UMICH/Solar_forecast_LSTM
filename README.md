# LSTM Solar Forecasting Model

## I. Model
### 1. Overview 
  This model is to forecast the solar irradiance of the next several hours according to the history data. The model not only uses the history data of solar irradiance, but also other related data, such as relative humidity, cloud, etc., to make predictions.
### 2. Structure of the Model
  The structure of the model is not fixed, you can change it by modifying the code in *utils/utils.py*.

  By default, the model has a normalization layer, one LSTM layer, two dense layers, and a RELU activation layer.

  The normalization layer is to set all data into the range of [0, 1]. For example, the solar irradiance may range from 0 to 1500, while the cloud fraction is in [0, 1]. A mild change of solar irradiance may much more strongly affect the result than a drastic change of cloud fraction, which is apparently unintended. Therefore, to make all parameters have equal influences to the result, we use normalization to let all data be in the range of [0, 1]. In addition, the initial parameters of LSTM and dense layers are set very small. A large value may cause vanishing gradient so it is necessary to normalize the data even if there is only one input factor.

  The LSTM layer has a input shape of (*t_x* - *num_to_pred*, *num_of_parameters*), where *t_x* is the total steps, *num_to_pred* is the number of hours whose values are to be predicted by the model, and *num_of_parameters* is the number of factors which may relate to the result of solar irradiance. For example, we have (*t_x*, *num_to_pred*, *num_of_parameters*) = (40, 8, 7). Then we are going to use the first 32(=40-8) hours' data to predict the next 8 hours. We are also using 3 factors to do the predictions, say solar irradiance, cloud fractions and relative humidity.

  The LSTM layer outputs a vector of parameters, which are the factors of the solar irradiance that the model learns by itself. The dense layers are to generate the final results according to these factors.

  Finally, there is a RELU activation layer, which ensures that the output will be not less than zero.

  ### 3. Parameters to Change the Structure of the model

  There are several configuration parameters stored in *config* file that can be modified. "*t_x*" and "*num_to_pred*" have been mentioned above. "*out_d*" represents the number of features that are learned by the model. It should not be set either too large or too small. If it is too large, it will take a really long time to train the model; if it is too small, the model will not be able to give accurate predictions.

  ### 4. Hyper-parameters

  Hyper-parameters are also stored in *config* file. As the trained model can be saved to local, we recommend you to set a not too large epoch to keep track of the loss. When it goes wrong, use "ctrl+c" to stop training. After and only after the program goes through all epochs, the model and the weights will be saved to local.

  We also recommend a higher learning rate (*lr* in the file), for example 0.1, and a smaller batch size, for example, 100, at the beginning of the training. When the loss goes down and "gets stuck", set a lower learning rate(0.001) and larger batch size(500-2000).



## II. How to use the model
 
### First, do configurations: 
  The configurations are in the "config" file.  
   
  "data_file": "data/NREL_data_2010_2014_1hr.mat",  #####  the dataset used

  "t_x" : 40,                                       #####  means the number of data, t_x - num_to_pred is the number of previous time steps of data
  "num_to_pred": 8,                                 #####  the number of next time steps to be predicted
  "out_d" :100,                                     #####  the output dimension of the rnn units 
  "num_of_dense1": 100,                             #####  the number of cells in dense layer 1 
  "dropout": 0.5,                                   #####  a value to control the overfitting

  "epoch": 60000,                                   #####  number of loops when training
  "batch-size": 512,                                #####  the batch of data to feed during per training time
  "lr": 0.001,                                      #####  learning rate

  "file_model": "model_1hr_t_x_40_out_d_100_num_of_dense1_100.json",   ##### file to save the trained model for prediction
  "file_weights": "weights_1hr_t_x_40_out_d_100_num_of_dense1_100.h5"  ##### file to save the trained coefficients for prediction


If you change only the "epoch", "batch-size" or "lr", feel free to use the pre-trained model. If you want to change the model by changing the "t_x", "num_to_pred" or "out_d", you may use the "-n" or "--new" instruction. Otherwise, the changes of the configurations will be ignored.

### Then, run the model, such as following:
1. Erase the pre-trained model and train a new one, using only one feature:
  - > python main.py -n" or "python main.py --new
  
2. Use the pre-trained model in step 1, and continue the training:
  - > python main.py
  
3. Create a new model using all features to predict:
  - > python main.py -a -n
  
4. Continue training on the model in step 3:
  - > python main.py -a
  
5. Generate long-term predictions on the model using only one feature:
  - > python main.py -p -g2
 
6. Generate long-term predictions on the model using all feature:
  - > python main.py -a -p -g2
 
  The final RMSE of the test set is near 100, which seems to be a good result. 

###  The output data is in "output/" folder.

## III. Citation and contact
If you use our source code, please kindly cite the following reference: Chen, X., X. Huang, Y. Cai, H. Shen, and J. Lu, 2020: Intra-day forecast of ground horizontal irradiance using long short-term memory network (LSTM). J. Meteor. Soc. Japan, 98, https://doi.org/10.2151/jmsj.2020-048.

If you have any questions, suggestions, or bug reports, please email xiuchen@umich.edu.
