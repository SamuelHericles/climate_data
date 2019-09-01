import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

class mlp(object):
    def __init __ (self, x, y):
        self.x = x
        self.y = y

    def test_mlp(self,i,j):
        warnings.filterwarnings('ignore')

        train_x,test_x,train_y,test_y = train_test_split(slef.x,self.y,test_size=0.2)

        MLP_reg = MLPRegressor(hidden_layer_sizes = (i*30,j*30), activation='tanh',
                          verbose=True, warm_start=True,
                          max_iter = 300)

        MLP_reg.fit(train_x,train_y)#Kernel default : RBF 
        predicts = MLP_reg.predict(test_x)

        rmse = math.sqrt(mean_squared_error(test_y,predicts))
        score_r2 = r2_score(test_y,predicts)

        return self.i*30,self.j*30,self.rmse,self.score_r2
'''
# Function for try mlp with each parameters
def net_mlp(i,j):

    warnings.filterwarnings('ignore')

    MLP_reg = MLPRegressor(hidden_layer_sizes = (i*30,j*30), activation='tanh',
                      verbose=False, warm_start=False,
                      max_iter = 300)

    return MLP_reg  


# Function for measure results
def train_test_score_var(x,i,j):

    MLP_reg = net_mlp(i,j)

    train_x,test_x,train_y,test_y = train_test_split(x,var_wind_speed,
                                                     test_size=0.2)

    MLP_reg.fit(train_x,train_y)#Kernel default : RBF 
    predicts = MLP_reg.predict(test_x)

    rmse = math.sqrt(mean_squared_error(test_y,predicts))
    score_r2 = r2_score(test_y,predicts)

    return score_r2,rmse


# Do net_mlp function with each parameters  
def testing_all_of_them(i,j):

    r2_temp_inst,rmse_temp_inst            = train_test_score_var(var_temp_inst,i,j) 
    r2_umid_inst,rmse_umid_inst            = train_test_score_var(var_umid_inst,i,j)
    r2_pto_orvalho_inst,rmse_pto_orvalho   = train_test_score_var(var_dew_inst,i,j)
    r2_hour,rmse_hour                      = train_test_score_var(var_hour,i,j) 
    r2_pressure,rmse_pressure              = train_test_score_var(var_pressure,i,j) 
    r2_wind_guest,rmse_wind_guest          = train_test_score_var(var_wind_guest,i,j)
    r2_radiation,rmse_radiation            = train_test_score_var(var_radiation,i,j)
    r2_precipitation,rmse_precipitation    = train_test_score_var(var_precipitation,i,j)
    r2_diff_temp,rmse_diff_temp              = train_test_score_var(var_diff_temp,i,j) 
    r2_diff_umid,rmse_diff_umid              = train_test_score_var(var_diff_umid,i,j)
    r2_diff_dew,rmse_diff_dew                = train_test_score_var(var_diff_dew,i,j)
    r2_diff_pressure,rmse_diff_pressure      = train_test_score_var(var_diff_pressure,i,j)

    return r2_temp_inst,rmse_temp_inst,r2_umid_inst,rmse_umid_inst,r2_pto_orvalho_inst,rmse_pto_orvalho,r2_hour,rmse_hour,r2_pressure,rmse_pressure,r2_wind_geust,rmse_wind_guest,r2_radiation,rmse_radiation,r2_precipitation,rmse_precipitation,r2_diff_temp,rmse_diff_temp,r2_diff_umid,rmse_diff_umid,r2_diff_dew,rmse_diff_dew, r2_diff_pressure,rmse_diff_pressure
'''
if(__name__ == "__main__"):
    test_mlp(x,y,i,j)