# -*- coding: utf-8 -*-
"""
Created on Tue Aug  26 10:49:35 2025
@author: nhand
"""

import numpy as np
from pickle import load
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#===============================================
# function to predict peak displacement using ANN model trained from the mixed ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_Mixed_InputScaler.pkl
#   2. S1To5_Mixed_OutputScaler.pkl
#   3. S1To5_Mixed_ANN_Model.keras
# mu= friction coefficient of the bearings
# Td= pendulum period of the bearings
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 1.0 s
# return expected peak displacement DM (in meter)
# usage example:
# DM = ANN_MixedGM(0.02, 2.0, np.array([9.135, 4.4718, 2.6858, 1.7622, 1.4265])/9.81)
def ANN_MixedGM(mu, Td, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_Mixed_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_Mixed_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([mu]), np.array([Td]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_Mixed_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the pulse-like ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_PulseLike_InputScaler.pkl
#   2. S1To5_PulseLike_OutputScaler.pkl
#   3. S1To5_PulseLike_ANN_Model.keras
# mu= friction coefficient of the bearings
# Td= pendulum period of the bearings
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 1.0 s
# return expected peak displacement DM (in meter)
# usage example:
# DM = ANN_PulseLikeGM(0.02, 2.0, np.array([9.135, 4.4718, 2.6858, 1.7622, 1.4265])/9.81)
def ANN_PulseLikeGM(mu, Td, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_PulseLike_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_PulseLike_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([mu]), np.array([Td]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_PulseLike_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the no-pulse ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S1To5_NoPulse_InputScaler.pkl
#   2. S1To5_NoPulse_OutputScaler.pkl
#   3. S1To5_NoPulse_ANN_Model.keras
# mu= friction coefficient of the bearings
# Td= pendulum period of the bearings
# S1To5 (measured in g)= spectral acceleration at 1s- to 5s periods, step= 1.0 s
# return expected peak displacement DM (in meter)
# usage example:
# DM = ANN_NoPulseGM(0.02, 2.0, np.array([9.135, 4.4718, 2.6858, 1.7622, 1.4265])/9.81)
def ANN_NoPulseGM(mu, Td, S1To5):
    # load scaler
    input_scaler= load(open('S1To5_NoPulse_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S1To5_NoPulse_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([mu]), np.array([Td]), np.ravel(S1To5)),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S1To5_NoPulse_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the mixed ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S3_Mixed_InputScaler.pkl
#   2. S3_Mixed_OutputScaler.pkl
#   3. S3_Mixed_ANN_Model.keras
# mu= friction coefficient of the bearings
# Td= pendulum period of the bearings
# S3 (measured in g)= spectral acceleration at 3-s period
# return expected peak displacement DM (in meter)
# usage example:
# DM = ANN_S3_MixedGM(0.02, 2.0, S3)
def ANN_S3_MixedGM(mu, Td, S3):
    # load scaler
    input_scaler= load(open('S3_Mixed_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S3_Mixed_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([mu]), np.array([Td]), np.array([S3])),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S3_Mixed_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the pulse-like ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S3_PulseLike_InputScaler.pkl
#   2. S3_PulseLike_OutputScaler.pkl
#   3. S3_PulseLike_ANN_Model.keras
# mu= friction coefficient of the bearings
# Td= pendulum period of the bearings
# S3 (measured in g)= spectral acceleration at 3-s period
# return expected peak displacement DM (in meter)
# usage example:
# DM = ANN_S3_PulseLikeGM(0.02, 2.0, S3)
def ANN_S3_PulseLikeGM(mu, Td, S3):
    # load scaler
    input_scaler= load(open('S3_PulseLike_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S3_PulseLike_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([mu]), np.array([Td]), np.array([S3])),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S3_PulseLike_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
#===============================================
# function to predict peak displacement using ANN model trained from the no-pulse ground motion group
# The below 3 files must be put in the same folder to this file:
#   1. S3_NoPulse_InputScaler.pkl
#   2. S3_NoPulse_OutputScaler.pkl
#   3. S3_NoPulse_ANN_Model.keras
# mu= friction coefficient of the bearings
# Td= pendulum period of the bearings
# S3 (measured in g)= spectral acceleration at 3-s period
# return expected peak displacement DM (in meter)
# usage example:
# DM = ANN_S3_NoPulseGM(0.02, 2.0, S3)
def ANN_S3_NoPulseGM(mu, Td, S3):
    # load scaler
    input_scaler= load(open('S3_NoPulse_InputScaler.pkl', 'rb'))
    output_scaler= load(open('S3_NoPulse_OutputScaler.pkl', 'rb'))
    # create input vector
    X= np.concatenate((np.array([mu]), np.array([Td]), np.array([S3])),axis=0)
    # scale inputs
    X_transf= input_scaler.transform([X])
    # load ANN model
    model= load_model('S3_NoPulse_ANN_Model.keras')
    # predict
    Y_transf= model.predict(X_transf, verbose=None)
    # transform the prediction
    Y= output_scaler.inverse_transform(Y_transf)
    return Y[0][0]
