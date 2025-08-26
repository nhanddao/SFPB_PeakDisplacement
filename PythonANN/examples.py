# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:29:23 2024
@author: nhand
"""
import numpy as np
import SFPB_PeakDisp

# Properties of the bearings in the isolation system:
mu= 0.04; # friction coefficient
Td= 3.5; # pendulum period
# 1s-spectral acceleration of the site
S1= 0.6 # measured in g. i.e, A1= 0.6g
# assume that the design spectrum follows a hyperbole curve for periods larger than 1.0s
# then spectral acceleration at 1.0 s, 2.0 s,...,5.0 s are:
S1To5= S1/np.arange(1.0,6.0,1.0)
S3= S1To5[2] # 3s-period spectral acceleration
print('============================================')
print('PEAK DISPLACEMENT PER DIFFERENT PREDICTORS\n(in meters)')
print('===========================================')

print('=============================')
print('ANN MODELS USING S1 TO S5:\n')
#=====================================
print('[+] Mixed ground motion:\n    ' + str(
      SFPB_PeakDisp.ANN_MixedGM(mu,Td, S1To5)
      )+ '\n')
#=====================================
print('[+] Pulse-like ground motion:\n    ' + str(
      SFPB_PeakDisp.ANN_PulseLikeGM(mu,Td, S1To5)
      )+ '\n')
#=====================================
print('[+] No-pulse ground motion:\n    ' + str(
      SFPB_PeakDisp.ANN_NoPulseGM(mu,Td, S1To5)
      )+ '\n')
print('=============================')
print('ANN MODELS USING S3:\n')
#=====================================
print('[+] Mixed ground motion:\n    ' + str(
      SFPB_PeakDisp.ANN_S3_MixedGM(mu,Td, S3)
      )+ '\n')
#=====================================
print('[+] Pulse-like ground motion:\n    ' + str(
      SFPB_PeakDisp.ANN_S3_PulseLikeGM(mu,Td, S3)
      )+ '\n')
#=====================================
print('[+] No-pulse ground motion:\n    ' + str(
      SFPB_PeakDisp.ANN_S3_NoPulseGM(mu,Td, S3)
      )+ '\n')