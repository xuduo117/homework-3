#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 22:45:27 2017

@author: xuduo
"""



import numpy as np
import matplotlib.pyplot as plt
import astropy
from scipy.integrate import simps
from scipy.interpolate import interp1d

###plank function
def brightness_temp(temp,v):
    b_t=2*h*v**3/c**2/(e**(h*v/k/temp)-1)#*1e23  ## in Jy
#    b_t=2*h*c**2/lambda_1**5/(e**(h*c/k/temp/lambda_1)-1)*1e23
    return b_t

### dust model
def dust_model_Qabs(dust_model_01,lambda_1):
    lambda_dust_01=dust_model_01[:,0]
    Qabs_01=dust_model_01[:,1]
    f2 = interp1d(lambda_dust_01,Qabs_01, kind='slinear',fill_value='extrapolate')   ### 'cubic'
    Qabs_01_inter=f2(lambda_1) 
    return Qabs_01_inter

def dust_power_out(Qabs_01_inter,temp_low,temp_high,v,R_dust_01,P_in):
    
#    P_out_1=np.pi*simps(Qabs_01_inter*brightness_temp(temp_low,v)*np.pi*R_dust_01**2,v,dx=1)
#    P_out_2=np.pi*simps(Qabs_01_inter*brightness_temp(temp_high,v)*np.pi*R_dust_01**2,v,dx=1)
    temp_middle=(temp_low+temp_high)/2.0
    P_out_middle=np.pi*simps(Qabs_01_inter*brightness_temp(temp_middle,v)*1e23*np.pi*4*R_dust_01**2,v,dx=1)
    if P_out_middle<P_in:
        return temp_middle,temp_high
    if P_out_middle>P_in:
        return temp_low,temp_middle


def dust_temp_fit(Qabs_01_inter,temp_low1,temp_high1,v,R_dust_01,P_in_01,delta_num,ctt_num):
    temp_low=temp_low1
    temp_high=temp_high1
    delta=temp_high-temp_low
    ctt=0
    while (delta>delta_num) and (ctt<ctt_num):
        temp_low_new,temp_high_new=dust_power_out(Qabs_01_inter,temp_low,temp_high,v,R_dust_01,P_in_01)
        temp_low=temp_low_new
        temp_high=temp_high_new
        delta=temp_high-temp_low
        ctt=ctt+1
#    return temp_low_new,temp_high_new
    return (temp_low+temp_high)/2.0

def print_force_10(P_in_01,M_dust_01):
    v_1=np.sqrt(G*M_star/(Distance_1*AU))
#    v_2=np.sqrt(G*M_star/(Distance_2*AU))
    f_grav=(G*M_star*M_dust_01/(Distance_1*AU)**2)
    f_rad=P_in_01/1e23/c
    print '%.4e' % (f_rad),'%.4e' % (f_grav),'%.4e' % (f_rad/f_grav),'%.4e' % (f_rad*v_1/c),'%.3e' % (400*(f_grav/f_rad)*(Distance_1)**2/M_star*M_sun)
     
def print_force_130(P_in_01,M_dust_01):
    v_1=np.sqrt(G*M_star/(Distance_2*AU))
    f_grav=(G*M_star*M_dust_01/(Distance_2*AU)**2)
    f_rad=P_in_01/1e23/c
    print '%.4e' % (f_rad),'%.4e' % (f_grav),'%.4e' % (f_rad/f_grav),'%.4e' % (f_rad*v_1/c),'%.3e' % (400*(f_grav/f_rad)*(Distance_2)**2/M_star*M_sun)
    

R_sun=astropy.constants.R_sun.cgs.to_value()
M_sun=astropy.constants.M_sun.cgs.to_value()
G=astropy.constants.G.cgs.to_value()
L_sun=astropy.constants.L_sun.cgs.to_value()
sigma_sb=astropy.constants.sigma_sb.cgs.to_value()
AU=astropy.constants.au.cgs.to_value()

M_star=1.92*M_sun
star_radius=R_sun*1.842
star_temp=8590#8590
Distance_1= 10    ## in  AU
Distance_2= 130 ##206265*7.61#7.61    ## in  AU



L_star=4*np.pi*star_radius**2*sigma_sb*star_temp**4


###constant
c=2.99792458*10**10
k=1.380650*10**(-16)
h=6.626069*10**(-27)
e=np.exp(1)

###variable 
v=np.logspace(11.4,15.8,num=10000)
#v_ghz=np.logspace(9,13,num=1000)/1e9
lambda_1=c/v*1e4


dust_model_01=np.loadtxt('../dust_model/0.1um.txt')
dust_model_1=np.loadtxt('../dust_model/1um.txt')
dust_model_10=np.loadtxt('../dust_model/10um.txt')

#lambda_dust_01=dust_model_01[:,0]
#Qabs_01=dust_model_01[:,1]
#f2 = interp1d(lambda_dust_01,Qabs_01, kind='slinear',fill_value='extrapolate')   ### 'cubic'
#Qabs_01_inter=f2(lambda_1) 

Qabs_01_inter=dust_model_Qabs(dust_model_01,lambda_1)
Qabs_1_inter=dust_model_Qabs(dust_model_1,lambda_1)
Qabs_10_inter=dust_model_Qabs(dust_model_10,lambda_1)
Qabs_1m=np.ones(len(v))
R_dust_01=0.1*1e-4
R_dust_1=1*1e-4
R_dust_10=10*1e-4
R_dust_1m=0.1

M_dust_01=4./3.*np.pi*R_dust_01**3*2
M_dust_1=4./3.*np.pi*R_dust_1**3*2
M_dust_10=4./3.*np.pi*R_dust_10**3*2
M_dust_1m=4./3.*np.pi*R_dust_1m**3*2


#b_t_10=brightness_temp(10.0,v)
b_t_star=brightness_temp(star_temp,v)
L_nu_star=np.pi*b_t_star*4*np.pi*star_radius**2
f_nu_star_1=L_nu_star/(4*np.pi*(Distance_1*AU)**2)*1e23
f_nu_star_2=L_nu_star/(4*np.pi*(Distance_2*AU)**2)*1e23

xlim=np.array([0.05,1000])
###plot jobs



P_in_01=simps(Qabs_01_inter*f_nu_star_1*np.pi*R_dust_01**2,v,dx=1)
P_in_1=simps(Qabs_1_inter*f_nu_star_1*np.pi*R_dust_1**2,v,dx=1)
P_in_10=simps(Qabs_10_inter*f_nu_star_1*np.pi*R_dust_10**2,v,dx=1)
P_in_1m=simps(Qabs_1m*f_nu_star_1*np.pi*R_dust_1m**2,v,dx=1)


P_in_01_2=simps(Qabs_01_inter*f_nu_star_2*np.pi*R_dust_01**2,v,dx=1)
P_in_1_2=simps(Qabs_1_inter*f_nu_star_2*np.pi*R_dust_1**2,v,dx=1)
P_in_10_2=simps(Qabs_10_inter*f_nu_star_2*np.pi*R_dust_10**2,v,dx=1)
P_in_1m_2=simps(Qabs_1m*f_nu_star_2*np.pi*R_dust_1m**2,v,dx=1)


T_01_1= dust_temp_fit(Qabs_01_inter,5,8000,v,R_dust_01,P_in_01,0.05,20)
T_1_1= dust_temp_fit(Qabs_1_inter,5,8000,v,R_dust_1,P_in_1,0.05,20)
T_10_1= dust_temp_fit(Qabs_10_inter,5,8000,v,R_dust_10,P_in_10,0.05,20)
T_1m_1= dust_temp_fit(Qabs_1m,5,8000,v,R_dust_1m,P_in_1m,0.05,20)

T_01_2= dust_temp_fit(Qabs_01_inter,5,8000,v,R_dust_01,P_in_01_2,0.05,20)
T_1_2= dust_temp_fit(Qabs_1_inter,5,8000,v,R_dust_1,P_in_1_2,0.05,20)
T_10_2= dust_temp_fit(Qabs_10_inter,5,8000,v,R_dust_10,P_in_10_2,0.05,20)
T_1m_2= dust_temp_fit(Qabs_1m,5,8000,v,R_dust_1m,P_in_1m_2,0.05,20)


print_force_10(P_in_01,M_dust_01)
print_force_130(P_in_01_2,M_dust_01)
print_force_10(P_in_1,M_dust_1)
print_force_130(P_in_1_2,M_dust_1)
print_force_10(P_in_10,M_dust_10)
print_force_130(P_in_10_2,M_dust_10)
print_force_10(P_in_1m,M_dust_1m)
print_force_130(P_in_1m_2,M_dust_1m)


