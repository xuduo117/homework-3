#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 14:42:59 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
import astropy
import astropy.constants
from scipy.integrate import simps
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters
import lmfit


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

    
def dust_spec(dust_temp,Qabs_01_inter,v,R_dust_01):
    b_t_dust=brightness_temp(dust_temp,v)*1e23
    L_nu_dust=np.pi*b_t_dust*4*np.pi*R_dust_01**2
    f_nu_dust_1=L_nu_dust*Qabs_01_inter
    return f_nu_dust_1
    
def inter_model(dust_spec_1):
    f2 = interp1d(lambda_1,dust_spec_1, kind='slinear',fill_value='extrapolate')   ### 'cubic'
    spec_01_1_inter=f2(lambda_obs) 
    return spec_01_1_inter


def residual_all(params,data,dust_spec_1,dust_spec_2):
    spec_scale_1=params['p_0']
    spec_scale_2=params['p_1']
    model=spec_scale_1*(inter_model(dust_spec_1))+spec_scale_2*(inter_model(dust_spec_2))
#    model=spec_scale_1*np.log10(inter_model(dust_spec_01_1))+spec_scale_2*np.log10(inter_model(dust_spec_01_2))
#    model=
#    data=np.log10(data)
    return (np.log10(model)-np.log10(data))
#    return data-model

def fitting_para(p_0,p_0min,p_0max,p_1,p_1min,p_1max,dust_spec_1m_1,dust_spec_1m_2):
    
    params = Parameters()
        
    #params.add('p_0', value=1e23,min=1e15,max=1e46)
    #params.add('p_1', value=1e33,min=1e15,max=1e46)
    params.add('p_0', value=p_0,min=p_0min,max=p_0max)
    params.add('p_1', value=p_1,min=p_1min,max=p_1max)
    #residual_all(params,data,dust_spec_01_1,dust_spec_01_2)
    out3 = minimize(residual_all, params,method='leastsq', args=(flux_obs,dust_spec_1m_1,dust_spec_1m_2))
    model=out3.params['p_0'].value*dust_spec_1m_1+out3.params['p_1'].value*dust_spec_1m_2
    model=model*scale_factor
    return out3,model

def print_fitting(out_01,r):
    print out_01.params['p_0'].value*1e20,out_01.params['p_1'].value*1e20,\
    out_01.params['p_0'].value*1e20*4./3.*np.pi*r**3*2,out_01.params['p_1'].value*1e20*4./3.*np.pi*r**3*2,\
    out_01.chisqr


R_sun=astropy.constants.R_sun.cgs.to_value()
L_sun=astropy.constants.L_sun.cgs.to_value()
sigma_sb=astropy.constants.sigma_sb.cgs.to_value()
AU=astropy.constants.au.cgs.to_value()
pc=astropy.constants.pc.cgs.to_value()

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
#v=np.logspace(10,16,num=10000)
#v_ghz=np.logspace(9,13,num=1000)/1e9
lambda_1=c/v*1e4


scale_factor=1e20

data=np.loadtxt('/Users/xuduo/Desktop/sed_1.txt')
flux_obs=data[:,1]*1e-3*4*np.pi*(7.7*pc)**2 /scale_factor
lambda_obs=data[:,0]

#plt.plot(lambda_obs,flux_obs)
#plt.xscale('log')
#plt.yscale('log')
#plt.xlabel(r"$\lambda\, [\mu m]$",fontsize=15)
#plt.ylabel(r'$\rm F_{\nu}\, [Jy]$',fontsize=15)
#plt.savefig('part_4_obs_sed.pdf',bbox_inches='tight')
#
#plt.clf()



#"""
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


#b_t_10=brightness_temp(10.0,v)
b_t_star=brightness_temp(star_temp,v)
L_nu_star=np.pi*b_t_star*4*np.pi*star_radius**2
f_nu_star_1=L_nu_star/(4*np.pi*(Distance_1*AU)**2)*1e23
f_nu_star_2=L_nu_star/(4*np.pi*(Distance_2*AU)**2)*1e23

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

    
dust_spec_01_1=dust_spec(T_01_1,Qabs_01_inter,v,R_dust_01)
dust_spec_1_1=dust_spec(T_1_1,Qabs_1_inter,v,R_dust_1)
dust_spec_10_1=dust_spec(T_10_1,Qabs_10_inter,v,R_dust_10)
dust_spec_1m_1=dust_spec(T_1m_1,Qabs_1m,v,R_dust_1m)

dust_spec_01_2=dust_spec(T_01_2,Qabs_01_inter,v,R_dust_01)
dust_spec_1_2=dust_spec(T_1_2,Qabs_1_inter,v,R_dust_1)
dust_spec_10_2=dust_spec(T_10_2,Qabs_10_inter,v,R_dust_10)
dust_spec_1m_2=dust_spec(T_1m_2,Qabs_1m,v,R_dust_1m)



#def fitting_para(p_0,p_1,dust_spec_1m_1,dust_spec_1m_2):
#    
#params = Parameters()
#    
##params.add('p_0', value=1e23,min=1e15,max=1e46)
##params.add('p_1', value=1e33,min=1e15,max=1e46)
#params.add('p_0', value=p_0,min=1e-10,max=1.0e20)
#params.add('p_1', value=p_1,min=1e-10,max=1.0e20)
##residual_all(params,data,dust_spec_01_1,dust_spec_01_2)
#out3 = minimize(residual_all, params,method='leastsq', args=(flux_obs,dust_spec_1m_1,dust_spec_1m_2))
#return out3


out_01,model_01=fitting_para(1e15,1e7,1e18,5e16,1e11,1e20,dust_spec_01_1,dust_spec_01_2)
print lmfit.report_errors(out_01.params)

out_1,model_1=fitting_para(5e12,1e6,1e20,5e13,1e10,1e20,dust_spec_1_1,dust_spec_1_2)
print lmfit.report_errors(out_1.params)

out_10,model_10=fitting_para(1e15,1e7,1e18,5e13,1e8,1e18,dust_spec_10_1,dust_spec_10_2)
print lmfit.report_errors(out_10.params)

out_1m,model_1m=fitting_para(1e5,1e-2,1e15,1e8,1e-1,1e15,dust_spec_1m_1,dust_spec_1m_2)
print lmfit.report_errors(out_1m.params)

#plt.plot()

print_fitting(out_01,R_dust_01)
print_fitting(out_1,R_dust_1)
print_fitting(out_10,R_dust_10)
print_fitting(out_1m,R_dust_1m)

    
    

#"""
xlim=np.array([7,1000])
#ylim=np.array([1e-8,1e15])
#ylim=np.array([1e30,1e50])
ylim=np.array([1e37,10**41.5])
###plot jobs
fig = plt.figure(2)
fig.clf()
ax1 = fig.add_subplot(111)

#ax1.plot(lambda_1,dust_spec_01_1,label="dust 0.1um 10AU",color="red",linewidth=2)
#ax1.plot(lambda_1,dust_spec_1_1,label="dust 1um 10AU",color="green",linewidth=2)
#ax1.plot(lambda_1,dust_spec_10_1,label="dust 10um 10AU",color="blue",linewidth=2)
#ax1.plot(lambda_1,dust_spec_1m_1,label="dust 1mm 10AU",color="yellow",linewidth=2)
#
#ax1.plot(lambda_1,dust_spec_01_2,label="dust 0.1um 130AU",color="cyan",linewidth=2)
#ax1.plot(lambda_1,dust_spec_1_2,label="dust 1um 130AU",color="pink",linewidth=2)
#ax1.plot(lambda_1,dust_spec_10_2,label="dust 10um 130AU",color="brown",linewidth=2)
#ax1.plot(lambda_1,dust_spec_1m_2,label="dust 1mm 130AU",color="orange",linewidth=2)


ax1.plot(lambda_1,model_01,label="dust 0.1um",color="green",linewidth=2)
ax1.plot(lambda_1,model_1,label="dust 1um",color="blue",linewidth=2)
ax1.plot(lambda_1,model_10,label="dust 10um",color="orange",linewidth=2)
ax1.plot(lambda_1,model_1m,label="dust 1mm",color="red",linewidth=2)
ax1.plot(lambda_obs,flux_obs*scale_factor,label="observed",color="black",linewidth=2)



ax1.set_xlim(xlim)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.yaxis.get_data_interval()


ax2 = ax1.twiny()
ax2.set_xscale('log')
ax2.set_xlabel(r"$\nu\, [Hz]$",fontsize=15)
ax2.set_xlim(c/xlim*1e4)
ax2.set_ylim(ylim)

#ax2.set_xticks([1e11,1e10,1e9])
#ax2.set_xticklabels([r'$10^1$','8','99','1'])



#plt.xlim(9,4.5e3)
#plt.ylim(1e-19,1e-11)
#ax1.tick_params(labelsize=14)


ax1.set_xlabel(r"$\lambda\, [\mu m]$",fontsize=15)
ax1.set_ylabel(r'$\rm L_{\nu}\, [Jy\, cm^{2}]$',fontsize=15)
#ax1.set_title("Blackbody",fontsize=14)
ax1.legend(loc=0,fontsize=14)
#plt.savefig('part_3_dust_sed.pdf',bbox_inches='tight')
plt.savefig('part_4_fiting_01_sed.pdf',bbox_inches='tight')
plt.show()
#"""


