#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:35:08 2017

@author: xuduo
"""


import numpy as np
import matplotlib.pyplot as plt
import astropy
import scipy.integrate as integrate


###plank function
def brightness_temp(temp,v):
    b_t=2*h*v**3/c**2/(e**(h*v/k/temp)-1)#*1e23  ## in Jy
#    b_t=2*h*c**2/lambda_1**5/(e**(h*c/k/temp/lambda_1)-1)*1e23
    return b_t

R_sun=astropy.constants.R_sun.cgs.to_value()
L_sun=astropy.constants.L_sun.cgs.to_value()
sigma_sb=astropy.constants.sigma_sb.cgs.to_value()
AU=astropy.constants.au.cgs.to_value()

star_radius=1.842*R_sun
star_temp=8590
Distance_1= 10    ## in  AU
Distance_2= 130    ## in  AU



L_star=4*np.pi*star_radius**2*sigma_sb*star_temp**4


###constant
c=2.99792458*10**10
k=1.380650*10**(-16)
h=6.626069*10**(-27)
e=np.exp(1)

###variable 
v=np.logspace(11,15.8,num=1000)
#v_ghz=np.logspace(9,13,num=1000)/1e9
lambda_1=c/v*1e4



result = integrate.simps(brightness_temp(star_temp,v),v,dx=1)


#b_t_10=brightness_temp(10.0,v)
b_t_star=brightness_temp(star_temp,v)
L_nu_star=np.pi*b_t_star*4*np.pi*star_radius**2
f_nu_star_1=L_nu_star/(4*np.pi*(Distance_1*AU)**2)*1e23
f_nu_star_2=L_nu_star/(4*np.pi*(Distance_2*AU)**2)*1e23

xlim=np.array([0.05,1000])
###plot jobs
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(lambda_1,f_nu_star_1,label="Fomalhaut 10AU",color="red",linewidth=2)
ax1.plot(lambda_1,f_nu_star_2,label="Fomalhaut 130AU",color="green",linewidth=2)
#plt.plot(lambda_1,b_t_10,label="10K blackbody",color="green",linewidth=2)
ax1.set_xlim(xlim)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.yaxis.get_data_interval()


ax2 = ax1.twiny()
ax2.set_xscale('log')
ax2.set_xlabel(r"$\nu\, [Hz]$",fontsize=15)
ax2.set_xlim(c/xlim*1e4)
#ax2.set_xticks([1e11,1e10,1e9])
#ax2.set_xticklabels([r'$10^1$','8','99','1'])



#plt.xlim(9,4.5e3)
#plt.ylim(1e-19,1e-11)
#ax1.tick_params(labelsize=14)


ax1.set_xlabel(r"$\lambda\, [\mu m]$",fontsize=15)
ax1.set_ylabel(r'$\rm F_{\nu}\, [Jy]$',fontsize=15)
#ax1.set_title("Blackbody",fontsize=14)
ax1.legend(loc=0,fontsize=14)
plt.savefig('part_1.pdf',bbox_inches='tight')
plt.show()