#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 08:11:06 2025

@author: albertsmith
"""


import sys
try:
    from ipywidgets import interactive_output, HBox, VBox
    import ipywidgets as widgets
except:
    widgets=None
    
from IPython.display import display
import matplotlib.pyplot as plt
from . import Defaults
import functools

def set_dark(dark=True):
    if dark:
        x=56 if Defaults['Colab'] else 17
        plt.rcParams["figure.facecolor"]=(x/256,x/256,x/256)
        plt.rcParams["axes.facecolor"]=(x/256,x/256,x/256)
        plt.rcParams["axes.edgecolor"]=(1,1,1)
        plt.rcParams["axes.labelcolor"]=(1,1,1)
        plt.rcParams["xtick.color"]=(1,1,1)
        plt.rcParams["ytick.color"]=(1,1,1)
        plt.rcParams["text.color"]=(1,1,1)
    else:
        plt.style.use('default')


def zoom(ax):
    if Defaults['Colab']:
        return ColabZoom(ax)
    plt.close(ax.figure)
    
    hdisplay=display(ax.figure,display_id=True)
    
    threeD=ax.name=='3d'
    
    matrix='AxesImage' in [child.__class__.__name__ for child in ax.get_children()]
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    xc0=sum(xlim)/2
    yc0=sum(ylim)/2
    xr0=abs(xlim[1]-xlim[0])
    yr0=abs(ylim[1]-ylim[0])
    xd=1 if xlim[1]>xlim[0] else -1
    yd=1 if ylim[1]>ylim[0] else -1
    zc0=0
    zr0=10
    if threeD:
        zlim=ax.get_zlim()
        zc0=sum(zlim)/2
        zr0=zlim[1]-zlim[0]
        zd=1 if zlim[1]>zlim[0] else -1
    
    
    
    def update(xc,xz,yc,yz,zc=0,zz=0):
        xz=10**xz
        yz=10**yz
        zz=10**zz
        if matrix:yz=xz
        xr=xr0/xz
        yr=yr0/yz
        zr=zr0/zz
        ax.set_xlim((xc-xr/2,xc+xr/2)[::xd])
        ax.set_ylim((yc-yr/2,yc+yr/2)[::yd])
        if threeD:ax.set_zlim((zc-zr/2,zc+zr/2)[::zd])
        hdisplay.update(ax.figure)
        
    x_center = widgets.FloatSlider(min=min(xlim), max=max(xlim), step=xr0/100, value=xc0, description='x center')
    y_center = widgets.FloatSlider(min=min(ylim), max=max(ylim), step=yr0/100, value=yc0, description='y center')
    x_zoom = widgets.FloatSlider(min=0,max=2,step=.02,value=0,description='xy zoom' if matrix else 'x zoom')
    y_zoom = widgets.FloatSlider(min=0,max=2,step=.02,value=0,description='y zoom')
    
    if threeD:
        z_center = widgets.FloatSlider(min=zlim[0], max=zlim[1], step=zr0/100, value=zc0, description='z center')
        z_zoom = widgets.FloatSlider(min=0,max=2,step=.02,value=0,description='z zoom')
        
        center_box=VBox([x_center,y_center,z_center])
        zoom_box=VBox([x_zoom,y_zoom,z_zoom])
        
        interactive_output(update,{
             'xc':x_center,
             'xz':x_zoom,
             'yc':y_center,
             'yz':y_zoom,
             'zc':z_center,
             'zz':z_zoom})
    else:
        center_box=VBox([x_center,y_center])
        zoom_box=VBox([x_zoom,x_zoom if matrix else y_zoom])
        interactive_output(update,{
             'xc':x_center,
             'xz':x_zoom,
             'yc':y_center,
             'yz':y_zoom})
        
    s_box=HBox([zoom_box,center_box])
    
    display(s_box)
    
def ColabZoom(ax):
    plt.close(ax.figure)
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    xc0=sum(xlim)/2
    yc0=sum(ylim)/2
    xr0=abs(xlim[1]-xlim[0])
    yr0=abs(ylim[1]-ylim[0])
    
    def update(xc,xz,yc,yz,zc=0,zz=0):
        xz=10**xz
        yz=10**yz

        xr=xr0/xz
        yr=yr0/yz
        xd=1 if xlim[1]>xlim[0] else -1
        yd=1 if ylim[1]>ylim[0] else -1
        
        
        fig,ax0=plt.subplots(figsize=ax.figure.get_size_inches())
        
        for line in ax.get_lines():
            ax0.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), 
                     linestyle=line.get_linestyle(), color=line.get_color())

        
        ax0.set_xlim((xc-xr/2,xc+xr/2)[::xd])
        ax0.set_ylim((yc-yr/2,yc+yr/2)[::yd])
        ax0.set_xlabel(ax.get_xlabel())
        ax0.set_ylabel(ax.get_ylabel())
        if ax.get_legend() is not None:
            ax0.legend(*[[t.get_text() for t in ax.get_legend().get_texts()]])
        plt.show()
        
    x_center = widgets.FloatSlider(min=min(xlim), max=max(xlim), step=xr0/100, value=xc0, description='x center')
    y_center = widgets.FloatSlider(min=min(ylim), max=max(ylim), step=yr0/100, value=yc0, description='y center')
    x_zoom = widgets.FloatSlider(min=0,max=2,step=.02,value=0,description='x zoom')
    y_zoom = widgets.FloatSlider(min=0,max=2,step=.02,value=0,description='y zoom')
    
    center_box=VBox([x_center,y_center])
    zoom_box=VBox([x_zoom,y_zoom])
    q=interactive_output(update,{
         'xc':x_center,
         'xz':x_zoom,
         'yc':y_center,
         'yz':y_zoom})
    
    s_box=HBox([zoom_box,center_box])
    
    display(q,s_box)

    
    

def use_zoom(plot):
    @functools.wraps(plot)  #Preserve the doc string
    
    def setup(*args,**kwargs):
        ax=plot(*args,**kwargs)
        
        
        # Instances where we don't use zoom
        if not(Defaults['zoom']):  #Turned off
            return ax
        if widgets is None:
            return ax
        if 'ax' in kwargs and kwargs['ax'] is not None:  #Axis provided
            return ax
        for arg in args:
            if arg.__class__.__name__=='AxesSubplot':    #Axis provided
                return ax
        

        if Defaults['Colab']:
            if ax.name=='3d':return ax
            if 'AxesImage' in [child.__class__.__name__ for child in ax.get_children()]:return ax
            ColabZoom(ax)
        else:
            zoom(ax)
        
        return ax
    return setup
    
