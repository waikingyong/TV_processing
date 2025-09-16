# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:41:18 2023

@author: Riley.Elstermann
@co-author: Benjamin.Boland

Script to recreate wellcad wsg vbs scripts in python. 

Help with wellcad python API found here: 
    https://pywellcad.readthedocs.io/

Problems or suggestions to improve pywellcad, submit an issue in the GitHub:
    https://github.com/alt-lu/pywellcad    
    
PLEASE ENSURE YOU CHANGE THE DIRECTORY TO THE RIGHT FOLDER 
"""

import wellcad.com
import re
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
colorama_init()


""" ENSURE DIRECTORY BELOW IS CORRECT BEFORE RUNNING 
 Can be set to whatever file you want to use for processing the scripts without it being in same folder 
 Ensure there is a final '/' at the end and that all '\' are replaced by '/' """

tvdir = 'C:/Proc_TV/V2/'

###############################################################################
###############################################################################

#------------------------------------------------------------------------------
def boreholeextract(borehole):
    '''
    Function to obtain a pandas dataframe df of all the logs within a WCL file.
    
    Inputs: 
        borehole - pywellcad borehole object from relevant WCL filepath
        
    Outputs: 
        lognames   - list of acceptable log names output from WCL file 
        logdict    - dictionary of acceptable log titles as keys and 
                     associated dataframes
    '''
    
    # Obtaining number of logs within file
    nlogs = borehole.nb_of_logs

    # Setting parameters for loop
    logdict = {}
    logs = [borehole.get_log(i) for i in range(nlogs)]
    lognames = [log.name for log in logs]

    # print('lognames a: ', lognames)
    
    # Define list of logs to process using pandas dataframes, add more as needed
    processlogs = ['AZI', 'AZIMUTH', 'TILT', 'NG', 
                   'SANG', 'SANGB', 'L_GAMMAR', 'L_GAMMA',
                   'CAL', 'MAG', 'L_SANGB', 'L_SANG', 
                   'L_NSGSANGB', 'L_NSGSANG', 'L_GSANGB', 'L_GSANG',
                   'DEN(SS)', 'DEN(SS)#1', 'DEN(LS)', 'DEN(LS)#1', 'DEN(CDL)', 'COMP']
    
    # Remove log names in lognames that are not in processlogs
    lognames = [logname for logname in lognames if logname in processlogs]

    # Cycle through acceptable log names and extract depth data and log data
    for logname in lognames:
        log = borehole.get_log(logname)
        title = log.name
        try:
            data = np.array(log.data_table[1:])
            tmpdf =  pd.DataFrame(data, columns=['Depth (m)', title]).round(2)
            depthfilt = ((tmpdf['Depth (m)']>=0))
            tmpdf.loc[depthfilt, 'Depth (m)'] = tmpdf.loc[depthfilt, 'Depth (m)'].round(1)
            logdict[logname] = tmpdf
        except Exception:
            continue
            
    return lognames, logdict


#------------------------------------------------------------------------------
def logtotuple(logdf):
    '''
    Function to place our data back into WellCAD data format of tuple of tuples.
    
    Inputs: 
        logdf       - pandas dataframe with one log and one data column e.g. DEPTH, AZIMUTH
        
    Outputs: 
        datatuples  - WELLCAD log tuple of tuples
    '''

    cols = [logdf.columns.tolist()]
    data = [tuple(i) for i in logdf.values]
    logtuple = cols + data
    logtuple = tuple([tuple(i) for i in logtuple])

    return logtuple


#------------------------------------------------------------------------------
def comment_match(comment):
    '''
    Function to locate in the contractor comment section whether a statement has
    been made about the hole being blocked to explain why the logs stop higher up.
    
    Inputs: 
        comment - Contractor comment taken from headers
        
    Outputs: 
        depth   - numerical value of blocked depth
    '''  
    
    # Reading comment to check for "Blocked at  " statements
    match = re.search(r'\bblocked(?: at| @)? (\d+(\.|d+)?)(?:m\b)?', comment, re.IGNORECASE)
    if match:
        # Extract the number part 
        depth = match.group(1)
        if '.' in depth:
            depth = float(depth) + 7
        else:
            depth = int(depth) + 7
    else:
        depth = 0
    
    return depth


# #------------------------------------------------------------------------------        
def holedepthchecker(holeid, header, rgblog, amplog):
    '''
    Function to compare the depth of the hole against the logged depth by 
    contractor for discrepancies. Now prints verbose explanations if problems are detected.
    '''
    # Section to check for WSG files
    try:
        drillers_depth = float(header.get_item_text('DRDP') or 0)
        loggers_depth = float(header.get_item_text('LOTD') or 0) + 5
        log_bottom = float(header.get_item_text('LB') or 0) + 5
        wsg_comment = header.get_item_text('Comments') or ""
        depth = comment_match(wsg_comment)

        # Convert depth to string, show "NONE" if depth == 0, else show as "xx.xx m"
        depth_str = "NONE" if depth == 0 else f"{depth:.2f} m"

        check = False
        if loggers_depth < drillers_depth:
            if depth < loggers_depth:
                print(
                    f"{holeid} {Fore.YELLOW}{Style.BRIGHT}may have a logging depth problem to be checked.{Style.RESET_ALL}\n"
                    f"  Driller's depth (DRDP): {drillers_depth:.2f} m\n"
                    f"  Logger's depth (LOTD + 5): {loggers_depth:.2f} m\n"
                    f"  Blocked/commented depth (from Comments): {depth_str}\n"
                    f"  Contractor comment: {wsg_comment}\n"
                    "  (WARNING : Logger's depth is less than driller's depth, and no 'blocked' comment explains it.)"
                )
                check = True
        if log_bottom < (loggers_depth - 3) and check:
            if depth < loggers_depth:
                print(
                    f"{holeid} {Fore.YELLOW}{Style.BRIGHT}may have a logging depth problem to be checked.{Style.RESET_ALL}\n"
                    f"  Log bottom (LB + 5): {log_bottom:.2f} m\n"
                    f"  Logger's depth (LOTD + 5): {loggers_depth:.2f} m\n"
                    f"  Blocked/commented depth (from Comments): {depth_str}\n"
                    f"  Contractor comment: {wsg_comment}\n"
                    "  (WARNING : Log bottom is much shallower than logger's depth, and no 'blocked' comment explains it.)"
                )

    except Exception as e:
        print(f"Warning: Exception in WSG depth check for {holeid}: {e}")
        pass

    # Section to check for Epiroc files
    try:
        epiroc_drillers_depth_raw = header.get_item_text('DEPTHDRILLER1')
        # Only try to parse if value is present
        if epiroc_drillers_depth_raw:
            import re
            m = re.search(r"(\d+(\.\d+)?)", epiroc_drillers_depth_raw)
            epiroc_drillers_depth = float(m.group(1)) if m else 0
            otvbottom = round(rgblog.bottom_depth, 1) if rgblog is not None else None
            atvbottom = round(amplog.bottom_depth, 1) if amplog is not None else None
            epiroc_comment = str(header.get_item_text('COMMENT1')) or ""
            depth = comment_match(epiroc_comment)

            # Convert depth to string, show "NONE" if depth == 0, else show as "xx.xx m"
            depth_str = "NONE" if depth == 0 else f"{depth:.2f} m"

            if atvbottom is not None and otvbottom is not None:
                epiroc_log_bottom = max(atvbottom, otvbottom) + 5
            elif atvbottom is not None:
                epiroc_log_bottom = atvbottom + 5
            elif otvbottom is not None:
                epiroc_log_bottom = otvbottom + 5
            else:
                epiroc_log_bottom = None

            if epiroc_log_bottom is not None and epiroc_log_bottom < epiroc_drillers_depth:
                if depth < epiroc_log_bottom:
                    print(
                        f"{holeid} {Fore.YELLOW}{Style.BRIGHT}may have a logging depth problem to be checked.{Style.RESET_ALL}\n"
                        f"  Epiroc driller's depth (DEPTHDRILLER1): {epiroc_drillers_depth:.2f} m\n"
                        f"  Log bottom (max image log + 5): {epiroc_log_bottom:.2f} m\n"
                        f"  Blocked/commented depth (from COMMENT1): {depth_str}\n"
                        f"  Contractor comment: {epiroc_comment}\n"
                        "  (WARNING : bottom is less than driller's depth, and no 'blocked' comment explains it.)"
                    )
    except Exception as e:
        print(f"Warning: Exception in Epiroc depth check for {holeid}: {e}")
        pass

    return


#------------------------------------------------------------------------------
def log_obtain(holeid, borehole, logdict, log1): 
    '''
    Obtain the specified log from a borehole, its table, DataFrame, top depth, and a missing-log condition flag.
    Prints a warning if the CAL log is missing.
    '''
    # Set up return defaults
    missing_log = False
    log = None
    table = []
    df = pd.DataFrame()
    top = None

    # Attempt to get the log
    try:
        log = borehole.get_log(log1)
    except Exception:
        log = None

    # If log is missing, warn and handle CAL log
    if log is None:
        print(holeid + f"{Fore.YELLOW}{Style.BRIGHT} is missing {log1} log.{Style.RESET_ALL}")
        if log1.strip().upper() == "CAL":
            print(f"{holeid} {Fore.YELLOW}{Style.BRIGHT}WARNING: CAL (caliper) log is missing. Please check WellCAD file.{Style.RESET_ALL}")
        missing_log = True
        return log, table, df, top, missing_log

    # Get top depth based on log type
    try:
        if log1 in ['RGB', 'AMP']:
            top = math.floor(log.top_depth * 10) / 10
        elif log1 in ['L_SANGB', 'L_SANG', 'L_NSGSANGB', 'L_NSGSANG', 'AZIMUTH', 'TILT']:
            top = round(log.top_depth, 1)
        elif len(str(log.top_depth)) > 3 and int(str(log.top_depth)[3]) >= 8:
            top = round(log.top_depth, 1)
        else:
            top = math.floor(log.top_depth * 10) / 10
    except Exception:
        top = None

    # Get table and DataFrame for logs except RGB and AMP
    if log1 not in ['RGB', 'AMP']:
        try:
            table = list(log.data_table)
        except Exception:
            table = []
        try:
            df = logdict[log1]
        except Exception:
            df = pd.DataFrame()
    else:
        table = []
        df = pd.DataFrame()

    return log, table, df, top, missing_log
    
#------------------------------------------------------------------------------
def log_editor(table, log, holeid, viscond, top, bottom, header, logtype): 
    '''
    Function to edit AZIMUTH & TILT logs in WellCAD.
    
    Inputs: 
        table - list version of data being processed
        log - pywellcad log version of data being processed
        holeid - name of hole currently being processed
        viscond - visual display condition pass check
        top - top depth of borehole log
        bottom - bottom depth of borehole log
        header - needed for pywellcad coding in headers
        logtype - input log being processed
        
    Outputs: 
        table - list version of data
        log - pywellcad log version of data
        com1a, com1b, com1c - checks for adjusting comments later
        addition - additional comment statement; may be needed later for adjusting comments
        first_valid_reading_index, last_valid_reading_index - indexes used for comments
    ''' 
    
    if len(table) == 1:
        print(holeid + f"{Fore.RED}{Style.BRIGHT} {logtype} log has no data.{Style.RESET_ALL}")
        pass
    else:
        if logtype == 'Azimuth':
            log.name = 'AZIMUTH'
            log.pen_color = 16711680    # digits for colour blue
            log.scale_high = 360
        elif logtype == 'Tilt':
            log.name = 'TILT'
            log.pen_color = 255         # digits for colour red
            log.scale_high = 40
        log.scale_low = 0    
        
        # Rounding values (1dp for depth and 2 for log) 
        table = [((i[0],i[1]) if j == 0 else (round(i[0],1),round(i[1],2))) for j,i in enumerate(table)]
        table[0] = ('Depth [m]', logtype.upper())
        
        # Setting position of header
        # Horizontal check
        if log.left_position > 0.123 or log.right_position > 0.26:
            log.left_position = 0.1229
            log.right_position = 0.259
        
        # Awaiting vertical functionality to implement to ensure logs in correct order in column
        #
        #
        #
        
        # Setting correct top and bottom depths
        if viscond:
            logtop = top
        else:
            logtop = round(log.top_depth,1)
        logbottom = round(log.bottom_depth,1)
        

        # Extending azi log to ends of hole with null values
        if logbottom != bottom:
            while round(logbottom,1) < bottom:
                logbottom = logbottom + 0.1
                value = (round(logbottom,1), -999.25)
                table.append(value)
                
        if logtop != top:
            while round(logtop,1) > top:
                logtop = round((logtop - 0.1),1)
                value = (logtop, -999.25)
                table.insert(1,value)
        
        # Setting parameters
        first_valid_reading_index = None
        last_valid_reading_index = None
        
        # Extrapolating AZI log
        #
        # Obtaining index values for first and last value
        for i, (depth, reading) in enumerate(table[1:]):                        
            if reading > 0:
                if first_valid_reading_index is None:
                    first_valid_reading_index = i
                last_valid_reading_index = i
        
        # Check values
        check1 = False
        check2 = False
        
        # Adjusting AZI values
        for i, (depth, reading) in enumerate(table[1:]):
                           
            # Replacing null readings    
            if reading < 0:
                # Replaces top null values with the top most valid reading
                if i < first_valid_reading_index:
                    table[i+1] = (depth, table[1:][first_valid_reading_index][1])                
                    check1 = True
                    
                # Replaces bottom null values with the bottom most valid reading
                elif i > last_valid_reading_index:
                    table[i+1] = (depth, table[1:][last_valid_reading_index][1])
                    check2 = True
                    
                # Notifies if gaps are missing in data - Should never pass
                elif i > first_valid_reading_index and i < last_valid_reading_index:
                    table[i+1] = (depth, table[i][1])
            
    
        # Adding in comments
        com1a = False
        com1b = False
        com1c = False
        if round(table[1:][first_valid_reading_index][0],1) > 0:
            if check1 and check2:
                comment = header.get_item_text('Processor Comments:')
                addition = " " + logtype + " extrapolated above " + str(round(table[1:][first_valid_reading_index][0],1)) + " m and below " + str(round(table[1:][last_valid_reading_index][0],1)) + " m."
                header.set_item_text('Processor Comments:', comment + addition)
                com1a = True

            elif check1 and not check2:
                comment = header.get_item_text('Processor Comments:')
                addition = " " + logtype + " extrapolated above " + str(round(table[1:][first_valid_reading_index][0],1)) + " m."
                header.set_item_text('Processor Comments:', comment + addition)
                com1b = True
                
            else:
                addition = ''
                
        elif check2 and not check1:
            comment = header.get_item_text('Processor Comments:')
            addition = " " + logtype + " extrapolated below " + str(round(table[1:][last_valid_reading_index][0],1)) + " m."
            header.set_item_text('Processor Comments:', comment + addition) 
            com1c = True

        else:
            addition = ""
        # print(addition)
        # Setting the table in WellCAD to the manipulated list
        log.data_table = tuple(table)
        
    return table, log, com1a, com1b, com1c, addition, first_valid_reading_index, last_valid_reading_index


#------------------------------------------------------------------------------
def weighted_median(df, val, weight):
    '''
    Function to calculate the weighted median of the caliper log.
    
    Inputs: 
        df - caliper dataframe
        val - Caliper values
        weight - Weight values
        
    Output: 
        The weighted median value
    ''' 
    df_sorted = df.sort_values(val)
    cumsum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum() / 2.
    return df_sorted[cumsum >= cutoff][val].iloc[0]


#------------------------------------------------------------------------------
def reject_outliers(data,median, m=2.):
    '''
    Function to remove all outliers from dataset.
    
    Inputs: 
        data - caliper values
        median - median value of caliper values
        
    Output: 
        The weighted median value
    ''' 
    d = np.abs(data - median)
    mdev = np.median(d)
    s = 2*d / (mdev if mdev else 1.)
    return data[s < m]


#------------------------------------------------------------------------------
def cal_editor(holeid, calcond, caltable, callog, caldf, caltop, top, bottom, header):
    '''
    Function to edit only CAL log in WellCAD.
    
    Inputs: 
        holeid - name of hole currently being processed
        calcond - condition to skip fn if True
        caltable - list version of CAL data
        callog - pywellcad log version of cal data
        caldf - df version of cal data
        caltop - top depth of borehole log
        top & bottom - top & bottom depth of borehole log
        header - needed for pywellcad coding in headers
        
    Outputs: 
        callog - updated version of callog from this function
    ''' 
    if calcond == True:
        # If cal log is missing, skips this section
        return
    elif len(caltable) == 1:
        print(holeid + f"{Fore.RED}{Style.BRIGHT} CAL log has no data.{Style.RESET_ALL}")
        return
    else:
        callog.scale_low = 0
        callog.scale_high = 200
        
        # Rounding values (1dp for depth and 2 for Cal) 
        caltable = [((i[0],i[1]) if j == 0 else (round(i[0],1),round(i[1],2))) for j,i in enumerate(caltable)]
        
        # Obtaining bottom depth for CAL
        calbottom = round(callog.bottom_depth,1)
          
        # Removes any null values
        caldf = caldf[caldf['CAL'] >= 0]    
        # Total depth of hole
        valid_depth = caldf['Depth (m)'].iloc[-1]-caldf['Depth (m)'].iloc[0]            
        # Statement to obtain how much of the end of hole to use for extrapolation
        if valid_depth < 100:
            y = 10.3456 - 10.1247*math.pow(valid_depth,0.00361592)
            highest_reading = int(round(valid_depth*y,1)*10+1)
            caldf = caldf[-highest_reading:-1]
        else:
            caldf = caldf[-51:-1]
        # An Exponential weighting distribution
        exp_weight = np.exp(np.linspace(-0.3, 3.2, len(caldf['CAL'])))
        caldf['WEIGHT'] = exp_weight
        # Weighted median of CAL data
        w_median = weighted_median(caldf,'CAL','WEIGHT')
        
        # Removing outliers affecting data
        caldf_not_outliers = reject_outliers(caldf['CAL'],w_median)
        caldf = caldf[caldf.index.isin(caldf_not_outliers.index)]
        
        # An Exponential weighting distribution
        exp_weight2 = np.exp(np.linspace(-0.3, 3.2, len(caldf['CAL'])))
        caldf['WEIGHT'] = exp_weight2
        
        # Obtaining the weighted median with outliers removed
        w_o_median = weighted_median(caldf,'CAL','WEIGHT')
        
        # Adding null values at top of hole to match top depth
        if caltop != top:
            while round(caltop,1) > top:
                caltop = caltop - 0.1
                value = (round(caltop,1), -999.25)
                caltable.insert(1,value)
                
                                
        # Setting parameters
        first_valid_reading_index3 = None
        last_valid_reading_index3 = None
        
        # Obtaining index values for first and last value
        for i, (depth, reading) in enumerate(caltable[1:]):                        
           if reading > 0:
               if first_valid_reading_index3 is None:
                   first_valid_reading_index3 = i
               last_valid_reading_index3 = i
        
        
        check5 = False
        check6 = False
        # Extending caliper log to ends of hole with null values
        if calbottom != bottom:                        
            # Loop to add on the remaining null values 
            while round(calbottom,1) < bottom:
                calbottom = calbottom + 0.1
                value = (round(calbottom,1), w_o_median)
                caltable.append(value)
                check6 = True
               
        # Extrapolating CAL log            
        #        
        # Adjusting CAL values
        for i, (depth, reading) in enumerate(caltable[1:]):
                         
           # Replacing null readings    
           if reading < 0:
               # Replaces top null values with the top most valid reading
               if i < first_valid_reading_index3:
                   caltable[i+1] = (depth, caltable[1:][first_valid_reading_index3][1])
                   check5 = True
                   
               # Replaces bottom null values with the bottom most valid reading
               elif i > last_valid_reading_index3 - 1:
                   caltable[i+1] = (depth, w_o_median)
                   check6 = True 
               
               # Notifies if gaps are missing in data
               elif i > first_valid_reading_index3 and i < last_valid_reading_index3:
                   caltable[i+1] = (depth, caltable[i][1])             
    
        # Replacing last 4 valid reading from bottom up till fail with the w_o_m if values beyond last reading have been changed and only if they sit more than 4 out of the w_o_m      
        for i in [-1,0,1,2,3]:
            if caltable[last_valid_reading_index3-i][1] > w_o_median+4 or caltable[last_valid_reading_index3-i][1] < w_o_median-4:
                caltable[last_valid_reading_index3-i] = (caltable[last_valid_reading_index3-i][0], w_o_median)
            else:
                if i == -1:
                    break
                else:
                    i = i - 1
                    break


        # Adding in comments
        if round(caltable[1:][first_valid_reading_index3][0],1) > 0:
            if check5 and check6:
                comment = header.get_item_text('Processor Comments:')
                cal = " Caliper extrapolated above " + str(round(caltable[1:][first_valid_reading_index3][0],1)) + " m and below " + str(round(caltable[1:][last_valid_reading_index3][0]-i*0.1-0.1,1)) + " m."
                header.set_item_text('Processor Comments:', comment + cal)
            elif check5 and not check6:
                comment = header.get_item_text('Processor Comments:')
                cal = " Caliper extrapolated above " + str(round(caltable[1:][first_valid_reading_index3][0],1)) + " m."
                header.set_item_text('Processor Comments:', comment + cal)
        elif check6 and not check5:
            comment = header.get_item_text('Processor Comments:')
            cal = " Caliper extrapolated below " + str(round(caltable[1:][last_valid_reading_index3][0]-i*0.1-0.1,1)) + " m."
            header.set_item_text('Processor Comments:', comment + cal)
    
                          
        # Setting the table in WellCAD to the manipulated list
        callog.data_table = tuple(caltable)
    
    return callog


#------------------------------------------------------------------------------
def pinorm(azi): 
    '''
    Function takes input azi array in degrees between 0 to 360 
    OR -180 to 180 and returns the normalisation factor and normalised 
    array between 0 and 180. For use with inverse sin and cos functions.
    '''
    # First ensure all azimuth values are positive
    offset = np.abs(np.nanmin(azi))
    azi = azi + offset
    
    # Normalise all positive azimuths between 0 and 1
    norm = np.nanmax(azi)
    normazi = azi / norm
    
    # Multiply by pi
    normazi = normazi * np.pi/2
    
    return offset, norm, normazi


#------------------------------------------------------------------------------
def azifilter(holeid, header, azidf, tiltdf, azitop, azibottom, top, bottom, min_samples, comment, alpha=0.1):
    '''
    function to take a relevant pandas df of depth and azimuth and return a 
    smoothed azimuth, should work on magnetic compass, gyro and nsgyro data.
    This function will only work if depth column is 0th column and azi column 
    is 1st column in dataframe. 
    
    Inputs: 
        azidf - pandas dataframe with depth and azimuth
        
    Outputs: 
        azidf - same as input dataframe but with filtered azimuth column
        no_rotation - false unless tilt is low enough to be a '_tn' file or 
                      if no clusters for filter can be found
        
    '''    
    # Extract depth and azimuth data from azidf
    depth = azidf.iloc[:,0].values
    azi = azidf.iloc[:,1].values      
    
    # Convert azimuth from 0-360 to 0-pi
    offset, norm, normazi = pinorm(azi)
    
    # Defining trigonometric functions for use later
    trig = np.cos
    trig2 = np.sin
    invtrig = np.arcsin
    
    ######################################################################################################################################################
    ######################################################################################################################################################
    '''      
    Main filter used for hs_m files using the DBSCAN (Density-Based Spatial
    Clustering of Applications with Noise) to identify clusters in the azimuth 
    log and to remove the poor readings caused by magnetism.
    
    Variables can be changed by user to include more or less as requried.
    
    Through testing, the following shows best results for most cases.
    Default: eps=3, min_samples=25
    
    eps: "The maximum distance between two points for them to be considered 
          neighbors. Points that are within eps distance of each other are 
          considered part of the same cluster."
    
    min_samples: "The minimum number of points required for a point to be 
                  considered a core point. Points that have fewer than 
                  min_samples neighbors are labeled as noise."
    
    
    Generally, you only want to try to increase or decrease the no. of
    min_samples depending on whether you want to tighten how many are included
    or to include more points respectively.
    (i.e. higher number ==> less data included in cluster)
    
    '''
    
    Azi_stack = np.column_stack((depth, azi))
    db = DBSCAN(eps=3, min_samples=min_samples).fit(Azi_stack)
        
    ######################################################################################################################################################
    ######################################################################################################################################################
         
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)    

    
    # Unhashtag the following code if you wanted to visualise the raw azi in 
    # python if no clusters were found and needed to do a manual check
    # plt.plot(depth, azi, label='raw azi',color='royalblue')
    # plt.title(holeid)
    # plt.legend()
    # plt.show()
    

    # Check to see if mean for tilt is low enough that it shouldn't be a hs_m file
    no_rotation = False
    if n_clusters_ == 0:
        testdf = tiltdf[tiltdf['L_SANG'] >= 0]
        if testdf["L_SANG"].mean() <= 5:
            print(holeid, f"{Fore.YELLOW}{Style.BRIGHT}is now treated as a tn_m file due to tilt reading.{Style.RESET_ALL}")
            comment = header.get_item_text('Processor Comments:')
            file_type_change = " File considered as a tn_m file due to low tilt readings."
            header.set_item_text('Processor Comments:', comment + file_type_change) 
            
        # Hopefully a rare case where tilt is high enough to remain as a hs_m file
        else:
            print(holeid, f"{Fore.RED}{Style.BRIGHT}requires a manual check for possible rotation.{Style.RESET_ALL}")
        no_rotation = True
        return azidf, no_rotation

        
    # Visually displaying the clusters
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    azidf8 = pd.DataFrame(columns=['Depth (m)','AZIMUTH'])
    for k, col in zip(unique_labels, colors):    
        class_member_mask = labels == k

        xy = Azi_stack[class_member_mask & core_samples_mask]

        azidf7 = pd.DataFrame(xy,columns=['Depth (m)','AZIMUTH'])
        azidf8 = pd.concat([azidf8, azidf7], ignore_index = False)
        
        # Unhastag the following for a graph output in python if wanted
    #    if k == -1:
    #        # Black used for noise.
    #        col = [0, 0, 0, 1]
    #
    #    # Clusters
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=6,
    #     )
       
    #     # Noise
    #     xy = Azi_stack[class_member_mask & ~core_samples_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=2,
    #     )
            
    # plt.title(holeid + f" Estimated number of clusters: {n_clusters_}")
    # plt.show()
    
    
    ###########################################################################    
    # CHECK FOR DEPTH GAPS BETWEEN THE CLUSTERS TO SEE WHETHER OR NOT TO ADD INTO PROCESSOR COMMENTS

    # Tuple form of log
    depth_checker = logtotuple(azidf8)

    # Adding in bottom depth of azi
    depth_checker = list(depth_checker)
    depth_checker += [(azibottom,0)]
    if depth_checker[0][0] != azitop:
        depth_checker.insert(1,(azitop,0))
    depth_checker = tuple(depth_checker)

    # Obtaining gaps of greater of 2m and adding into a list
    gaps = []
    for i in range(1, len(depth_checker) - 1):
        current_depth = round(depth_checker[i][0],1)
        next_depth = round(depth_checker[i+1][0],1)
        
        if next_depth - current_depth > 2:
            gaps += [[current_depth,next_depth]]

    # Adding interpolated depth gaps into Processor Comments
    if len(gaps) > 0:
        interpolated_depths1 = " Azimuth interpolated between"
        interpolated_depths2 = ""
        for i, (depth1, depth2) in enumerate(gaps):
            interpolated_depths2 +=  " " + str(depth1) +  " - " + str(depth2) + " m,"
        interpolated_depths = interpolated_depths1 + interpolated_depths2[:-1] + '.' 
        header.set_item_text('Processor Comments:', comment + interpolated_depths)
        
        header.set_item_text('Magnetic Influence on AZI:', 'Observed')
    
    
    ###########################################################################
    # INTERPOLATING THE AZIMUTHS BETWEEN THE FILTERED AREAS 
      
    all_depths = pd.DataFrame({'Depth (m)':np.arange(top,bottom + 0.1, 0.1)})
    all_depths['Depth (m)'] = round(all_depths['Depth (m)'],1)
    azidf = pd.merge(all_depths, azidf8, on='Depth (m)', how='left')
    
    # Check to make sure the AZIMUTH hasn't extrapolated further than necessary
    if float(azidf['Depth (m)'][-1:]) != azibottom:
        azidf = azidf[0:-1]

    # Obtaining Azimuths
    azi2 = azidf.iloc[:,1].values

    # Converting to radians and doing cosine of each
    for i in range(len(azi2)):
        azi2[i] = trig((azi2[i]) * math.pi/180)
        
    # Interpolating the missing depths - Process ensures it crosses over 0-360
    azidf["AZIMUTH"] = azi2
    azidf = azidf.interpolate(method='linear', limit_direction='both')

    # Reobtaining the azimuths in degrees
    azidf4 = pd.merge(all_depths, azidf8, on='Depth (m)', how='left')
    # Converting to radians and doing sin of each (used for + or - reading)
    for i in range(len(azidf4["AZIMUTH"])):
        azidf4["AZIMUTH"][i] = trig2((azidf4["AZIMUTH"][i]) * math.pi/180)
        
    # Interpolating the missing depths - Process ensures it crosses over 0-360
    azidf4 = azidf4.interpolate(method='linear', limit_direction='both')
    

    # Azimuth readings in radians of cosine   
    trigazi2 = azidf["AZIMUTH"]
    
    # Azimuth readings in radians of sin    
    trigazi3 = azidf4["AZIMUTH"]
    
    # Changing the values from sin due to the polarity change across 0-360
    for i in range(len(trigazi2)):
        # Range between 0 to pi which can be changed back to degrees as normal
        if (trigazi2[i] >= 0 and trigazi3[i] >= 0):
            trigazi3[i] = invtrig(trigazi3[i])
            trigazi3[i] = (trigazi3[i] * 180/np.pi)
        
        # Range between pi/2 to 3*pi/2 which requires slight change
        if (trigazi2[i] < 0 and trigazi3[i] < 0) or (trigazi2[i] < 0 and trigazi3[i] >= 0):
            trigazi3[i] = invtrig(trigazi3[i])
            trigazi3[i] = 180 - (trigazi3[i] * 180/np.pi)
            
        # Range between 3pi/2 to 2pi which requires slight change
        elif (trigazi2[i] >= 0 and trigazi3[i] < 0):
            trigazi3[i] = invtrig(trigazi3[i])
            trigazi3[i] = 360 + (trigazi3[i] * 180/np.pi) 


    azidf["AZIMUTH"] = trigazi3
    azidf2 = azidf
    
    return azidf2, no_rotation
    
    
#------------------------------------------------------------------------------
def main():
    # init list of rejected azimuth files (if user rejects any azimuths, have to manually edit + rotate in wellcad)
    rejected_azimuth_files = []   
    
    # Initially obtaining files to process
    try:
        # Find all files in the tv directory
        tvfiles = [os.path.join(tvdir, i) for i in os.listdir(tvdir)]
        
        # Filter files by allowable extensions, in this case wellcad files
        exts = ['.wcl', '.WCL', '.Wcl']
        tvfiles = [tvfile for tvfile in tvfiles if any(ext in tvfile for ext in exts)]
        
        # List containing all files processed already for filtering
        tvfilescompleted = [tvfile for tvfile in tvfiles if '_preliminary' not in tvfile]
        # List of scripted/QAQC'd files
        tvfilesscripted = [tvfile for tvfile in tvfiles if '!preliminary' in tvfile] 
        
        # Note that all files to process should have suffix _preliminary_
        tvfiles = [tvfile for tvfile in tvfiles if '_preliminary_' in tvfile]
        
        # Obtaining holeids of completed files
        tvfilesid = []
        for tvfilecomplete in tvfilescompleted:
            t = re.split(r'[\\\/]', tvfilecomplete)[-1]
            parts = t.split('_')
            tvfilesid.append(parts[1])
        
        # Filtering out files already processed
        for tvfileid in tvfilesid:
            for tvfile in tvfiles:
                if tvfileid in tvfile:
                    tvfiles.remove(tvfile)
                    
        tvfilesuncompleted = str(len(tvfiles))
        
        # Ordering tvfiles to have all 'hs_m' files processed first
        for tvfile in tvfiles:
                lista = [file for file in tvfiles if "_hs_m" in file]
                listb = [file for file in tvfiles if "_hs_m" not in file]
        tvfiles = lista + listb
    
    except Exception:
        print("Error occurred while trying to determine which tvfiles to process. \nEnsure all unwanted files have been deleted.")
        return
        
###################################################################################
    # Start of main loop opening and editing each file
    tally = 0
    for tvfile in tvfiles: 
        
        # Obtaining file name for errors
        filename = os.path.basename(tvfile).split(os.sep)[0]
        
        # Obtaining the holeid of the hole
        holeid = filename.split('_')[-3]
    
            
        # Setting app to WellCAD    
        app = wellcad.com.Application()  
               
        # Sets the hole its working on based on tvfile  
        if "_hs_m" in filename and tally == 0:
            print('---------------\nBeginning hs_m files')    
            tally += 1                   
            print('opening WELLCAD file: {} ...'.format(filename))       
        elif "_hs_m" in filename and tally > 0:                     
            print('opening WELLCAD file: {} ...'.format(filename))            
        elif "_hs_m" not in filename and tally > 0:
            print('Completed hs_m files\n---------------') 
            tally = 0                     
            print('opening WELLCAD file: {} ...'.format(filename))            
        else:
            print('opening WELLCAD file: {} ...'.format(filename))
            
        try:
            borehole = app.open_borehole(path=tvfile)
        except Exception:
            print(holeid + f"{Fore.RED}{Style.BRIGHT} cannot be opened. File likely hasn't been downloaded properly.{Style.RESET_ALL}")
            continue
        
        # Generate pandas dataframe, dictionary of logs and merged dataframe from WCL
        lognames, logdict = boreholeextract(borehole)

        # Obtaining Processor Comment for the hole and adding 'Automation:' for any possible items added later
        header = borehole.header
        comment = header.get_item_text('Processor Comments:')
        if len(comment) == 0:
            header.set_item_text('Processor Comments:', comment + 'Automation:')
        else:
            header.set_item_text('Processor Comments:', comment + ' Automation:')
        
        # Holeid thats logged in the WellCAD file
        holename = header.get_item_text('WELLNAME') or header.get_item_text('WELL')
        
        # Ensuring the name of the hole in WellCAD matches the holeid    
        if holename != holeid:
            print(f"{Fore.RED}{Style.BRIGHT}Hole name in file doesn't match filename: {Style.RESET_ALL}" + holeid)
            app.close_borehole(prompt_for_saving=False)
            continue
        
        
        ########################################################################### 
        # SECTION TO REMOVE HEADERS
        ###########################################################################                   
        # Define headers to remove from wcl file
        deltitles = ['AZI', 'TILT', 'L_GAMMA', 'SANG', 'SANGB', 'L_GAMMAR','L_GSANGB', 'L_GSANG', 'DEN(SS)', 'DEN(SS)#1', 'DEN(LS)', 'DEN(LS)#1', 'DEN(CDL)', 'COMP',]     
        if '_nsg' in filename:
            deltitles.extend(('L_SANGB', 'L_SANG'))
            
        # remove log names not in file from deltitles
        deltitles = [i for i in deltitles if i in lognames]
        
        # Loop for removing the unwanted headers
        for deltitle in deltitles: 
            try:                 
                borehole.remove_log(deltitle)            
            except Exception as error:
                print(error)

        ########################################################################### 
        # SECTION TO OBTAIN DATA AS TABLES/DF FOR CHECKS & ANALYSIS
        ###########################################################################        
        try:
            # Check to ensure if any azimuth logs are present in file
            required_lognames = ['L_SANGB', 'L_NSGSANGB', 'AZIMUTH']
            if any(item in lognames for item in required_lognames):
                pass
            else:
                print(f"{filename}{Fore.RED}{Style.BRIGHT} has no AZIMUTH logs present. Processing stopped, manual adjustments/processing required.{Style.RESET_ALL}")
                app.close_borehole(prompt_for_saving=False)
                continue
            
            
            # Obtaining data for AZIMUTH & TILT logs
            if '_m.' in filename:
                if 'L_SANGB' in lognames:
                    azilog, azitable, azidf, azitop, azicond = log_obtain(holeid, borehole, logdict, 'L_SANGB')
                    tiltlog, tilttable, tiltdf, tilttop, tiltcond = log_obtain(holeid, borehole, logdict, 'L_SANG')
                    
                elif 'AZIMUTH' in lognames:
                    azilog, azitable, azidf, azitop, azicond = log_obtain(holeid, borehole, logdict, 'AZIMUTH')
                    tiltlog, tilttable, tiltdf, tilttop, tiltcond = log_obtain(holeid, borehole, logdict, 'TILT')
            
            elif '_nsg.' in filename:
                azilog, azitable, azidf, azitop, azicond = log_obtain(holeid, borehole, logdict, 'L_NSGSANGB')
                tiltlog, tilttable, tiltdf, tilttop, tiltcond = log_obtain(holeid, borehole, logdict, 'L_NSGSANG')
                
            # Obtaining data for RBG log
            rgblog, rgbtable, rgbdf, rgbtop, rgbcond = log_obtain(holeid, borehole, logdict, 'RGB')

        except Exception as error:
            print(error)
            # Stops processing if one of these logs fail to be present
            app.close_borehole(prompt_for_saving=False)
            continue
        
        # Obtaining data for NG log
        nglog, ngtable, ngdf, ngtop, ngcond = log_obtain(holeid, borehole, logdict, 'NG')        
        # Obtaining data for CAL log
        callog, caltable, caldf, caltop, calcond = log_obtain(holeid, borehole, logdict, 'CAL')        
        # Obtaining data for MAG log
        maglog, magtable, magdf, magtop, magcond = log_obtain(holeid, borehole, logdict, 'MAG')        
        # Obtaining data for AMP log
        amplog, amptable, ampdf, amptop, ampcond = log_obtain(holeid, borehole, logdict, 'AMP')
                
        ###########################################################################
        # SET VISIBLE RANGE DEPTH 
        ###########################################################################       
        # Check logged depth against visual depth in WellCAD to prevent data loss
        # Console outputs if checks pass
        holedepthchecker(holeid, header, rgblog, amplog)
        
        # Specific check to see if the Azimuth & Tilt are the furtherest extends of the log
        viscond = False
        if calcond == False and magcond == False \
            and (round(azitop + 0.1,1) and round(tilttop + 0.1,1)) <= caltop \
            and (round(azitop + 0.1,1) and round(tilttop + 0.1,1)) <= ngtop \
            and (round(azitop + 0.1,1) and round(tilttop + 0.1,1)) <= magtop \
            and (round(azitop + 0.1,1) and round(tilttop + 0.1,1)) <= rgbtop:
            
            if azitable[1][1] < 0 and tilttable[1][1] < 0 and azitable[2][1] > 0 and tilttable[2][1] > 0:                   
                del azitable[1]
                del tilttable[1]
                
                azilog.data_table = tuple(azitable)
                tiltlog.data_table = tuple(tilttable)
                
                topdepthadjust = (azitop + 0.1)
                bottomdepthadjust = round(borehole.bottom_depth,1)
                borehole.set_visible_depth_range(top_depth=topdepthadjust, 
                                                 bottom_depth=bottomdepthadjust)                
                # Set visual condition to True to adjust visual display
                viscond = True
              
        # Fixing the visual display if needed and obtaining borehole top depth
        if viscond:
            top = azitop+0.1
        else:
            top = round(borehole.top_depth,1) 
            
        # Ensuring top depth is correctly set
        if rgblog.bottom_depth != rgblog.top_depth:  
            if rgbtop < top:
                top = rgbtop
        try:
            if nglog.bottom_depth != nglog.top_depth:
                if ngtop < top:
                    top = ngtop
                    
        except Exception:
            pass
        
        # Check if the ATV amplitude log (amplog) exists before accessing its attributes.
        # If the AMP log is missing in the WellCAD file, amplog will be None.
        # This avoids AttributeError when processing files without ATV amplitude logs.
        if amplog is not None and amplog.bottom_depth != amplog.top_depth:
            if amptop < top:
                top = amptop

        caltruetop = None
        if caltop == top and azitop != top:
            for i, (depth, reading) in enumerate(caltable[1:]):
                if reading > 0:
                    if caltruetop is None:
                        caltruetop = round(depth,1)
            toplist = [azitop, tilttop, caltruetop, magtop, rgbtop, ngtop]
            if nglog.bottom_depth == nglog.top_depth:
                toplist = toplist[:-1]
            top = min(toplist)
            
            if round(caltop,1) != round(caltruetop,1):
                while caltop < caltruetop:
                    del(caltable[1])
                    caltop=caltable[1][0]
            
        azibottom = round(azilog.bottom_depth,1)
        bottom = round(borehole.bottom_depth,1)
        
        # Ensuring bottom depth is capturing bottom of OTV logs
        rgbbottom = math.ceil(rgblog.bottom_depth * 10) / 10 
        if bottom < rgbbottom:
            bottom = rgbbottom
        
        # Just security of dataloss - typically only shows when casing goes above ground level
        otvtop = round(rgblog.top_depth, 1)
        if otvtop < 0:
            print(filename + f"{Fore.YELLOW}{Style.BRIGHT} has OTV Imaging above 0.0m which should be looked at.{Style.RESET_ALL}")
        # Ensuring visual range top starts at 0 if negative values                    
        if top < 0:
            borehole.set_visible_depth_range(top_depth = 0, bottom_depth = bottom)
            top = 0  
        
        ###########################################################################
        # Check to ensure correct number of logs are present
        nlogs = borehole.nb_of_logs
        
        # Gives information on unusual circumstances of logs present   
        if nlogs < 11:
            print("### --- WARNING : " + filename + f"{Fore.YELLOW}{Style.BRIGHT} may be missing logs. Please check.{Style.RESET_ALL} --- ###")
        elif nlogs > 11:
            print("### --- WARNING : " + filename + f"{Fore.YELLOW}{Style.BRIGHT} may have more logs than usual. Please check.{Style.RESET_ALL} --- ###")
                
        ###########################################################################
        # EDITING LOGS (CORRECT SCALES, NAMES, COLOUR, EXTRAPOLATING, ETC.)
        ########################################################################### 
        # AZIMUTH LOG                      
        azitable, azilog, com1a, com1b, com1c, azicomment, first_valid_reading_index1, last_valid_reading_index1 = log_editor(azitable, azilog, holeid, viscond, top, bottom, header, 'Azimuth')
        
        ###########################################################################
        # TILT LOG
        tilttable, tiltlog, com2a, com2b, com2c, tiltcomment, first_valid_reading_index2, last_valid_reading_index2 = log_editor(tilttable, tiltlog, holeid, viscond, top, bottom, header, 'Tilt')
        
        ###########################################################################
        # CAL LOG
        try:
            callog = cal_editor(holeid, calcond, caltable, callog, caldf, caltop, top, bottom, header)
        except Exception:
            pass
        
        ###########################################################################
        # NG LOG        
        try:
            nglog.scale_low = 0
            nglog.scale_high = 200
            nglog.filter = 3
        except Exception:
            pass
        
        # Checking to see if log is missing
        if ngcond == False:
            NGdata = list(set([t[1] for t in ngtable[1:]]))
            if all(element <= 0 for element in NGdata):
                print(filename + f"{Fore.YELLOW}{Style.BRIGHT} may be missing NG log.{Style.RESET_ALL}")
                    
        ###########################################################################
        # MAG LOG
        if magcond == False:
            maglog.scale_low = 0
            maglog.scale_high = 2
            
            # Checking to see if log is missing
            MAG_data = list(set([t[1] for t in magtable[1:]]))
            if all(element <= 0 for element in MAG_data):
                print(filename + f"{Fore.YELLOW}{Style.BRIGHT} may be missing MAG log.{Style.RESET_ALL}")
                    
        ###########################################################################
        # DEPTH LOG                    
        depthlog = borehole.depth
        depthlog.scale = 10
        depthlog.decimals = 1
        depthlog.horizontal_grid_spacing = 0.2
        
        
        ###########################################################################
        # Rotating OTV & ATV logs of '_hs' files
        ###########################################################################
        # Only use filter on tv files containing mag compass azimuth data
        if '_hs_m' in filename:            
            comment2 = header.get_item_text('Processor Comments:')
            # Filter mag compass azimuth data
            azidf = pd.DataFrame(azitable[1:], columns=["Depth (m)", "AZIMUTH"])
            min_samples = 15
            azidf2, no_rotation = azifilter(holeid, header, azidf, tiltdf, azitop, azibottom, top, bottom, min_samples, comment2, alpha=0.1)
            # Only applies to _hs_m files after check in function 
            if no_rotation == False:
                
                # Extract depth and azimuth data from azidf
                depth = azidf.iloc[:,0].values
                azivalue = azidf.iloc[:,1].values
                
                def azifilt():
                    class GraphApp:
                        def __init__(self, root):
                            self.root = root
                            self.rejected = False
                            self.root.title("Interactive Line Graph")
                            
                            self.min_samples = 15
            
                            # Create a Figure for the line graph
                            self.fig = Figure(figsize=(6, 4), dpi=100)
                            self.ax = self.fig.add_subplot(111)
                            
                            # Initial plot
                            self.line1, = self.ax.plot(depth,azivalue,label='raw azi', color='royalblue')
                            self.line2, = self.ax.plot(azidf2['Depth (m)'],azidf2['AZIMUTH'], label='interpolated azi', color='red')
                            self.ax.set_ylim(0,360)
                            
                            self.ax.set_title(holeid + "\nMin Samples = " + str(min_samples))
                            self.ax.set_xlabel("Depth (m)\n\nLess Strict Filter <----------------------------------------------------------------------------------> More Strict Filter")
                            self.ax.set_ylabel("Azimuth")
                            self.ax.legend()
                            
                            # Create a canvas to display the graph
                            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
                            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                            
            
                            
                            self.button_frame = ttk.Frame(self.root)
                            self.button_frame.pack(side=tk.BOTTOM, fill="x")
                            
                            # Reject button
                            self.rejectbutton = ttk.Button(self.button_frame, text='Reject',
                                                          command=self.reject)
                            self.rejectbutton.pack(side=tk.LEFT, fill="x", expand=1)
                            
                            # -5 Min Samples Button
                            self.negfivebutton = ttk.Button(self.button_frame, text='-5',
                                                          command=lambda: self.adjust_value(-5))
                            self.negfivebutton.pack(side=tk.LEFT, fill="x", expand=1)
                            
                            # -1 Min Samples Button
                            self.neg1button = ttk.Button(self.button_frame, text='-1',
                                                          command=lambda: self.adjust_value(-1))
                            self.neg1button.pack(side=tk.LEFT, fill="x", expand=1)
                            
                            # +1 Min Samples Button
                            self.pos1button = ttk.Button(self.button_frame, text='+1',
                                                          command=lambda: self.adjust_value(1))
                            self.pos1button.pack(side=tk.LEFT, fill="x", expand=1)
                            
                            # +5 Min Samples Button
                            self.posfivebutton = ttk.Button(self.button_frame, text='+5',
                                                          command=lambda: self.adjust_value(5))
                            self.posfivebutton.pack(side=tk.LEFT, fill="x", expand=1)
                            
                            # Accept Button
                            self.acceptbutton = ttk.Button(self.button_frame, text='Accept',
                                                          command=self.accept)
                            self.acceptbutton.pack(side=tk.LEFT, fill="x", expand=1)
                    
                        
                                                          
                        def adjust_value(self,change):
                            # New value                
                            new_min_samples = self.min_samples + change
                            if 5 <= new_min_samples <= 25:
                                # Apply azifilter with no value
                                azidf2,no_rotation = azifilter(holeid, header, azidf, tiltdf, azitop, azibottom, top, bottom, new_min_samples, comment2, alpha=0.1)   # Update the azidf dataframe
                                
                                if no_rotation:
                                    # Update title with new min samples value
                                    self.ax.set_title(holeid + "\nMin Samples = " + str(new_min_samples - 1) + " - NO CLUSTERS FOUND BEYOND THIS VALUE!")
                                    
                                    self.ax.relim()
                                    self.ax.autoscale_view()
                                    self.canvas.draw()
                                else:
                                    # Update title with new min samples value
                                    self.ax.set_title(holeid + "\nMin Samples = " + str(new_min_samples))
                                    # Carry forward new value
                                    self.min_samples=new_min_samples
                                    # Update plot data
                                    self.line2.set_xdata(azidf2['Depth (m)'])
                                    self.line2.set_ydata(azidf2['AZIMUTH'])
                                    
                                    self.ax.relim()
                                    self.ax.autoscale_view()
                                    self.canvas.draw()
                            
                                                        
                        def accept(self):
                            self.rejected = False
                            # Convert mag compass azimuth data back to tuple
                            azilogtuple = logtotuple(azidf2)
                                                                
                            # Set log data table to new filtered data
                            azilog.data_table = azilogtuple
                            
                            # Image rotation check across 0 - 360 transition        
                            for i in range(1, len(azilogtuple) - 1):
                                current_value = round(azilogtuple[i][1],1)
                                next_depth = round(azilogtuple[i+1][0],1)
                                next_value = round(azilogtuple[i+1][1],1)
                                
                                # Checking for the abnormal value if it exists between 0.01 and 359.99
                                if (current_value >= 359.95 and 0.5 <= next_value <= 355) or (current_value <= 0.05 and 5 <= next_value <= 359.5):
                                    # Replace the value associated with the bad depth with 0.01 m.
                                    for i, (depth, value) in enumerate(azilogtuple):
                                        if depth == next_depth:
                                            temp = list(azilogtuple)                                                 
                                            temp[i] = (depth, 0.01)
                                            azilogtuple = tuple(temp)
                                    azilog.data_table = azilogtuple
                                            
                                if 5 <= abs(next_value - current_value) <= 350:
                                    # Replace the value associated with the bad depth with the value of the one just before.
                                    for i, (depth, value) in enumerate(azilogtuple):
                                        if depth == next_depth:
                                            temp = list(azilogtuple)                                                 
                                            temp[i] = (depth, round(azilogtuple[i-1][1],1))
                                            azilogtuple = tuple(temp)
                                    azilog.data_table = azilogtuple
                                
                                
                            # Rotate logs clockwise by filtered azimuth
                            rotatelogs = ['RGB', 'IMG', 'AMP', 'TT', 'RGB#1', 'IMG#1', 'AMP#1', 'TT#1']
                            
                            # Remove log names not in file from rotatelogs
                            logs = [borehole.get_log(i) for i in range(nlogs)]
                            lognames2 = [log.name for log in logs]
                            rotatelogs = [i for i in rotatelogs if i in lognames2]
                             
                            for rotatelog in rotatelogs: 
                                try:
                                    borehole.rotate_image(log=rotatelog, 
                                                          prompt_user=False,
                                                          config="RotateBy='AZIMUTH'")
                                except:
                                    continue
                        
                            self.root.destroy()
                        
                        
                        def reject(self):
                            header.set_item_text('Processor Comments:', comment2)
                            print(f"{Fore.RED}{Style.BRIGHT}Rejected Azimuth Filter for: {Style.RESET_ALL}" + holeid)
                            print("MANUALLY EDIT AND ROTATE AZI IN WELLCAD!")
                            self.rejected = True
                            self.root.destroy()
                    
                    app = GraphApp(root)
                    root.mainloop()
                    return app.rejected
        
                if __name__ == "__main__":
                    root = tk.Tk()
                    root.geometry("1200x800")
                    azimuth_rejected = azifilt()
                    root.quit()
          
            
        # Only rotate wcl file image data if high side, not true north already
        if '_hs_nsg' in filename: 
           
            # Image rotation check across 0 - 360 transition        
            for i in range(1, len(azitable) - 1):
                current_value = round(azitable[i][1],1)
                next_depth = round(azitable[i+1][0],1)
                next_value = round(azitable[i+1][1],1)
                
                # Checking for the abnormal value if it exists between 0.01 and 359.99
                if (current_value >= 359.9 and 0.1 <= next_value <= 359.5) or (current_value <= 0.1 and 0.5 <= next_value <= 359.9):
                    # Replace the value associated with the bad depth with 0.01 m.
                    for i, (depth, value) in enumerate(azitable):
                        if depth == next_depth:                         
                            azitable[i] = (depth, 0.01)
                    azilog.data_table = tuple(azitable)
                            
                            
            # Rotate logs clockwise by filtered azimuth
            rotatelogs = ['RGB', 'IMG', 'AMP', 'TT', 'RGB#1', 'IMG#1', 'AMP#1', 'TT#1']
            try:
                # Remove log names not in file from rotatelogs
                logs = [borehole.get_log(i) for i in range(nlogs)]
                lognames2 = [log.name for log in logs]
                rotatelogs = [i for i in rotatelogs if i in lognames2]
                
                for rotatelog in rotatelogs: 
                    borehole.rotate_image(log=rotatelog, 
                                          prompt_user=False,
                                          config="RotateBy='AZIMUTH'")
            except:
                continue
              
        
        # Ensure 'Magnetic influence on Azi' is set to N/A
        if '_nsg.' in filename:
            header.set_item_text('Magnetic Influence on AZI:', 'N/A')   
            
         
        ###########################################################################
        # Check to see if Processor Comment has changed and cleaning it up if possible
        new_processor_comment = header.get_item_text('Processor Comments:')
        # print(new_processor_comment + "t")
        if new_processor_comment[-1] == ':':
            header.set_item_text('Processor Comments:', comment)
        else:        
            if com1a and com2a:
                if str(round(azitable[1:][first_valid_reading_index1][0],1)) == str(round(tilttable[1:][first_valid_reading_index2][0],1)) and str(azitable[1:][last_valid_reading_index1][0]) == str(tilttable[1:][last_valid_reading_index2][0]):
                    combined = " Azimuth and Tilt extrapolated above " + str(round(tilttable[1:][first_valid_reading_index2][0],1)) + " m and below " + str(tilttable[1:][last_valid_reading_index2][0]) + " m."
                    mod_processor_comment = new_processor_comment.replace(azicomment,"")
                    mod_processor_comment = mod_processor_comment.replace(tiltcomment, combined)
                    header.set_item_text('Processor Comments:', mod_processor_comment)
                    
            elif com1b and com2b:
                if str(round(azitable[1:][first_valid_reading_index1][0],1)) == str(round(tilttable[1:][first_valid_reading_index2][0],1)):
                    combined = " Azimuth and Tilt extrapolated above " + str(round(tilttable[1:][first_valid_reading_index2][0],1)) + " m."
                    mod_processor_comment = new_processor_comment.replace(azicomment,"")
                    mod_processor_comment = mod_processor_comment.replace(tiltcomment, combined)
                    header.set_item_text('Processor Comments:', mod_processor_comment)
            
            elif com1c and com2c:
                if str(azitable[1:][last_valid_reading_index1][0]) == str(tilttable[1:][last_valid_reading_index2][0]):
                    combined = " Azimuth and Tilt extrapolated below " + str(tilttable[1:][last_valid_reading_index2][0]) + " m."
                    mod_processor_comment = new_processor_comment.replace(azicomment,"")
                    mod_processor_comment = mod_processor_comment.replace(tiltcomment, combined)
                    header.set_item_text('Processor Comments:', mod_processor_comment)
            
                     
        outfile = str([tvfile.replace("_preliminary_", "!preliminary_")])
    
        if '_hs' in filename: 
            outfile = outfile.replace("_hs", "_tn")
            
        outfile = outfile[2:-2]    
        
        # If azimuth filter was rejected, add the marker to the filename for easy ID
        if 'azimuth_rejected' in locals() and azimuth_rejected:
            outfile = re.sub(r'(!preliminary)', r'NEED_MANUAL_AZI_EDIT_ROTATE_\1', outfile, count=1)
            rejected_azimuth_files.append(os.path.basename(outfile))
        
        borehole.save_as(outfile)           
        app.close_borehole(prompt_for_saving=False)
            
    
    ###########################################################################
    # STATISTICS FOR CONSOLE OUTPUT
    ###########################################################################
    # Refreshing directory to obtain new list of files
    tvfiles = [os.path.join(tvdir, i) for i in os.listdir(tvdir)]   
    exts = ['.wcl', '.WCL', '.Wcl'] 
    tvfilesafter = [i for i in tvfiles if any(ext in i for ext in exts)]   
    
    # List of unscripted files
    tvfilesuncompleteda = [tvfile for tvfile in tvfilesafter if '_preliminary' in tvfile]
    # List of scripted/QAQC'd files
    tvfilesscripteda = [tvfile for tvfile in tvfilesafter if '!preliminary' in tvfile] 
    
    # Number of scripted files just done
    number_scripted = str(len(tvfilesscripteda) - len(tvfilesscripted)) 
    
    # Obtaining holeids of scripted files
    tvfilesscriptafterid = []
    for i in tvfilesscripteda:
        t = re.split(r'[\\\/]', i)[-1]
        parts = t.split('_')
        tvfilesscriptafterid.append(parts[1])
    
    # Removing all processed files from list
    for j in tvfilesscriptafterid:
        for i in tvfilesuncompleteda:
            if j in i:
                tvfilesuncompleteda.remove(i)
    
    # Obtaining the ID/s of the unprocessed files
    tvfilesuncompletedaid = []
    for i in tvfilesuncompleteda:
        t = re.split(r'[\\\/]', i)[-1]
        parts = t.split('_')
        tvfilesuncompletedaid.append(parts[2])
    
    
    if number_scripted == tvfilesuncompleted:                     
        print(f"---------------\n{Fore.GREEN}{Style.BRIGHT}Complete{Style.RESET_ALL}")
    else:
        print(f"---------------\n{Fore.YELLOW}{Style.BRIGHT}Complete{Style.RESET_ALL}")
    print(number_scripted + '/' + tvfilesuncompleted + ' files proccessed.')
    if len(tvfilesuncompletedaid) != 0:
        print('Holes not processed are: ' + str(tvfilesuncompletedaid))

    # Print out all files with rejected azimuth at end of script
    if rejected_azimuth_files:
        print(f"\n{Fore.RED}{Style.BRIGHT}### --- WARNING : FOLLOWING FILES HAD AZI FILTER REJECTED. NEED TO MANUALLY EDIT + ROTATE IN WELLCAD! --- ###{Style.RESET_ALL}")
        for fname in rejected_azimuth_files:
            print(f"  - {fname}")
# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    main()  
