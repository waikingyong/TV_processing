# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:41:18 2023

Original authors:
    Riley.Elstermann
    Benjamin.Boland

Script to recreate wellcad wsg vbs scripts in python.

Help with wellcad python API:
    https://pywellcad.readthedocs.io/

Problems or suggestions for pywellcad:
    https://github.com/alt-lu/pywellcad

"""

import wellcad.com
import re
import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from statsmodels.nonparametric.smoothers_lowess import lowess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.font import Font
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from colorama import init as colorama_init
from colorama import Fore, Style
from datetime import datetime
colorama_init()

# ------------------------------- SUPPORT / EXISTING FUNCTIONS -------------------------------

def boreholeextract(borehole):
    """Extracts specified logs from a WellCAD borehole object into a dictionary of DataFrames.

    Args:
        borehole (wellcad.com.Borehole): The WellCAD borehole object to extract logs from.

    Returns:
        tuple: A tuple containing:
            - list: The names of the logs that were processed.
            - dict: A dictionary where keys are log names and values are pandas
              DataFrames containing the log data ('Depth (m)', log_name).
    """
    nlogs = borehole.nb_of_logs
    logdict = {}
    logs = [borehole.get_log(i) for i in range(nlogs)]
    lognames = [log.name for log in logs]
    processlogs = [
        'AZI', 'AZIMUTH', 'TILT', 'NG',
        'SANG', 'SANGB', 'L_GAMMAR', 'L_GAMMA',
        'CAL', 'MAG', 'L_SANGB', 'L_SANG',
        'L_NSGSANGB', 'L_NSGSANG', 'L_GSANGB', 'L_GSANG',
        'DEN(SS)', 'DEN(SS)#1', 'DEN(LS)', 'DEN(LS)#1', 'DEN(CDL)', 'COMP'
    ]
    lognames = [ln for ln in lognames if ln in processlogs]
    for name in lognames:
        log = borehole.get_log(name)
        try:
            data = np.array(log.data_table[1:])
            tmpdf = pd.DataFrame(data, columns=['Depth (m)', name]).round(2)
            depthfilt = tmpdf['Depth (m)'] >= 0
            tmpdf.loc[depthfilt, 'Depth (m)'] = tmpdf.loc[depthfilt, 'Depth (m)'].round(1)
            logdict[name] = tmpdf
        except Exception:
            continue
    return lognames, logdict

def logtotuple(logdf):
    """Converts a pandas DataFrame into a tuple of tuples suitable for WellCAD log data.

    The first inner tuple contains the column headers. Subsequent tuples contain the data rows.

    Args:
        logdf (pd.DataFrame): The DataFrame to convert.

    Returns:
        tuple: A tuple of tuples representing the DataFrame with its header.
    """
    cols = [logdf.columns.tolist()]
    data = [tuple(r) for r in logdf.values]
    return tuple([tuple(r) for r in cols + data])

def comment_match(comment):
    """Searches for a 'blocked at' depth in a comment string and returns a calculated depth.

    It looks for patterns like 'blocked at 123.4m' and adds 7 to the found depth.

    Args:
        comment (str): The comment string to search within.

    Returns:
        float: The matched depth plus 7, or 0 if no match is found.
    """
    match = re.search(r'\bblocked(?: at| @)? (\d+(\.\d+)?)(?:m\b)?', comment, re.IGNORECASE)
    if match:
        depth = float(match.group(1))
        return depth + 7
    return 0

def holedepthchecker(holeid, header, rgblog, amplog):
    """Checks for depth discrepancies in borehole data from different sources (WSG, Epiroc).

    It compares driller's depth, logger's depth, and log bottom depths from the
    WellCAD header and log objects, printing warnings for potential issues.

    Args:
        holeid (str): The identifier for the borehole.
        header (wellcad.com.Header): The WellCAD header object.
        rgblog (wellcad.com.Log): The RGB image log object.
        amplog (wellcad.com.Log): The Amplitude image log object.
    """
    try:
        drillers_depth = float(header.get_item_text('DRDP') or 0)
        loggers_depth = float(header.get_item_text('LOTD') or 0) + 5
        log_bottom = float(header.get_item_text('LB') or 0) + 5
        wsg_comment = header.get_item_text('Comments') or ""
        blocked_depth = comment_match(wsg_comment)
        check = False
        if loggers_depth < drillers_depth and blocked_depth < loggers_depth:
            print(f"{holeid} {Fore.YELLOW}{Style.BRIGHT}possible depth discrepancy (Logger < Driller depth).{Style.RESET_ALL}")
            check = True
        if check and log_bottom < (loggers_depth - 3) and blocked_depth < loggers_depth:
            print(f"{holeid} {Fore.YELLOW}{Style.BRIGHT}log bottom shallower than expected. Log Bottom: {log_bottom:.2f}m, Logger depth: {loggers_depth:.2f}m, Driller depth: {drillers_depth:.2f}m{Style.RESET_ALL}")
    except Exception as e:
        print(f"Depth check (WSG) warning {holeid}: {e}")
    try:
        epiroc_raw = header.get_item_text('DEPTHDRILLER1')
        if epiroc_raw:
            m = re.search(r"(\d+(\.\d+)?)", epiroc_raw)
            epiroc_depth = float(m.group(1)) if m else 0
            otvbottom = round(rgblog.bottom_depth, 1) if rgblog else None
            atvbottom = round(amplog.bottom_depth, 1) if amplog else None
            epiroc_comment = header.get_item_text('COMMENT1') or ""
            blocked_depth2 = comment_match(epiroc_comment)
            bottom_combo = None
            if otvbottom is not None and atvbottom is not None:
                bottom_combo = max(otvbottom, atvbottom) + 5
            elif otvbottom is not None:
                bottom_combo = otvbottom + 5
            elif atvbottom is not None:
                bottom_combo = atvbottom + 5
            if bottom_combo and bottom_combo < epiroc_depth and blocked_depth2 < bottom_combo:
                print(f"{holeid} {Fore.YELLOW}{Style.BRIGHT}Epiroc possible depth discrepancy. Tool Bottom: {bottom_combo:.2f}m, Driller Depth: {epiroc_depth:.2f}m{Style.RESET_ALL}")
    except Exception as e:
        print(f"Depth check (Epiroc) warning {holeid}: {e}")

def log_obtain(holeid, borehole, logdict, logname):
    """Obtains a specific log and its associated data from a borehole.

    Args:
        holeid (str): The identifier for the borehole.
        borehole (wellcad.com.Borehole): The WellCAD borehole object.
        logdict (dict): A dictionary of pre-extracted log DataFrames.
        logname (str): The name of the log to obtain.

    Returns:
        tuple: A tuple containing:
            - wellcad.com.Log or None: The log object, or None if not found.
            - list: The log's data table, or an empty list on failure.
            - pd.DataFrame: The log's data as a DataFrame, or an empty one.
            - float or None: The log's top depth, or None on failure.
            - bool: A flag indicating if the log is missing (True) or present (False).
    """
    try:
        log = borehole.get_log(logname)
    except Exception:
        log = None
    if log is None:
        print(holeid + f"{Fore.YELLOW}{Style.BRIGHT} missing {logname} log.{Style.RESET_ALL}")
        if logname.upper() == "CAL":
            print(holeid + f"{Fore.YELLOW}{Style.BRIGHT} CAL (caliper) log missing - check file.{Style.RESET_ALL}")
        return None, [], pd.DataFrame(), None, True
    try:
        if logname in ['RGB', 'AMP']:
            top = math.floor(log.top_depth * 10) / 10
        elif logname in ['L_SANGB', 'L_SANG', 'L_NSGSANGB', 'L_NSGSANG', 'AZIMUTH', 'TILT']:
            top = round(log.top_depth, 1)
        elif len(str(log.top_depth)) > 3 and int(str(log.top_depth)[3]) >= 8:
            top = round(log.top_depth, 1)
        else:
            top = math.floor(log.top_depth * 10) / 10
    except Exception:
        top = None
    table = []
    df = pd.DataFrame()
    if logname not in ['RGB', 'AMP']:
        try:
            table = list(log.data_table)
        except Exception:
            table = []
        df = logdict.get(logname, pd.DataFrame())
    return log, table, df, top, False

def log_editor(table, log, holeid, viscond, top, bottom, header, logtype):
    """Edits and standardizes a log's properties and data table.

    This includes setting name and color, scaling, filling gaps, and extrapolating
    to match specified top and bottom depths.

    Args:
        table (list): The log's data table.
        log (wellcad.com.Log): The WellCAD log object to edit.
        holeid (str): The identifier for the borehole.
        viscond (bool): A visibility condition flag.
        top (float): The target top depth for the log.
        bottom (float): The target bottom depth for the log.
        header (wellcad.com.Header): The WellCAD header object to update with comments.
        logtype (str): The type of log being edited (e.g., 'Azimuth', 'Tilt').

    Returns:
        tuple: A tuple containing:
            - list: The modified log data table.
            - wellcad.com.Log: The modified log object.
            - bool: Flag for extrapolation at top and bottom.
            - bool: Flag for extrapolation only at top.
            - bool: Flag for extrapolation only at bottom.
            - str: The comment addition string.
            - int or None: The index of the first valid data point.
            - int or None: The index of the last valid data point.
    """
    if len(table) == 1:
        print(holeid + f"{Fore.RED}{Style.BRIGHT} {logtype} log has no data.{Style.RESET_ALL}")
        return table, log, False, False, False, "", None, None
    if logtype == 'Azimuth':
        log.name = 'AZIMUTH'; log.pen_color = 16711680; log.scale_high = 360
    elif logtype == 'Tilt':
        log.name = 'TILT'; log.pen_color = 255; log.scale_high = 40
    log.scale_low = 0
    table = [((r[0], r[1]) if i == 0 else (round(r[0], 1), round(r[1], 2))) for i, r in enumerate(table)]
    table[0] = ('Depth [m]', logtype.upper())
    if log.left_position > 0.123 or log.right_position > 0.26:
        log.left_position = 0.1229; log.right_position = 0.259
    log_top_actual = top if viscond else round(log.top_depth, 1)
    logbottom = round(log.bottom_depth, 1)
    if logbottom != bottom:
        while logbottom < bottom:
            logbottom = round(logbottom + 0.1, 1)
            table.append((logbottom, -999.25))
    if log_top_actual != top:
        while log_top_actual > top:
            log_top_actual = round(log_top_actual - 0.1, 1)
            table.insert(1, (log_top_actual, -999.25))
    first_valid = last_valid = None
    for i, (d, v) in enumerate(table[1:]):
        if v > 0:
            if first_valid is None:
                first_valid = i
            last_valid = i
    check1 = check2 = False
    for i, (d, v) in enumerate(table[1:]):
        if v < 0:
            if first_valid is not None and i < first_valid:
                table[i + 1] = (d, table[1:][first_valid][1]); check1 = True
            elif last_valid is not None and i > last_valid:
                table[i + 1] = (d, table[1:][last_valid][1]); check2 = True
            elif first_valid is not None and last_valid is not None and first_valid < i < last_valid:
                table[i + 1] = (d, table[i][1])
    addition = ""; com1a = com1b = com1c = False
    if first_valid is not None:
        fd = round(table[1:][first_valid][0], 1)
        ld = round(table[1:][last_valid][0], 1)
        if fd > 0:
            if check1 and check2:
                addition = f" {logtype} extrapolated above {fd} m and below {ld} m."
                com1a = True
            elif check1 and not check2:
                addition = f" {logtype} extrapolated above {fd} m."
                com1b = True
        elif check2 and not check1:
            addition = f" {logtype} extrapolated below {ld} m."
            com1c = True
    if addition:
        pc = header.get_item_text('Processor Comments:')
        header.set_item_text('Processor Comments:', pc + addition)
    log.data_table = tuple(table)
    return table, log, com1a, com1b, com1c, addition, first_valid, last_valid

def weighted_median(df, val, weight):
    """Calculates the weighted median of a DataFrame column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        val (str): The name of the column to calculate the median for.
        weight (str): The name of the column containing the weights.

    Returns:
        float: The weighted median value.
    """
    df_sorted = df.sort_values(val)
    csum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum() / 2.
    return df_sorted[csum >= cutoff][val].iloc[0]

def reject_outliers(data, median, m=2.):
    """Filters outliers from a dataset based on the median absolute deviation.

    Args:
        data (pd.Series or np.ndarray): The data to filter.
        median (float): The median of the dataset.
        m (float, optional): The Z-score-like threshold. Defaults to 2.0.

    Returns:
        pd.Series or np.ndarray: The data with outliers removed.
    """
    d = np.abs(data - median)
    mdev = np.median(d)
    s = 2 * d / (mdev if mdev else 1.)
    return data[s < m]

def cal_editor(holeid, calcond, caltable, callog, caldf, caltop, top, bottom, header):
    """Processes and edits a caliper (CAL) log.

    This includes scaling, cleaning data, calculating a representative value for
    the bottom of the hole, and extrapolating to fill gaps.

    Args:
        holeid (str): The identifier for the borehole.
        calcond (bool): A condition flag, if True the function exits early.
        caltable (list): The caliper log's data table.
        callog (wellcad.com.Log): The WellCAD caliper log object.
        caldf (pd.DataFrame): The DataFrame of caliper data.
        caltop (float): The original top depth of the caliper log.
        top (float): The target top depth for all logs.
        bottom (float): The target bottom depth for all logs.
        header (wellcad.com.Header): The WellCAD header object to update.
    """
    if calcond or len(caltable) == 1:
        if not calcond and len(caltable) == 1:
            print(holeid + f"{Fore.RED}{Style.BRIGHT} CAL log has no data.{Style.RESET_ALL}")
        return
    callog.scale_low = 0; callog.scale_high = 200
    caltable = [((r[0], r[1]) if i == 0 else (round(r[0], 1), round(r[1], 2))) for i, r in enumerate(caltable)]
    calbottom = round(callog.bottom_depth, 1)
    caldf = caldf[caldf['CAL'] >= 0]
    valid_depth = caldf['Depth (m)'].iloc[-1] - caldf['Depth (m)'].iloc[0]
    if valid_depth < 100:
        y = 10.3456 - 10.1247 * math.pow(valid_depth, 0.00361592)
        highest = int(round(valid_depth * y, 1) * 10 + 1)
        caldf = caldf[-highest:-1]
    else:
        caldf = caldf[-51:-1]
    exp_weight = np.exp(np.linspace(-0.3, 3.2, len(caldf)))
    caldf['WEIGHT'] = exp_weight
    w_med = weighted_median(caldf, 'CAL', 'WEIGHT')
    non_out = reject_outliers(caldf['CAL'], w_med)
    caldf = caldf[caldf.index.isin(non_out.index)]
    exp_weight2 = np.exp(np.linspace(-0.3, 3.2, len(caldf)))
    caldf['WEIGHT'] = exp_weight2
    w2 = weighted_median(caldf, 'CAL', 'WEIGHT')
    if caltop != top:
        while round(caltop, 1) > top:
            caltop = round(caltop - 0.1, 1)
            caltable.insert(1, (caltop, -999.25))
    first_valid = last_valid = None
    for i, (d, v) in enumerate(caltable[1:]):
        if v > 0:
            if first_valid is None:
                first_valid = i
            last_valid = i
    check5 = check6 = False
    if calbottom != bottom:
        while calbottom < bottom:
            calbottom = round(calbottom + 0.1, 1)
            caltable.append((calbottom, w2))
            check6 = True
    for i, (d, v) in enumerate(caltable[1:]):
        if v < 0:
            if i < first_valid:
                caltable[i + 1] = (d, caltable[1:][first_valid][1]); check5 = True
            elif i > last_valid - 1:
                caltable[i + 1] = (d, w2); check6 = True
            elif first_valid < i < last_valid:
                caltable[i + 1] = (d, caltable[i][1])
    for j in [-1, 0, 1, 2, 3]:
        if caltable[last_valid - j][1] > w2 + 4 or caltable[last_valid - j][1] < w2 - 4:
            caltable[last_valid - j] = (caltable[last_valid - j][0], w2)
        else:
            break
    addition = ""
    if first_valid is not None:
        topd = round(caltable[1:][first_valid][0], 1)
        botd = round(caltable[1:][last_valid][0] - j * 0.1 - 0.1, 1)
        if topd > 0:
            if check5 and check6:
                addition = f" Caliper extrapolated above {topd} m and below {botd} m."
            elif check5 and not check6:
                addition = f" Caliper extrapolated above {topd} m."
        elif check6 and not check5:
            addition = f" Caliper extrapolated below {botd} m."
    if addition:
        current = header.get_item_text('Processor Comments:')
        header.set_item_text('Processor Comments:', current + addition)
    callog.data_table = tuple(caltable)

def alpha_trim_filter(depth, azi_deg, window_m=2, alpha=0.50, passes=2):
    """Applies a multi-pass, alpha-trimmed mean filter to azimuth data.

    This is a robust smoothing filter that is less sensitive to outliers than a
    simple moving average.

    Args:
        depth (np.ndarray): Array of depth values.
        azi_deg (np.ndarray): Array of azimuth values in degrees.
        window_m (int, optional): The size of the moving window in meters. Defaults to 2.
        alpha (float, optional): The proportion of data to trim from each end of the
            sorted window. Defaults to 0.50.
        passes (int, optional): The number of filter passes to apply. Defaults to 2.

    Returns:
        np.ndarray: The smoothed azimuth data in degrees.
    """
    if len(depth) < 3:
        return azi_deg
    
    azi_rad = np.unwrap(np.deg2rad(azi_deg))
    
    for _ in range(passes):
        step = pd.Series(depth).diff().replace(0, np.nan).median()
        if pd.isna(step) or step <= 0:
            step = 0.1
        
        k = int(round(window_m / step))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        
        half = k // 2
        padded = np.pad(azi_rad, (half, half), mode="edge")
        out = np.zeros_like(azi_rad)
        t_each = int(np.floor((alpha * k) / 2.0))
        
        for i in range(len(azi_rad)):
            window_slice = padded[i:i + k]
            sw = np.sort(window_slice)
            if t_each > 0:
                sw = sw[t_each: -t_each]
            out[i] = np.mean(sw)
        
        azi_rad = out # Use the output of this pass as the input for the next

    return np.mod(np.rad2deg(azi_rad), 360)

def azifilter(holeid, header, azidf, tiltdf, azitop, azibottom, top, bottom, min_samples, comment, eps=3.0, alpha=0.1):
    """Filters azimuth data using DBSCAN to identify and interpolate the main trend.

    It identifies the core trend of azimuth data, ignoring noise and spikes,
    then interpolates this trend over a regular depth interval.

    Args:
        holeid (str): The identifier for the borehole.
        header (wellcad.com.Header): The WellCAD header object.
        azidf (pd.DataFrame): DataFrame with 'Depth (m)' and 'AZIMUTH' columns.
        tiltdf (pd.DataFrame): DataFrame with tilt data for cross-checking.
        azitop (float): The top depth of the azimuth data.
        azibottom (float): The bottom depth of the azimuth data.
        top (float): The target top depth for the final output.
        bottom (float): The target bottom depth for the final output.
        min_samples (int): The `min_samples` parameter for DBSCAN.
        comment (str): The base comment to which messages can be appended.
        eps (float, optional): The `eps` parameter for DBSCAN. Defaults to 3.0.
        alpha (float, optional): Unused parameter. Defaults to 0.1.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The filtered and interpolated azimuth data.
            - bool: A flag indicating if rotation should be skipped (True) or
              applied (False).
    """
    depth = azidf.iloc[:, 0].values
    azi_deg = azidf.iloc[:, 1].values
    stack = np.column_stack((depth, azi_deg))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(stack)
    labels = db.labels_
    
    ###### --------- COMMENTED OUT BY WAI.YONG ON 17 SEPT 2025. -------- ######
    # DISCUSSED WITH LOUISA M. HS SHOULD ALWAYS BE ROTATED TO TN , REGARDLESS OF TILT. 
    # -------------------------------------------------------------------------
    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # if n_clusters == 0:
    #     tdf = tiltdf[tiltdf['L_SANG'] >= 0]
    #     if not tdf.empty and tdf['L_SANG'].median() <= 5:
    #         print(holeid, f"{Fore.YELLOW}{Style.BRIGHT}treated as TN_M because median Tilt < 5 deg and no coherent azi trend found. Azi rotation NOT applied.{Style.RESET_ALL}")
    #         pc = header.get_item_text('Processor Comments:')
    #         header.set_item_text('Processor Comments:', pc + "File considered as TN_M because median Tilt < 5 deg and no coherent azimuth trend found. Azi rotation NOT applied.")
    #     else:
    #         print(holeid, f"{Fore.RED}{Style.BRIGHT}no coherent azimuth trend found. Median Tilt > 5 deg. Manual review recommended.{Style.RESET_ALL}")
    #     return azidf, True
    # -------------------------------------------------------------------------

    core_mask = np.zeros_like(labels, dtype=bool)
    core_idx = getattr(db, "core_sample_indices_", [])
    if len(core_idx):
        core_mask[core_idx] = True
    core_df = pd.DataFrame(columns=['Depth (m)', 'AZIMUTH'])
    for k in set(labels):
        if k == -1:
            continue
        sel = (labels == k) & core_mask
        pts = stack[sel]
        if pts.size:
            core_df = pd.concat([core_df, pd.DataFrame(pts, columns=['Depth (m)', 'AZIMUTH'])], ignore_index=True)
    if core_df.empty:
        print(holeid, f"{Fore.RED}{Style.BRIGHT}Core sample extraction failed - manual review.{Style.RESET_ALL}")
        return azidf, True
    core_df = core_df.sort_values('Depth (m)').drop_duplicates('Depth (m)', keep='first')
    depth_checker = list(logtotuple(core_df))
    if depth_checker:
        depth_checker += [(azibottom, 0)]
        if depth_checker[0][0] != azitop:
            depth_checker.insert(1, (azitop, 0))
    gaps = []
    for i in range(1, len(depth_checker) - 1):
        d1 = round(depth_checker[i][0], 1)
        d2 = round(depth_checker[i + 1][0], 1)
        if d2 - d1 > 2:
            gaps.append([d1, d2])
    
    # --- FIX: Check if comment already exists before adding it ---
    if gaps and "Azimuth interpolated between" not in comment:
        gap_text = " Azimuth interpolated between" + "".join(f" {g[0]} - {g[1]} m," for g in gaps)
        gap_text = gap_text[:-1] + '.'
        header.set_item_text('Processor Comments:', comment + gap_text)
        header.set_item_text('Magnetic Influence on AZI:', 'Observed')

    master_depth = np.arange(top, bottom + 0.1, 0.1).round(1)
    cdepth = core_df['Depth (m)'].values
    cazi = core_df['AZIMUTH'].values
    sort_idx = np.argsort(cdepth)
    cdepth = cdepth[sort_idx]; cazi = cazi[sort_idx]
    unwrapped = np.unwrap(np.deg2rad(cazi))
    interp_unwrapped = np.interp(master_depth, cdepth, unwrapped)
    interp_deg = (np.rad2deg(interp_unwrapped)) % 360.0
    out_df = pd.DataFrame({'Depth (m)': master_depth, 'AZIMUTH': interp_deg})
    if abs(out_df['Depth (m)'].iloc[-1] - azibottom) > 0.05:
        out_df = out_df.iloc[:-1, :]
    return out_df, False

def should_adjust_visible_depth(log_data):
    """
    Checks if conditions are met to safely adjust the visible depth range at the start of processing.
    This is a specific check for a scenario where azimuth/tilt logs have an initial bad reading.

    Args:
        log_data (dict): A dictionary containing all necessary variables for the check,
                         including log conditions, tables, and top depths.

    Returns:
        bool: True if the conditions are met, False otherwise.
    """
    # Unpack for readability
    calcond, magcond = log_data['calcond'], log_data['magcond']
    azitop, tilttop = log_data['azitop'], log_data['tilttop']
    caltop, ngtop, magtop = log_data['caltop'], log_data['ngtop'], log_data['magtop']
    rgblog_top = log_data['rgblog_top']
    azitable, tilttable = log_data['azitable'], log_data['tilttable']

    # Condition 1: Caliper and Magnetometer logs must be valid.
    if calcond or magcond:
        return False

    # Condition 2: Azimuth and Tilt tops must be shallower or equal to other logs.
    # Note: The original logic `(A and B) <= C` is equivalent to `B <= C`.
    # This is preserved to avoid changing program logic.
    adjusted_azi_tilt_top = round(tilttop + 0.1, 1)
    if not (adjusted_azi_tilt_top <= caltop and
            adjusted_azi_tilt_top <= ngtop and
            adjusted_azi_tilt_top <= magtop and
            adjusted_azi_tilt_top <= rgblog_top):
        return False

    # Condition 3: Azimuth and Tilt tables must have sufficient data.
    if not (len(azitable) > 2 and len(tilttable) > 2):
        return False

    # Condition 4: Check for a specific pattern of initial bad readings
    # where the first reading is bad (< 0) and the second is good (> 0).
    if not (azitable[1][1] < 0 and tilttable[1][1] < 0 and
            azitable[2][1] > 0 and tilttable[2][1] > 0):
        return False

    return True


# ------------------------------- MAIN PROCESS -------------------------------

def create_main_gui():
    """
    Creates and runs the main GUI for selecting input files and output directory.
    """
    root = tk.Tk()
    root.title("Televiewer Processing Setup")
    root.geometry("800x250")

    # Bring the window to the front, make it top-most, and center it
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.eval('tk::PlaceWindow . center')

    # --- Styling ---
    style = ttk.Style(root)
    style.configure("Setup.TLabel", font=("Segoe UI", 11))
    style.configure("Setup.TButton", font=("Segoe UI", 11))
    style.configure("Setup.TEntry", font=("Segoe UI", 11))

    # --- Variables to store selections ---
    input_files = []
    output_dir = tk.StringVar()

    # --- Functions for button commands ---
    def select_input_files():
        nonlocal input_files
        files = filedialog.askopenfilenames(
            title="Select WellCAD files to process",
            filetypes=(("WellCAD files", "*.wcl *.WCL"), ("All files", "*.*"))
        )
        if files:
            input_files = root.tk.splitlist(files)
            input_entry.config(state='normal')
            input_entry.delete(0, tk.END)
            input_entry.insert(0, f"{len(input_files)} file(s) selected")
            input_entry.config(state='readonly')

    def select_output_dir():
        directory = filedialog.askdirectory(title="Select Output Folder")
        if directory:
            output_dir.set(directory)

    def start_processing():
        if not input_files:
            messagebox.showerror("Error", "No input files selected.")
            return
        if not output_dir.get():
            messagebox.showerror("Error", "No output directory selected.")
            return
        root.destroy() # Close GUI and proceed to main logic

    # --- GUI Layout ---
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Input files selection
    input_label = ttk.Label(main_frame, text="1. Select Input Files:", style="Setup.TLabel")
    input_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

    input_entry = ttk.Entry(main_frame, width=80, style="Setup.TEntry", state='readonly')
    input_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

    input_button = ttk.Button(main_frame, text="...", command=select_input_files, style="Setup.TButton", width=4)
    input_button.grid(row=1, column=1, sticky=tk.W)
    main_frame.columnconfigure(0, weight=1)

    # Output directory selection
    output_label = ttk.Label(main_frame, text="2. Select Output Folder:", style="Setup.TLabel")
    output_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(15, 5))

    output_entry = ttk.Entry(main_frame, textvariable=output_dir, width=80, style="Setup.TEntry", state='readonly')
    output_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=(0, 10))

    output_button = ttk.Button(main_frame, text="...", command=select_output_dir, style="Setup.TButton", width=4)
    output_button.grid(row=3, column=1, sticky=tk.W)

    # Process button
    process_button = ttk.Button(main_frame, text="3. PROCESS", command=start_processing, style="Setup.TButton")
    process_button.grid(row=4, column=0, columnspan=2, pady=(20, 10), ipady=5, sticky=(tk.W, tk.E))

    root.mainloop()

    return input_files, output_dir.get()


def print_summary(
    processed_files, 
    rejected_files, 
    missing_tilt_accepted, 
    missing_tilt_rejected, 
    missing_caliper_accepted, 
    missing_caliper_rejected
):
    """
    Prints a detailed summary of the processing run with redundant categories.
    """
    processed_basenames = [os.path.basename(f) for f in processed_files]

    # Category 1: ACCEPTED files (all processed files that were not rejected)
    accepted_files = sorted([f for f in processed_basenames if f not in rejected_files])

    # Category 2: REJECTED files
    rejected_files_list = sorted(rejected_files)

    # Category 3: MISSING TILT files
    missing_tilt_files = sorted(list(set(missing_tilt_accepted + missing_tilt_rejected)))

    # Category 4: MISSING CALIPER files
    missing_caliper_files = sorted(list(set(missing_caliper_accepted + missing_caliper_rejected)))

    print("\n##################################################################")
    print("##### -------------- COMPLETE. SUMMARY : ------------------ #####")
    print("##### -------------- Timestamp: " + str(datetime.now().strftime('%d-%m-%Y %H:%M')) + " ---------- #####\n")

    print(f"Total files processed: {len(processed_files)}\n")

    if accepted_files:
        print("ACCEPTED:")
        for f in accepted_files:
            print(f"- {f}")
        print()

    if rejected_files_list:
        print("REJECTED:")
        for f in rejected_files_list:
            print(f"  - {f}")
        print()
    
    if missing_tilt_files:
        print("MISSING TILT:")
        for f in missing_tilt_files:
            print(f"- {f}")
        print()

    if missing_caliper_files:
        print("MISSING CALIPER:")
        for f in missing_caliper_files:
            print(f"- {f}")
        print()

    print("################################################################")


def process_borehole_file(tvfile, app, output_dir):
    """
    Opens, processes, and saves a single WellCAD borehole file.

    This function contains the core logic for cleaning, standardizing, and
    filtering logs within a single file, including the interactive GUI step.

    Args:
        tvfile (str): The full path to the .wcl file to process.
        app (wellcad.com.Application): The active WellCAD application instance.
        output_dir (str): The directory to save the output file.

    Returns:
        tuple: A tuple containing:
            - str or None: The path of the saved output file if successful.
            - set: A set of strings indicating any missing logs ('tilt', 'caliper').
            - bool: A flag indicating if the file was rejected by the user.
    """
    filename = os.path.basename(tvfile)
    holeid = filename.split('_')[-3]
    print("\n--> opening WELLCAD file:", filename, "...")

    try:
        borehole = app.open_borehole(path=tvfile)
    except Exception:
        print(holeid + f"{Fore.RED}{Style.BRIGHT} cannot be opened (download issue?).{Style.RESET_ALL}")
        return None, set(), False

    lognames, logdict = boreholeextract(borehole)
    header = borehole.header
    comment = header.get_item_text('Processor Comments:')
    if not comment:
        header.set_item_text('Processor Comments:', 'Automation:')
        comment = 'Automation:'
    else:
        header.set_item_text('Processor Comments:', comment + ' Automation:')
        comment += ' Automation:'
    holename = header.get_item_text('WELLNAME') or header.get_item_text('WELL')
    if holename != holeid:
        print(f"{Fore.RED}{Style.BRIGHT}Filename / WELLNAME mismatch:{Style.RESET_ALL} {holeid}")
        app.close_borehole(prompt_for_saving=False)
        return None, set(), False

    del_list = ['AZI','TILT','L_GAMMA','SANG','SANGB','L_GAMMAR','L_GSANGB','L_GSANG',
                'DEN(SS)','DEN(SS)#1','DEN(LS)','DEN(LS)#1','DEN(CDL)','COMP']
    if '_nsg' in filename:
        del_list += ['L_SANGB','L_SANG']
    for d in [d for d in del_list if d in lognames]:
        try:
            borehole.remove_log(d)
        except Exception:
            pass

    tilt_log_name = ''
    tilt_cond = False
    try:
        needed = ['L_SANGB','L_NSGSANGB','AZIMUTH']
        if not any(n in lognames for n in needed):
            print(f"{filename}{Fore.RED}{Style.BRIGHT} no acceptable azimuth log - stop.{Style.RESET_ALL}")
            app.close_borehole(prompt_for_saving=False)
            return None, set(), False
        if '_m.' in filename:
            if 'L_SANGB' in lognames:
                azilog, azitable, azidf_df, azitop, _ = log_obtain(holeid, borehole, logdict, 'L_SANGB')
                tilt_log_name = 'L_SANG'
            else:
                azilog, azitable, azidf_df, azitop, _ = log_obtain(holeid, borehole, logdict, 'AZIMUTH')
                tilt_log_name = 'TILT'
        else:
            azilog, azitable, azidf_df, azitop, _ = log_obtain(holeid, borehole, logdict, 'L_NSGSANGB')
            tilt_log_name = 'L_NSGSANG'
        
        tiltlog, tilttable, tiltdf, tilttop, tilt_cond = log_obtain(holeid, borehole, logdict, tilt_log_name)
        rgblog, rgbtable, rgbdf, rgbtop, _ = log_obtain(holeid, borehole, logdict, 'RGB')
    except Exception as e:
        print("Log acquisition error:", e)
        app.close_borehole(prompt_for_saving=False)
        return None, set(), False

    nglog, ngtable, ngdf, ngtop, ngcond = log_obtain(holeid, borehole, logdict, 'NG')
    callog, caltable, caldf, caltop, calcond = log_obtain(holeid, borehole, logdict, 'CAL')
    maglog, magtable, magdf, magtop, magcond = log_obtain(holeid, borehole, logdict, 'MAG')
    amplog, amptable, ampdf, amptop, ampcond = log_obtain(holeid, borehole, logdict, 'AMP')

    missing_logs = set()
    if tilt_cond:
        missing_logs.add('tilt')
    if calcond:
        missing_logs.add('caliper')

    holedepthchecker(holeid, header, rgblog, amplog)

    viscond = False
    log_data_for_check = {
        'calcond': calcond, 'magcond': magcond,
        'azitop': azitop, 'tilttop': tilttop,
        'caltop': caltop, 'ngtop': ngtop, 'magtop': magtop,
        'rgblog_top': rgblog.top_depth if rgblog else 0,
        'azitable': azitable, 'tilttable': tilttable
    }
    if not tilt_cond and should_adjust_visible_depth(log_data_for_check):
        del azitable[1]; del tilttable[1]
        azilog.data_table = tuple(azitable); tiltlog.data_table = tuple(tilttable)
        borehole.set_visible_depth_range(top_depth=azitop + 0.1,
                                         bottom_depth=round(borehole.bottom_depth, 1))
        viscond = True

    top_depth = azitop + 0.1 if viscond else round(borehole.top_depth, 1)
    if rgblog and rgblog.bottom_depth != rgblog.top_depth and rgbtop < top_depth:
        top_depth = rgbtop
    if nglog and nglog.bottom_depth != nglog.top_depth and ngtop < top_depth:
        top_depth = ngtop
    if amplog and amplog.bottom_depth != amplog.top_depth and amptop < top_depth:
        top_depth = amptop

    caltruetop = None
    if not calcond and caltop == top_depth and azitop != top_depth:
        for (d, v) in caltable[1:]:
            if v > 0 and caltruetop is None:
                caltruetop = round(d, 1)
        cand = [azitop, tilttop, caltruetop, magtop, rgbtop, ngtop]
        cand = [c for c in cand if c is not None]
        if cand:
            top_depth = min(cand)
        if caltruetop and round(caltop, 1) != round(caltruetop, 1):
            while caltop < caltruetop:
                del caltable[1]; caltop = caltable[1][0]

    azibottom = round(azilog.bottom_depth, 1)
    bottom_depth = round(borehole.bottom_depth, 1)
    if rgblog:
        rgbbottom = math.ceil(rgblog.bottom_depth * 10) / 10
        if bottom_depth < rgbbottom:
            bottom_depth = rgbbottom
        if rgblog.top_depth < 0:
            print(filename + f"{Fore.YELLOW}{Style.BRIGHT} imaging above 0m.{Style.RESET_ALL}")
    if top_depth < 0:
        borehole.set_visible_depth_range(top_depth=0, bottom_depth=bottom_depth)
        top_depth = 0

    nlogs = borehole.nb_of_logs
    if nlogs < 11:
        print("### WARNING: Possibly missing logs:", filename)
    elif nlogs > 11:
        print("### WARNING: Extra logs present:", filename)

    azitable, azilog, com1a, com1b, com1c, azicomment, fv1, lv1 = log_editor(
        azitable, azilog, holeid, viscond, top_depth, bottom_depth, header, 'Azimuth')
    if not tilt_cond:
        tilttable, tiltlog, com2a, com2b, com2c, tiltcomment, fv2, lv2 = log_editor(
            tilttable, tiltlog, holeid, viscond, top_depth, bottom_depth, header, 'Tilt')
    else: # Provide default values if tilt log is missing
        com2a = com2b = com2c = False
        tiltcomment = ""
        fv2 = lv2 = None

    try:
        cal_editor(holeid, calcond, caltable, callog, caldf, caltop, top_depth, bottom_depth, header)
    except Exception:
        pass
    if nglog:
        try:
            nglog.scale_low = 0; nglog.scale_high = 200; nglog.filter = 3
        except Exception:
            pass
    if maglog:
        try:
            maglog.scale_low = 0; maglog.scale_high = 2
        except Exception:
            pass

    azimuth_rejected = False
    if '_hs_m' in filename:
        comment2 = header.get_item_text('Processor Comments:')
        azidf_work = pd.DataFrame(azitable[1:], columns=["Depth (m)", "AZIMUTH"])
        azidf2, no_rotation = azifilter(holeid, header, azidf_work, tiltdf,
                                        azitop, azibottom, top_depth, bottom_depth,
                                        15, comment2)

        if not no_rotation:
            depth_arr = azidf_work['Depth (m)'].values
            raw_azi = azidf_work['AZIMUTH'].values

            class GraphApp:
                """A Tkinter GUI application for interactively filtering azimuth data."""
                def __init__(self, root, tilt_data_available):
                    """
                    Initializes the GraphApp GUI by setting up plots, widgets, and event bindings.
                    """
                    self.root = root
                    self.tilt_data_available = tilt_data_available
                    self._initialize_variables_and_data()
                    self._setup_styles()
                    self._setup_plots()
                    self._layout_widgets()
                    self._bind_events()
                    self.update_plot()

                def _initialize_variables_and_data(self):
                    """Initializes Tkinter variables and prepares initial data for plotting."""
                    self.rejected = False
                    self.root.title("Azi Filter)")
                    self.min_samples = 15
                    self.filter_mode = tk.StringVar(value='dbscan')
                    self.alpha_window_m = tk.IntVar(value=2)
                    self.alpha_trim = tk.DoubleVar(value=0.50)
                    self.alpha_passes = tk.IntVar(value=2)
                    self.lowess_frac_pct = tk.StringVar(value="3")
                    self.lowess_passes = tk.StringVar(value="1")
                    self.dbscan_min_samples = tk.StringVar(value=str(self.min_samples))
                    self.dbscan_epsilon = tk.DoubleVar(value=3.0)
                    self.post_dbscan_alpha = tk.BooleanVar(value=False)

                    self.azidf2 = azidf2.copy()
                    self.azidf2_smoothed = None
                    self.last_valid_dbscan = self.azidf2.copy()
                    self.tilt_median = self._compute_tilt_median(tiltdf) if self.tilt_data_available else None
                    self.depth = depth_arr
                    self.raw_azi = raw_azi
                    self.holeid = holeid

                    self.max_dbscan_min_samples = self._compute_max_min_samples()
                    if self.max_dbscan_min_samples < 3:
                        self.max_dbscan_min_samples = 3
                    if self.min_samples > self.max_dbscan_min_samples:
                        self.min_samples = self.max_dbscan_min_samples
                        self.dbscan_min_samples.set(str(self.min_samples))

                    if self.tilt_data_available:
                        self.tilt_depth = [row[0] for row in tilttable[1:]]
                        self.tilt_values = [row[1] for row in tilttable[1:]]
                        self.valid_tilt_vals = [v for v in self.tilt_values if v > -900]
                        if not self.valid_tilt_vals:
                            self.valid_tilt_vals = [0.0]
                        self.tilt_median = float(np.median(self.valid_tilt_vals))
                    else:
                        self.tilt_depth = []
                        self.tilt_values = []
                        self.valid_tilt_vals = []
                        self.tilt_median = None


                def _setup_styles(self):
                    """Configures the ttk styles for the application widgets."""
                    self.style = ttk.Style()
                    self.style.configure("App.TLabel", font=("Segoe UI", 11))
                    self.style.configure("App.TRadiobutton", font=("Segoe UI", 11))
                    self.style.configure("App.TSpinbox", font=("Segoe UI", 11))
                    self.style.configure("App.TButton", font=("Segoe UI", 11))
                    self.style.configure("App.TCheckbutton", font=("Segoe UI", 11))

                def _setup_plots(self):
                    """Creates the matplotlib figure, axes, and initial plot lines."""
                    self.fig = Figure(figsize=(8, 4.5), dpi=100) # Adjusted figure size
                    self.ax = self.fig.add_subplot(1, 1, 1)
                    self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12)

                    self.line_raw, = self.ax.plot(self.depth, self.raw_azi, label='Raw AZI', color='royalblue', linewidth=1)
                    self.line_dbscan, = self.ax.plot(self.azidf2['Depth (m)'], self.azidf2['AZIMUTH'],
                                                     label='DBSCAN', color='red', linewidth=1.2)
                    self.line_dbscan_alpha, = self.ax.plot([], [], label='DBSCAN + AlphaTrim',
                                                           color='green', linestyle='--', linewidth=2, visible=False)
                    self.line_alpha, = self.ax.plot(self.depth,
                                                    self._compute_alpha_trim(self.depth, self.raw_azi),
                                                    label='Alpha-Trim (Direct)', color='red', visible=False)
                    self.line_lowess, = self.ax.plot(self.depth,
                                                     self._compute_lowess_curve(),
                                                     label='LOWESS', color='red', visible=False)

                    self.ax.set_ylim(0, 360)
                    self.ax.set_ylabel("Azimuth (deg)", fontsize=12)
                    self.ax.set_xlabel("Depth (m)", fontsize=12)
                    self.ax.tick_params(axis='both', labelsize=11)

                    self.ax.set_title(f"{holeid}\nDBSCAN Min Samples = {self.min_samples}")
                    self.ax.legend(loc='upper right', fontsize=10)

                def _layout_widgets(self):
                    """Creates and packs all the Tkinter widgets for the GUI."""
                    self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
                    self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

                    # Algorithm selection frame
                    self.algoframe = ttk.Frame(self.root)
                    self.algoframe.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
                    ttk.Label(self.algoframe, text="Algorithm:", style="App.TLabel").pack(side=tk.LEFT, padx=(8, 4))
                    ttk.Radiobutton(self.algoframe, text="DBSCAN (Recommended default)    ",
                                    variable=self.filter_mode, value='dbscan',
                                    command=self.update_plot, style="App.TRadiobutton").pack(side=tk.LEFT, padx=4)
                    ttk.Radiobutton(self.algoframe, text="LOWESS     ",
                                    variable=self.filter_mode, value='lowess',
                                    command=self.update_plot, style="App.TRadiobutton").pack(side=tk.LEFT, padx=4)
                    ttk.Radiobutton(self.algoframe, text="Alpha-Trim     ",
                                    variable=self.filter_mode, value='alpha',
                                    command=self.update_plot, style="App.TRadiobutton").pack(side=tk.LEFT, padx=4)

                    # Parameters master frame
                    self.paramframe = ttk.Frame(self.root, height=170)
                    self.paramframe.pack_propagate(False)
                    self.paramframe.pack(side=tk.TOP, fill=tk.X, pady=(2, 4))

                    self._layout_dbscan_params()
                    self._layout_alpha_params()
                    self._layout_lowess_params()
                    self._layout_action_buttons()

                def _layout_dbscan_params(self):
                    """Lays out widgets specific to the DBSCAN algorithm using the grid manager."""
                    self.dbscan_params = ttk.Frame(self.paramframe)
                    self.dbscan_params.columnconfigure(4, weight=1)

                    # --- Row 0: Min Samples and Epsilon Controls ---
                    ttk.Label(self.dbscan_params, text="Min Samples (bigger = smoother):", style="App.TLabel").grid(row=0, column=0, sticky='w', padx=(4, 4), pady=10)
                    
                    slider_container = ttk.Frame(self.dbscan_params)
                    slider_container.grid(row=0, column=1, sticky='w')
                    
                    self.dbscan_scale = tk.Scale(slider_container, from_=3, to=self.max_dbscan_min_samples,
                                                 orient=tk.HORIZONTAL, variable=self.dbscan_min_samples,
                                                 showvalue=False, length=160, resolution=1)
                    self.dbscan_scale.pack(side=tk.LEFT, padx=(0, 8))
                    
                    self.dbscan_value_label = ttk.Label(slider_container, text=str(self.dbscan_min_samples.get()), style="App.TLabel")
                    self.dbscan_value_label.pack(side=tk.LEFT, padx=(2, 24))

                    ttk.Label(self.dbscan_params, text="Epsilon (smaller = smoother. Try not to change):", style="App.TLabel").grid(row=0, column=2, sticky='w', padx=(4, 2), pady=10)
                    
                    self.epsilon_spin = ttk.Spinbox(self.dbscan_params, from_=1, to=20.0, increment=1, textvariable=self.dbscan_epsilon, width=5, format="%.1f", command=self._update_dbscan, font=Font(family='Segoe UI', size=11))
                    self.epsilon_spin.grid(row=0, column=3, sticky='w', padx=(0, 15))


                    # --- Row 1: Alpha Trim Checkbox and Parameters ---
                    self.dbscan_alpha_row = ttk.Frame(self.dbscan_params)
                    self.dbscan_alpha_row.grid(row=1, column=0, columnspan=5, sticky='w', pady=(8, 0))

                    self.dbscan_alpha_chk = ttk.Checkbutton(
                        self.dbscan_alpha_row,
                        text="Post-proc Alpha Trim (residual smooth)",
                        variable=self.post_dbscan_alpha,
                        command=self._toggle_post_dbscan_alpha,
                        style="App.TCheckbutton"
                    )
                    self.dbscan_alpha_chk.pack(side=tk.LEFT, padx=(0, 10))

                    self.post_dbscan_alpha_params = ttk.Frame(self.dbscan_alpha_row)
                    
                    ttk.Label(self.post_dbscan_alpha_params, text="Alpha (bigger = preserve edges):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.post_dbscan_alpha_spin = ttk.Spinbox(self.post_dbscan_alpha_params, from_=0.0, to=1.0, increment=0.1, textvariable=self.alpha_trim, width=5, format="%.2f", command=self._update_post_dbscan_alpha, font=Font(family='Segoe UI', size=11))
                    self.post_dbscan_alpha_spin.pack(side=tk.LEFT, padx=(4, 15))
                    
                    ttk.Label(self.post_dbscan_alpha_params, text="Window (m) (bigger = smoother):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.post_dbscan_window_spin = ttk.Spinbox(self.post_dbscan_alpha_params, from_=1, to=50, increment=1, textvariable=self.alpha_window_m, width=5, command=self._update_post_dbscan_alpha, font=Font(family='Segoe UI', size=11))
                    self.post_dbscan_window_spin.pack(side=tk.LEFT, padx=(4, 15))

                    ttk.Label(self.post_dbscan_alpha_params, text="Passes (bigger = smoother):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.post_dbscan_passes_spin = ttk.Spinbox(self.post_dbscan_alpha_params, from_=1, to=6, increment=1, textvariable=self.alpha_passes, width=3, command=self._update_post_dbscan_alpha, font=Font(family='Segoe UI', size=11))
                    self.post_dbscan_passes_spin.pack(side=tk.LEFT, padx=(4, 15))

                    # --- Row 2: Message ---
                    self.dbscan_message = ttk.Label(self.dbscan_params, text=("Avoid using large # samples if it alters background trend.\n" "Recommend : Start small, increase Min Samples until big mag spikes removed.\n" "Then smooth residual jitter using post-proc alpha trim, if needed."), style="App.TLabel", wraplength=1000, foreground="#a73f00")
                    self.dbscan_message.grid(row=2, column=0, columnspan=5, sticky='w', pady=(8,8))

                def _layout_alpha_params(self):
                    """Lays out widgets specific to the Alpha-Trim algorithm."""
                    self.alpha_params = ttk.Frame(self.paramframe)
                    ttk.Label(self.alpha_params, text="Alpha (bigger = preserve edges):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.alpha_spin = ttk.Spinbox(self.alpha_params, from_=0.0, to=1.0, increment=0.1, textvariable=self.alpha_trim, width=5, format="%.2f", command=self._update_alpha_curve, font=Font(family='Segoe UI', size=11))
                    self.alpha_spin.pack(side=tk.LEFT, padx=2)
                    ttk.Label(self.alpha_params, text="Window (m) (bigger = smoother):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.window_spin = ttk.Spinbox(self.alpha_params, from_=1, to=50, increment=1, textvariable=self.alpha_window_m, width=5, command=self._update_alpha_curve, font=Font(family='Segoe UI', size=11))
                    self.window_spin.pack(side=tk.LEFT, padx=2)
                    ttk.Label(self.alpha_params, text="Passes (bigger = smoother):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.passes_spin = ttk.Spinbox(self.alpha_params, from_=1, to=6, increment=1, textvariable=self.alpha_passes, width=3, command=self._update_alpha_curve, font=Font(family='Segoe UI', size=11))
                    self.passes_spin.pack(side=tk.LEFT, padx=2)

                def _layout_lowess_params(self):
                    """Lays out widgets specific to the LOWESS algorithm."""
                    self.lowess_params = ttk.Frame(self.paramframe)
                    ttk.Label(self.lowess_params, text="Frac (bigger = smoother):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.lowess_spin = ttk.Spinbox(self.lowess_params, from_=1, to=100, increment=1, textvariable=self.lowess_frac_pct, width=4, command=self._update_lowess_curve, font=Font(family='Segoe UI', size=11))
                    self.lowess_spin.pack(side=tk.LEFT, padx=(0, 2))
                    ttk.Label(self.lowess_params, text="Passes (bigger = smoother):", style="App.TLabel").pack(side=tk.LEFT, padx=2)
                    self.lowess_passes_spin = ttk.Spinbox(self.lowess_params, from_=1, to=10, increment=1, textvariable=self.lowess_passes, width=4, command=self._update_lowess_curve, font=Font(family='Segoe UI', size=11))
                    self.lowess_passes_spin.pack(side=tk.LEFT, padx=2)

                def _layout_action_buttons(self):
                    """Lays out the Accept and Reject buttons."""
                    self.actionframe = ttk.Frame(self.root)
                    self.actionframe.pack(side=tk.TOP, fill=tk.X, pady=(6, 8))
                    ttk.Button(self.actionframe, text="ACCEPT - QC result in Wellcad", command=self.accept, style="App.TButton").pack(side=tk.LEFT, fill="x", expand=1, padx=(16, 8))
                    ttk.Button(self.actionframe, text="REJECT - Manually smooth + rotate in Wellcad", command=self.reject, style="App.TButton").pack(side=tk.LEFT, fill="x", expand=1, padx=(8, 16))

                def _bind_events(self):
                    """Binds variable traces to their callback functions."""
                    def update_dbscan_value_label(value):
                        self.dbscan_value_label.config(text=str(value))
                    self.dbscan_scale.config(command=update_dbscan_value_label)

                    def show_dbscan_message(*_):
                        if self.filter_mode.get() == 'dbscan':
                            self.dbscan_message.grid(row=2, column=0, columnspan=5, sticky='w', padx=4, pady=(4,0))
                        else:
                            self.dbscan_message.grid_forget()
                    self.filter_mode.trace('w', show_dbscan_message)
                    
                    self.dbscan_min_samples.trace('w', lambda *_: self._update_dbscan())
                    self.dbscan_epsilon.trace('w', lambda *_: self._update_dbscan())
                    self.alpha_trim.trace('w', lambda *_: self._alpha_param_changed())
                    self.alpha_window_m.trace('w', lambda *_: self._alpha_param_changed())
                    self.alpha_passes.trace('w', lambda *_: self._alpha_param_changed())
                    self.lowess_frac_pct.trace('w', lambda *_: self._update_lowess_curve())
                    self.lowess_passes.trace('w', lambda *_: self._update_lowess_curve())

                # ---- Internal methods (logic unchanged except packing of alpha params) ----
                def _compute_max_min_samples(self):
                    """Calculates the maximum viable `min_samples` for DBSCAN.

                    It performs a binary search to find the largest `min_samples` value
                    that still results in at least one cluster being formed.

                    Returns:
                        int: The maximum `min_samples` value, or 3 if none is found.
                    """
                    depth = self.depth
                    azi = self.raw_azi
                    if depth is None or len(depth) == 0:
                        return 3
                    data = np.column_stack((depth, azi))
                    low, high = 3, len(data)
                    best = None

                    def has_cluster(ms):
                        try:
                            db = DBSCAN(eps=self.dbscan_epsilon.get(), min_samples=ms).fit(data)
                            labels = db.labels_
                            nc = len(set(labels)) - (1 if -1 in labels else 0)
                            return nc > 0
                        except Exception:
                            return False

                    if not has_cluster(low):
                        return low
                    while low <= high:
                        mid = (low + high) // 2
                        if has_cluster(mid):
                            best = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                    return best if best is not None else 3

                def _compute_tilt_median(self, tiltdf):
                    """Computes the median value from a tilt DataFrame.

                    Args:
                        tiltdf (pd.DataFrame): DataFrame containing tilt data.

                    Returns:
                        float or None: The median tilt, or None if it cannot be computed.
                    """
                    if tiltdf is None or tiltdf.empty:
                        return None
                    cands = [c for c in tiltdf.columns if any(k in c.upper() for k in ['TILT', 'L_SANG', 'SANG'])]
                    if not cands:
                        return None
                    s = pd.to_numeric(tiltdf[cands[0]], errors='coerce')
                    s = s[s >= 0]
                    return float(s.median()) if not s.empty else None

                def _compute_alpha_trim(self, depth, azi):
                    """Computes the alpha-trimmed mean for the given data.

                    Args:
                        depth (np.ndarray): Array of depth values.
                        azi (np.ndarray): Array of azimuth values.

                    Returns:
                        np.ndarray: The smoothed azimuth data.
                    """
                    try:
                        window_m = self.alpha_window_m.get()
                        alpha = self.alpha_trim.get()
                        passes = self.alpha_passes.get()
                    except (tk.TclError, ValueError):
                        # Handle cases where the spinbox is empty during user input
                        return azi 

                    return alpha_trim_filter(depth, azi, window_m, alpha, passes)

                def _compute_lowess_curve(self):
                    """Computes the LOWESS smoothed curve for the raw azimuth data.

                    Returns:
                        np.ndarray: The smoothed azimuth data.
                    """
                    x = self.depth
                    y = self.raw_azi
                    y_wrap = np.unwrap(np.deg2rad(y))
                    try:
                        frac = int(self.lowess_frac_pct.get()) / 100.0
                    except ValueError:
                        frac = 0.03
                    frac = max(0.01, min(1.0, frac))
                    try:
                        passes = int(self.lowess_passes.get())
                    except ValueError:
                        passes = 1
                    ord_idx = np.argsort(x)
                    x_sorted = x[ord_idx]
                    y_sorted = y_wrap[ord_idx].copy()
                    for _ in range(max(1, passes)):
                        y_sorted = lowess(y_sorted, x_sorted, frac=frac, return_sorted=False)
                    y_deg_sorted = np.rad2deg(y_sorted)
                    y_deg = np.empty_like(y_deg_sorted)
                    y_deg[ord_idx] = y_deg_sorted
                    return np.mod(y_deg, 360)

                def _alpha_param_changed(self):
                    """Callback for when any alpha-trim parameter changes."""
                    mode = self.filter_mode.get()
                    if mode == 'alpha':
                        self._update_alpha_curve()
                    elif mode == 'dbscan' and self.post_dbscan_alpha.get():
                        self._apply_post_dbscan_alpha()

                def _update_dbscan(self):
                    """Callback to re-run DBSCAN filtering when its parameters change."""
                    if self.filter_mode.get() != 'dbscan':
                        return
                    raw_min_samples = self.dbscan_min_samples.get().strip()
                    if not raw_min_samples.isdigit():
                        return
                    
                    try:
                        new_eps = self.dbscan_epsilon.get()
                    except (tk.TclError, ValueError):
                        return # Invalid float in spinbox

                    new_min_samples = int(raw_min_samples)
                    if new_min_samples > self.max_dbscan_min_samples:
                        new_min_samples = self.max_dbscan_min_samples
                        self.dbscan_min_samples.set(str(new_min_samples))
                    new_min_samples = max(3, new_min_samples)
                    
                    df_new, no_rot = azifilter(self.holeid, header,
                                               pd.DataFrame({'Depth (m)': self.depth, 'AZIMUTH': self.raw_azi}),
                                               tiltdf, self.depth[0], self.depth[-1],
                                               self.depth[0], self.depth[-1],
                                               new_min_samples,
                                               header.get_item_text('Processor Comments:'),
                                               eps=new_eps)
                    if no_rot:
                        self.dbscan_min_samples.set(str(self.min_samples))
                        warn = "WARN : No coherent azi trend. Min Samples too large."
                        if self.tilt_median is not None and self.tilt_median <= 5:
                            warn += " Median tilt < 5 deg."
                        self.ax.set_title(f"{self.holeid}\n{warn}", fontsize=10)
                    else:
                        self.min_samples = new_min_samples
                        self.azidf2 = df_new.copy()
                        self.last_valid_dbscan = self.azidf2.copy()
                        self.line_dbscan.set_xdata(self.azidf2['Depth (m)'])
                        self.line_dbscan.set_ydata(self.azidf2['AZIMUTH'])
                        self.azidf2_smoothed = None
                        self.line_dbscan_alpha.set_visible(False)
                        if self.post_dbscan_alpha.get():
                            self._apply_post_dbscan_alpha()
                        self.ax.set_title(f"{self.holeid}\nDBSCAN Min Samples = {self.min_samples}")
                    self._refresh_legend()
                    self.canvas.draw()

                def _apply_post_dbscan_alpha(self):
                    """Applies the alpha-trim filter as a post-processing step to the DBSCAN result."""
                    if self.azidf2 is None or self.azidf2.empty:
                        return
                    depth_db = self.azidf2['Depth (m)'].values
                    azi_db = self.azidf2['AZIMUTH'].values
                    smoothed = self._compute_alpha_trim(depth_db, azi_db)
                    self.azidf2_smoothed = pd.DataFrame({'Depth (m)': depth_db, 'AZIMUTH': smoothed})
                    self.line_dbscan_alpha.set_xdata(depth_db)
                    self.line_dbscan_alpha.set_ydata(smoothed)
                    self.line_dbscan_alpha.set_visible(True)
                    self._refresh_legend()
                    self.canvas.draw()

                def _update_post_dbscan_alpha(self, *args):
                    """Callback to update the post-DBSCAN alpha trim when its parameters change."""
                    if self.filter_mode.get() == 'dbscan' and self.post_dbscan_alpha.get():
                        self._apply_post_dbscan_alpha()

                def _update_alpha_curve(self):
                    """Callback to update the main alpha-trim curve when its parameters change."""
                    if self.filter_mode.get() != 'alpha':
                        return
                    self.line_alpha.set_ydata(self._compute_alpha_trim(self.depth, self.raw_azi))
                    self.canvas.draw()

                def _update_lowess_curve(self):
                    """Callback to update the LOWESS curve when its parameters change."""
                    if self.filter_mode.get() != 'lowess':
                        return
                    self.line_lowess.set_ydata(self._compute_lowess_curve())
                    self.canvas.draw()

                def _toggle_post_dbscan_alpha(self):
                    """Shows or hides the post-DBSCAN alpha-trim parameters and updates the plot."""
                    if self.filter_mode.get() != 'dbscan':
                        return
                    if self.post_dbscan_alpha.get():
                        # Pack alpha params inline beside checkbox
                        self.post_dbscan_alpha_params.pack(side=tk.LEFT, padx=(8, 0), anchor='s')
                        self._apply_post_dbscan_alpha()
                    else:
                        self.post_dbscan_alpha_params.pack_forget()
                        self.line_dbscan_alpha.set_visible(False)
                        self.azidf2_smoothed = None
                        self._refresh_legend()
                        self.canvas.draw()

                def _refresh_legend(self):
                    """Refreshes the plot legend based on the current filter mode."""
                    mode = self.filter_mode.get()
                    self.line_dbscan.set_visible(mode == 'dbscan')
                    self.line_dbscan_alpha.set_visible(mode == 'dbscan'
                                                       and self.post_dbscan_alpha.get()
                                                       and self.azidf2_smoothed is not None)
                    self.line_alpha.set_visible(mode == 'alpha')
                    self.line_lowess.set_visible(mode == 'lowess')
                    self.ax.legend(loc='upper right', fontsize=10)

                def update_plot(self):
                    """Updates the entire plot view based on the selected filter algorithm."""
                    mode = self.filter_mode.get()
                    for f in [self.dbscan_params, self.alpha_params, self.lowess_params]:
                        f.pack_forget()
                    
                    if mode == 'dbscan':
                        self.dbscan_params.pack(side=tk.LEFT, padx=(10, 4), fill='x', expand=True)
                        if self.post_dbscan_alpha.get():
                            if not self.post_dbscan_alpha_params.winfo_ismapped():
                                self.post_dbscan_alpha_params.pack(side=tk.LEFT, padx=(8, 0), anchor='s')
                        self._update_dbscan()
                    elif mode == 'alpha':
                        self.alpha_params.pack(side=tk.LEFT, padx=(10, 4))
                        self._update_alpha_curve()
                    elif mode == 'lowess':
                        self.lowess_params.pack(side=tk.LEFT, padx=(10, 4))
                        self._update_lowess_curve()
                    
                    self._refresh_legend()
                    self.canvas.draw()
                    
                    if mode == 'dbscan':
                        if not self.dbscan_message.winfo_ismapped():
                           self.dbscan_message.grid(row=2, column=0, columnspan=5, sticky='w', pady=(4,0))
                    else:
                        if self.dbscan_message.winfo_ismapped():
                            self.dbscan_message.grid_forget()

                def _get_current_curve(self):
                    """Gets the data of the currently active and displayed filtered curve.

                    Returns:
                        tuple: A tuple containing the depth and azimuth arrays for the
                               active curve.
                    """
                    mode = self.filter_mode.get()
                    if mode == 'dbscan':
                        if self.post_dbscan_alpha.get() and self.azidf2_smoothed is not None:
                            return (self.azidf2_smoothed['Depth (m)'].values,
                                    self.azidf2_smoothed['AZIMUTH'].values)
                        else:
                            return (self.azidf2['Depth (m)'].values,
                                    self.azidf2['AZIMUTH'].values)
                    elif mode == 'alpha':
                        return (self.depth, self.line_alpha.get_ydata())
                    elif mode == 'lowess':
                        return (self.depth, self.line_lowess.get_ydata())
                    return (self.azidf2['Depth (m)'].values, self.azidf2['AZIMUTH'].values)

                def accept(self):
                    """Accepts the current filter result, applies it, and closes the GUI."""
                    self.rejected = False
                    depth_used, azi_used = self._get_current_curve()
                    master_depth = np.arange(depth_used[0], depth_used[-1] + 0.01, 0.1)
                    interp_azimuth = np.interp(master_depth, depth_used, azi_used)
                    df_out = pd.DataFrame({'Depth (m)': master_depth, 'AZIMUTH': interp_azimuth})
                    azilogtuple = logtotuple(df_out)
                    azilog.data_table = azilogtuple
                    for i in range(1, len(azilogtuple) - 1):
                        cval = round(azilogtuple[i][1], 1)
                        nd = round(azilogtuple[i + 1][0], 1)
                        nval = round(azilogtuple[i + 1][1], 1)
                        if (cval >= 359.95 and 0.5 <= nval <= 355) or (cval <= 0.05 and 5 <= nval <= 359.5):
                            temp = list(azilogtuple)
                            for j, (d, v) in enumerate(temp):
                                if d == nd:
                                    temp[j] = (d, 0.01)
                            azilogtuple = tuple(temp)
                            azilog.data_table = azilogtuple
                        if 5 <= abs(nval - cval) <= 350:
                            temp = list(azilogtuple)
                            for j, (d, v) in enumerate(temp):
                                if d == nd and j > 0:
                                    temp[j] = (d, round(temp[j - 1][1], 1))
                            azilogtuple = tuple(temp)
                            azilog.data_table = azilogtuple
                    
                    rotatelogs = ['RGB', 'IMG', 'AMP', 'TT', 'RGB#1', 'IMG#1', 'AMP#1', 'TT#1']
                    existing = [borehole.get_log(i).name for i in range(borehole.nb_of_logs)]
                    logs_to_rotate = [r for r in rotatelogs if r in existing]
                    
                    successful_rotations = []
                    failed_rotations = []

                    if logs_to_rotate:
                        for r in logs_to_rotate:
                            try:
                                borehole.rotate_image(log=r, prompt_user=False, config="RotateBy='AZIMUTH'")
                                successful_rotations.append(r)
                            except Exception:
                                failed_rotations.append(r)
                        
                        if not failed_rotations:
                            print("Rotated.")
                        else:
                            if successful_rotations:
                                print(f"Successfully rotated: {', '.join(successful_rotations)}")
                            if failed_rotations:
                                print(f"{Fore.RED}{Style.BRIGHT}Failed to rotate: {', '.join(failed_rotations)}{Style.RESET_ALL}")

                    self.root.destroy()

                def reject(self):
                    """Rejects the filter result and closes the GUI, flagging for manual review."""
                    header.set_item_text('Processor Comments:', comment2)
                    print(f"{Fore.RED}{Style.BRIGHT}Rejected Azimuth Filter for: {Style.RESET_ALL}{self.holeid}")
                    print("MANUAL EDIT + ROTATE IN WELLCAD")
                    self.rejected = True
                    self.root.destroy()

            root = tk.Tk()
            root.lift()
            root.attributes('-topmost', True)
            root.after_idle(root.attributes, '-topmost', False)
            root.state('zoomed')
            root.geometry("2000x1300")
            gapp = GraphApp(root, tilt_data_available=not tilt_cond)
            root.mainloop()
            azimuth_rejected = gapp.rejected
            root.quit()

    if '_hs_nsg' in filename:
        for i in range(1, len(azitable) - 1):
            cv = round(azitable[i][1], 1)
            nd = round(azitable[i + 1][0], 1)
            nv = round(azitable[i + 1][1], 1)
            if (cv >= 359.9 and 0.1 <= nv <= 359.5) or (cv <= 0.1 and 0.5 <= nv <= 359.9):
                for j, (d, v) in enumerate(azitable):
                    if d == nd:
                        azitable[j] = (d, 0.01)
                azilog.data_table = tuple(azitable)
        try:
            names_now = [borehole.get_log(i).name for i in range(nlogs)]
            for r in [r for r in ['RGB', 'IMG', 'AMP', 'TT', 'RGB#1', 'IMG#1', 'AMP#1', 'TT#1'] if r in names_now]:
                borehole.rotate_image(log=r, prompt_user=False, config="RotateBy='AZIMUTH'")
        except Exception:
            pass

    if '_nsg.' in filename:
        header.set_item_text('Magnetic Influence on AZI:', 'N/A')

    new_comment = header.get_item_text('Processor Comments:')
    if new_comment.endswith(':'):
        header.set_item_text('Processor Comments:', comment)
    else:
        if all(x is not None for x in [fv1, lv1, fv2, lv2]):
            if com1a and com2a:
                if (round(azitable[1:][fv1][0], 1) == round(tilttable[1:][fv2][0], 1) and
                        azitable[1:][lv1][0] == tilttable[1:][lv2][0]):
                    combined = (f" Azimuth and Tilt extrapolated above "
                                f"{round(tilttable[1:][fv2][0], 1)} m and below "
                                f"{tilttable[1:][lv2][0]} m.")
                    tmp = new_comment.replace(azicomment, "").replace(tiltcomment, combined)
                    header.set_item_text('Processor Comments:', tmp)
            elif com1b and com2b:
                if round(azitable[1:][fv1][0], 1) == round(tilttable[1:][fv2][0], 1):
                    combined = (f" Azimuth and Tilt extrapolated above "
                                f"{round(tilttable[1:][fv2][0], 1)} m.")
                    tmp = new_comment.replace(azicomment, "").replace(tiltcomment, combined)
                    header.set_item_text('Processor Comments:', tmp)
            elif com1c and com2c:
                if azitable[1:][lv1][0] == tilttable[1:][lv2][0]:
                    combined = (f" Azimuth and Tilt extrapolated below "
                                f"{tilttable[1:][lv2][0]} m.")
                    tmp = new_comment.replace(azicomment, "").replace(tiltcomment, combined)
                    header.set_item_text('Processor Comments:', tmp)
    
    # Construct the output filename
    out_filename = os.path.basename(tvfile).replace("_preliminary_", "!preliminary_")
    if '_hs' in filename:
        out_filename = out_filename.replace("_hs", "_tn")
    if azimuth_rejected:
        out_filename = re.sub(r'(!preliminary)', r'NEED_MANUAL_AZI_EDIT_ROTATE_\1', out_filename, 1)
    
    # Join with the selected output directory
    outfile = os.path.join(output_dir, out_filename)

    borehole.save_as(outfile)
    app.close_borehole(prompt_for_saving=False)
    return outfile, missing_logs, azimuth_rejected

def main():
    """Main function to find and process preliminary televiewer files."""
    rejected_azimuth_files = []
    processed_files_paths = []
    missing_tilt_accepted = []
    missing_tilt_rejected = []
    missing_caliper_accepted = []
    missing_caliper_rejected = []
    
    # Use the new GUI to select files and output directory
    tvfiles, output_dir = create_main_gui()
    
    if not tvfiles or not output_dir:
        print("No files or output directory selected. Exiting program.")
        return

    app = wellcad.com.Application()
    tally = 0
    # Sort files to process hs_m files first
    tvfiles = sorted(tvfiles, key=lambda f: "_hs_m" not in os.path.basename(f))
    
    for tvfile in tvfiles:
        filename = os.path.basename(tvfile)
        if "_hs_m" in filename and tally == 0:
            print("---------------\nBeginning hs_m files")
            tally = 1
        elif "_hs_m" not in filename and tally == 1:
            print("Completed hs_m files\n---------------")
            tally = 0

        outfile, missing_logs, was_rejected = process_borehole_file(tvfile, app, output_dir)
        
        if outfile:
            out_basename = os.path.basename(outfile)
            processed_files_paths.append(outfile)
            
            if was_rejected:
                rejected_azimuth_files.append(out_basename)
            
            if 'tilt' in missing_logs:
                if was_rejected:
                    missing_tilt_rejected.append(out_basename)
                else:
                    missing_tilt_accepted.append(out_basename)

            if 'caliper' in missing_logs:
                if was_rejected:
                    missing_caliper_rejected.append(out_basename)
                else:
                    missing_caliper_accepted.append(out_basename)

    print_summary(
        processed_files_paths, 
        rejected_azimuth_files, 
        missing_tilt_accepted, 
        missing_tilt_rejected, 
        missing_caliper_accepted, 
        missing_caliper_rejected
    )

if __name__ == '__main__':
    main()
