#!/usr/bin/env python3

"""

Module contains a list of functions that are used in the HTAP parameterisation

Created on May 8, 2017
@author: sturnock

"""

import numpy as np
import matplotlib
import scipy.io as sp_io
# use the Agg environment to generate an image rather than outputting to screen (FOR USE WHEN SUBMITTING TO SPICE)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset
#import netCDF4

# IDL sav file containing ACCMIP burden to forcing relationship data 
ACCMIP_FILE = '/data/users/sturnock/ACCMIP_data/normalised_o3t_rf_ACCMIP_March2013.sav'
# Netcdf file containing gridded definition of HTAP2 source regions on 1x1 grid
HTAP_REGS = '/data/users/sturnock/HTAP_files/HTAP_Phase2_tier1NC1x1_v2.nc'

#################################################

#------------------------------------------------

def read_model_data_base(fname,cur_scn,spec,srf_name,p_out=False):
    '''
    Read in the HTAP-I model scenario data for baseline values
    '''
    ncid = Dataset(fname,'r')
    #extract data from netcdf file
    if p_out:
        # If using model output then need different variables
        lats = ncid.variables['lat_out'][:] #extract/copy the data
        lons = ncid.variables['lon_out'][:]
        levs = ncid.variables['level_out'][:]
        time = np.arange(1,13,1)#ncid.variables['time'][:]
        srf_o3_data_in = ncid.variables[spec+'_'+srf_name+'_'+cur_scn][:] # ozone data for specific model and scenario
        o3_3d_data_in  = ncid.variables[spec+'_3d_'+cur_scn][:]
        srf_o3_data    = srf_o3_data_in[0,:,:,:]
        o3_3d_data     = o3_3d_data_in[0,:,:,:,:]
    else:
        # If using normal 2010 base then this is fine
        lats = ncid.variables['lats'][:] #extract/copy the data
        lons = ncid.variables['lons'][:]
        levs = ncid.variables['levels'][:]
        time = ncid.variables['time'][:]
        srf_o3_data = ncid.variables[spec+'_'+srf_name+'_'+cur_scn][:] # ozone data for specific model and scenario
        o3_3d_data  = ncid.variables[spec+'_3d_'+cur_scn][:]  # 3d ozone data 
    
    #close the input NetCDF file 
    ncid.close()
    
    return lats,lons,levs,time,srf_o3_data,o3_3d_data

#------------------------------------------------

def read_model_data_base_H1(fname,cur_scn,spec,srf_name):
    '''
    Read in the HTAP-I model scenario data
    '''
    ncid = Dataset(fname,'r')
    #extract data from netcdf file
    lats = ncid.variables['lat'][:] #extract/copy the data
    lons = ncid.variables['lon'][:]
    levs = ncid.variables['lev'][:]
    time = ncid.variables['time'][:]
    srf_o3_data = ncid.variables['srf_o3_SR1'][:] # ozone data for specific model and scenario
    o3_3d_data  = ncid.variables['vmr_o3_SR1'][:]  # 3d ozone data 
    
    #close the input NetCDF file 
    ncid.close()
    
    return lats,lons,levs,time,srf_o3_data,o3_3d_data

#---------------------------------------------------

def read_process_emis_data(emis_file,nyrs,nregs,years_scn):
    '''
    Read in the fractional emission change data for the new scenario
    '''
    
    emis_data_out = np.zeros((nyrs,3,nregs), dtype='f') 
    # Also need CH4 changes
    ch4_fut_out = np.zeros((nyrs), dtype='f')
    
    with open(emis_file) as fp:
        #for line in fp: # read each line
        emis_lines = fp.read().splitlines() # read all lines of file into list and strip /n from end of line and also white space
        for (iln,line) in enumerate(emis_lines): #for each line in file
            for (iyr,year) in enumerate(years_scn):
                #print year
                
                if line.startswith('2010'):
                    ch4_base = float(line[38:45])
                    
                if line.startswith(year):  # if the year in emission file matches the requested scenario then proceed
                    print( 'Find fractional emission change between base year and future year {}'.format(year))
                    # Get CH4 abundance change for this particularly year
                    ch4_fut_out[iyr] = float(line[38:45])
                
                    # Find Region of interest on next 5 lines
                    for j in range(nregs):
                        
                        new_line = emis_lines[iln+j+1]
                        
                        emis_data_out[iyr,0,j] = float(new_line[6:15]) #NOx emissions for region
                        emis_data_out[iyr,1,j] = float(new_line[24:33]) #NMVOC emissions for region
                        emis_data_out[iyr,2,j] = float(new_line[15:24]) #CO emissions for region
                        
    return ch4_base,ch4_fut_out,emis_data_out

#---------------------------------------------------

def calc_emis_scal(emis_fract,ch4_base,ch4_fut,years_all,h1_emis,src_regs,nruns,change_ch4_20,a,b,lin,htap):
    '''
    Function to Calculate the emission scaling factors for each emission species across each region
    '''

    # Initial arrays
    emis_scal_f = np.zeros((len(years_all),nruns), dtype='f')
    emis_scal_g = np.zeros((len(years_all),nruns), dtype='f')
    if htap == 'H2':
        emis_scal_f = np.zeros((len(years_all),nruns,len(src_regs)), dtype='f')
        emis_scal_g = np.zeros((len(years_all),nruns,len(src_regs)), dtype='f')
    
    print( 'No. of Source regions for {} = {}'.format(htap,len(src_regs)))
    # Calculate emission scaling factors for each year
    for (iyr,year) in enumerate(years_all):
        
        # CH4 Emissions scaling factor (ONLY FOR HTAP1 SCENARIOS as HTAP2 are ALL emissions)
        
        ch4_scal_g_fact = calc_ch4_scal(ch4_base,ch4_fut[iyr],change_ch4_20,a,b,lin,htap)
        
        # Put CH4 scaling factors into the First point of the master emission scaling array as no longer using Baseline data
        if htap == 'H1':
            emis_scal_f[iyr,0] = ch4_scal_g_fact# for CH4 both factors should be scaling factor g 
            emis_scal_g[iyr,0] = ch4_scal_g_fact
            
        if htap == 'H2':
            emis_scal_f[iyr,:,0] = ch4_scal_g_fact# for CH4 both factors should be scaling factor g 
            emis_scal_g[iyr,:,0] = ch4_scal_g_fact
            
        # OTHER EMISSION SCALING FACTOR
        k = 1 # first point is for CH4 which is calculated separately above)
        #k = 0 # for HTAP2 Models set to zero, as not baseline or methane scenarios considered
        len_regs = len(src_regs)
        if htap == 'H2': len_regs = len(src_regs)-1
        for (iemis,emis) in enumerate(h1_emis):
            reg_h2 = 1   
            for ireg in range(len_regs):
                
                cur_fract = emis_fract[iyr,iemis,ireg]
                
                # Apart from NOX all other emission susing linear factor
                # After extracting requested fractional emission change need to calculate the change relative to the original 20% decrease
                # (i.e. a new 10% emission decrease (-0.1) is half the response of the original 20% reduction and therefore the ozone response needs scaling factor of +0.5) 
                linear_fact = (-cur_fract) / (0.2) #need minus sign to convert relative to original 20% reduction
                #(species,regs)
   
                # Emission scaling factor curve (g = 0.95f + 0.05f^2)
                curve_factor = ((a * linear_fact) + b) * linear_fact # gives same result as formula above
                #(species,regs)
                
                if htap == 'H1':
                    emis_scal_f[iyr,k] = linear_fact
                    emis_scal_g[iyr,k] = linear_fact
                if htap == 'H2':
                    emis_scal_f[iyr,iemis,reg_h2] = linear_fact
                    emis_scal_g[iyr,iemis,reg_h2] = linear_fact
                
                if emis is 'NOx': # If emissions are NOx then make scaling non-linear
                    if htap == 'H1':
                        emis_scal_g[iyr,k] = curve_factor
                    if htap == 'H2':
                        emis_scal_g[iyr,iemis,reg_h2] = curve_factor
               
                reg_h2 +=1
                k += 1
           
    return emis_scal_f, emis_scal_g

#------------------------------------------------

def calc_ch4_scal(ch4_base,ch4_fut,change_ch4_20,a,b,lin,htap):
    '''
    For each year calculate the linear and non-linear emission scaling factor for Methane
    '''
    
    # change in abundance of CH4 between future and baseline scenario (not base - fut to give sign right)
    ch4_change = ch4_fut - ch4_base 
    # CH4 scaling factor (CH4 change base - fut year / CH4 change base - 20% reduction)
    ch4_scal_lin = -(ch4_change / change_ch4_20)
    
    # ch4_scal_lin = -(change_ch4_20 / ch4_change)
    # calculate scaling factor ( g = 0.95f + 0.05f^2)
    # ch4_scal_g = (a * ch4_scal_lin) + (b * np.power(ch4_scal_lin,2)) 
    ch4_scal_g_half = (a * ch4_scal_lin) + b 
    #to account for lifetime effect of CH4 keep half curvature
    if lin == 1: #if want to account for lifetime effect
        ch4_scal_g_fact = (ch4_scal_g_half + 1.0) / 2.0 
        #ch4_scal_curv_fac = ch4_scal_curv_half * (np.power(ch4_scal_lin,2))
        ch4_scal_g = ch4_scal_g_fact * ch4_scal_g_fact
    else:
        #ch4_scal_curv_fac = (np.power(ch4_scal_lin,2)) * ch4_scal_curv
        ch4_scal_g = ch4_scal_lin * ch4_scal_g_half
         
    return ch4_scal_g

#---------------------------------------------------

def read_pro_ACCMIP_rf(fname,lats_1,lons_1):
    '''
    read in 2D grid from IDL sav file that contains ACCMIP multi-model relationship of burd to RF
    re scale grid to be on 1x1 degrees
    '''
    # read in IDL save file containing arrays
    accmip_idl = sp_io.readsav(ACCMIP_FILE)
    #print accmip_idl.keys()
    accmip_2d_rf = accmip_idl.o3t_rf_norm_w_per_m2_per_du_accmip_mmm # some grid boxes are nan where calculation is dodgy (over Antarctica mainly)
    accmip_lats   = accmip_idl.lat
    accmip_lons   = accmip_idl.lon
    
    print( accmip_2d_rf.shape)
    # ACCMIP lat and lons are same way as param and HTAP2 regions (+90 to -90 and 0 to 360)
    print( 'Now need to convert ACCMIP normalised RF onto same grid at HTAP param output')
    accmip_2d_rf_1x1 = convert_conc_grid_less_ann_ACCMIP(lats_1,lons_1,accmip_2d_rf,accmip_lats,accmip_lons)
    
    return accmip_2d_rf_1x1

#--------------------------------------------------------------------

def convert_conc_grid_less_ann_ACCMIP(lats_1,lons_1,data,lats_mod,lons_mod):
    '''
    To put ACCMIP model output grid onto standard 1x1 grid
    '''
    data_convert = np.zeros((len(lats_1),len(lons_1)))
    lon_sp = lons_mod[2] - lons_mod[1]
    lat_sp = lats_mod[1] - lats_mod[2]
    print( lat_sp,lon_sp)
    
    print( 'convert Annual Mean Data to 1x1 grid')
    # For each lat/lon find where existing grid point fits on 1x1 grid and use as value
    for (ilat,lat_val) in enumerate(lats_mod):
        
        for (ilon,lon_val) in enumerate(lons_mod):
                                
                lon_cond = np.logical_and(lons_1 >= lon_val,lons_1 < lon_val+lon_sp)
                lat_cond = np.logical_and(lats_1 >= lat_val,lats_1 < lat_val+lat_sp)
                
                lon_ind = np.where(lon_cond)
                lat_ind = np.where(lat_cond)
               
                for i in lat_ind[0]:
                    
                    for j in lon_ind[0]:
                        
                        data_convert[i,j] = data[ilat,ilon]
                        # For south pole on 1x1 grid
                        if ilat == len(lats_mod)-1:
                            data_convert[178:180,j] = data[-1,ilon]
                        
    return data_convert

#---------------------------------------------------

def calc_burd_ACCMIP(data_3d,base_3d,dp_arr,area_3d,accmip_rf_2d,nregs_H2,H2_recep_regs):
    '''
    Function to calculate annual mean global total column burden from a 3D input field
    # only calculate column O3 burden where O3 concentrations < 150ppb (definition of tropopause)
    # base_burd (kg) = vmr * (kg / kg * m s-2) * Pa (kg m-1 s-2) * m2
    Assumed based on monthly 3D data input
    Instead of using global 0.042 Wm-2 Du-1 conversation factor from ACCMIP multi-model mean,
    use 2D map of ACCMIP multi-model mean normalised O3 RF responses
    '''
    
    # Constants
    G        = 9.81      # gravity (ms-2)
    AVC      = 6.023e+23 #iris.coords.AuxCoord(6.023e+23,long_name='avagadros constant',units='mol^-1') #avagadro's constant (mol-1)
    MM_O3    = 0.048     # kg/mol O3
    MM_DA    = 0.02897   # kg/mol Dry Air
    DU_CONV  = 2.687e20  # to convert from molecules per m2 into Dobson Units
    
    print( data_3d.shape)
    print( type(data_3d))
    burd = np.zeros(data_3d.shape)
    
    burd[base_3d < 1.5e-07] = data_3d[base_3d < 1.5e-07] * (MM_O3/(MM_DA * G)) * dp_arr[base_3d < 1.5e-07] * area_3d[base_3d < 1.5e-07]
    
    print( 'Calculate Global Total Column Burden')
    # Kg
    burd_2d     = np.sum(burd,axis=1) # sum up burden over all levels
    burd_2d_ann = np.mean(burd_2d,axis=0) # leave as Kg for use in calculation below
    # TG
    burd_glo_sum     = np.sum(burd,axis=(1,2,3)) # Sum up burden over all levels, lats and lons
    burd_glo_sum_ann = np.mean(burd_glo_sum) * 1e-9 # to convert Kg to Tg
    print( 'Global annual mean Total Column burden of O3 = {:.3f} Tg'.format(burd_glo_sum_ann))
    
    # Dobson Units (global total column burden)
    # convert O3 burden from Kg into Dobson units = Kg * ( mol-1 / ( Conv_fac (molecules m-2) * m2 * Kg ))
    glob_area = np.sum(area_3d[0,0,:,:])
    dobs_glo_sum     = burd_glo_sum * (AVC / (DU_CONV * glob_area * MM_O3))
    dobs_glo_sum_ann = np.mean(dobs_glo_sum)
    print( 'Global Total Column burden of O3 = {:.2f} Du'.format(dobs_glo_sum_ann))
    # Calculate 2D field of O3 burden in dobson units
    dobs_2d_ann = burd_2d_ann * (AVC / (DU_CONV * area_3d[0,0,:,:]* MM_O3))
    
    # Convert from Global total column burden of Ozone into a radiative effect using 
    # 2D map of normalised multi-model mean O3 RF from ACCMIP in from Stevenson et al. 2013
    print( 'Convert 2D O3 burdens (Du) into a radiative forcing using the normlised map')
    rf_2d_ann = dobs_2d_ann * accmip_rf_2d
    # find out all the points that are not NaN
    nan_bol = np.isnan(rf_2d_ann)
    val_ind = np.where(nan_bol == False)
    area2d_1x1 = area_3d[0,0,:,:]
    rf_2d_ann_glo = np.average(rf_2d_ann[val_ind],weights=area2d_1x1[val_ind])*1e-3
    print( 'Global O3 Radiative Forcing = {:.5f} W m-2'.format(rf_2d_ann_glo)) # convert from mW m-2 to W m-2
    
    print( 'Calculate HTAP2 Regional O3 burdens and Radiative Forcings')
    ann_rf_base_htap_2 = np.zeros((len(H2_recep_regs)),dtype='f') # % regions plus global as first point
    ann_burd_base_htap_2 = np.zeros((len(H2_recep_regs)),dtype='f')
    lats_1,lons_1,htap_2_region_codes = get_HTAP_2_regions(HTAP_REGS)
    k=0
    for (ireg,reg) in enumerate(sorted(H2_recep_regs)): #Dictionary of items
        cur_reg_code = H2_recep_regs[reg]
        
        ann_burd_base_htap_2[k] = np.sum(burd_2d_ann[htap_2_region_codes == cur_reg_code]) * 1e-9
        ann_rf_base_htap_2[k] = calc_reg_mean_ann(rf_2d_ann, area2d_1x1, htap_2_region_codes, cur_reg_code,val_ind) * 1e-3
        k += 1
        
    return burd_glo_sum_ann, dobs_glo_sum_ann, rf_2d_ann_glo, ann_burd_base_htap_2, ann_rf_base_htap_2

# --------------------------------------

def get_HTAP_2_regions(file_name):
    '''
    Read in the HTAP-II Regional definition file
    '''
    ncid = Dataset(file_name,'r')
    #extract data from netcdf file
    lats = ncid.variables['lat'][:] #extract/copy the data
    lons = ncid.variables['long'][:]
    region_data = ncid.variables['region_code'][:] # ozone data for specific model and scenario
        
    #close the input NetCDF file 
    ncid.close()
    
    return lats,lons,region_data

#------------------------------------------------

def calc_reg_mean_ann(data,area,reg_code_grid,reg_code,valid_dat):
    '''
    Calculate regional mean values based on HTAP-1 regions
    '''
    
    val_data = data[valid_dat]
    area_val = area[valid_dat]
    reg_code_grid_val = reg_code_grid[valid_dat]
    ann_mean_total_wgt = np.nansum(val_data[reg_code_grid_val == reg_code] * area_val[reg_code_grid_val == reg_code])# annual mean total
    ann_mean_total     = ann_mean_total_wgt /  np.sum(area_val[reg_code_grid_val == reg_code]) # weighted to total area of region
    
    return ann_mean_total#, reg_area
    
#-----------------------------------------------------------

def find_reg(run_name):
    '''
    Find the current scenario and region that scaling is applied to
    '''
    # Assign an indicator for particular region
    scen = run_name[0:3]
    if scen == 'SR1': 
        reg_num = 0 #CH4
        print( 'Scenario {} is the Baseline and has no regional information!'.format(scen))
    elif scen == 'SR2': 
        reg_num = 0 #CH4
        reg = 'CH4'
        print( 'For Scenario {} the region of interest is Methane'.format(scen))
    else:
        reg = run_name[3:]
        if reg == 'EU': reg_num = 1
        if reg == 'NA': reg_num = 2
        if reg == 'SA': reg_num = 3
        if reg == 'EA': reg_num = 4
        if reg == 'RW': reg_num = 5
        print( reg_num)
        print( 'For Scenario {} the region of interest is {}'.format(scen,reg))
    return scen,reg_num,reg

#---------------------------------------------------

def calc_area_any(nlats,nlons,lats,lons):
    '''
    Calculate area of grid boxes for an unknown size grid
    '''
    #Calculate area of regions to use in weighting concentrations
    # Calculation for any sized grid box
    # S = R^2 * (Long2 - Long1) * (sin lat1 - sin lat2) - all in radians
    rad = 6371000.0           # radius of earth
    #PI = 3.141592653589793  
    PI180 = np.pi/180.0         # radians
    area2d = np.zeros((nlats,nlons), dtype='f')
    lat_sp = (abs(lats[1] - lats[0]))/2.0
    lon_sp = (abs(lons[1] - lons[0]))

    for ilat in range(nlats):
        yedg=np.sin((lats[ilat]+lat_sp)*PI180)-np.sin((lats[ilat]-lat_sp)*PI180)
        for ilon in range(nlons):
            area2d[ilat,ilon]=rad*rad*PI180*yedg*lon_sp
    
    print( 'Surface area of Earth {:.0f} km2'.format(np.sum(area2d)*1e-6))
        
    return area2d

#--------------------------------------------------------------------

def calc_H2_reg_response(ntime,srf_o3_data_all,area2d,nregs_H2,H2_recep_regs):
    '''
    Calculate the area weighted regional response value (surface) to particular emission scenario
    '''
    dofy = [31,28,31,30,31,30,31,31,30,31,30,31] # number of days in each month
    # Use changes for each region without Baseline concentrations included
    # Get HTAP2 Receptor Regions Mask
    lats_1,lons_1,HTAP_2_region_codes = get_HTAP_2_regions(HTAP_REGS)
        
    pert        = np.zeros((ntime,nregs_H2),dtype='f')
    pert_wght   = np.zeros((ntime,nregs_H2),dtype='f') 
    pert_tot    = np.zeros((nregs_H2),dtype='f') 
    reg_mean_vals = np.zeros((nregs_H2),dtype='f') 
    
    # Calculate areas of each region
    tot_area_regs = np.zeros((nregs_H2), dtype='f')
    for (ireg,reg) in enumerate(sorted(H2_recep_regs)): #Dictionary of items
        cur_reg_code = H2_recep_regs[reg]
        tot_area_regs[ireg] = np.sum(area2d[HTAP_2_region_codes == cur_reg_code])       
    
        # For each Month
        for itime in range(ntime):
            srf_o3_data_mon = srf_o3_data_all[itime,:,:]
            
            # Calculate area weighted total for each region
            pert[itime,ireg] = np.sum((srf_o3_data_mon[HTAP_2_region_codes == cur_reg_code] * area2d[HTAP_2_region_codes == cur_reg_code]))
            # Weight total of all individual grid boxes to total area or region and convert to ppb
            pert_wght[itime,ireg] = (pert[itime,ireg] * 1.0e9) / tot_area_regs[ireg] 
            # weight total to number of days in each month for annual mean values
            pert_tot[ireg] += (pert_wght[itime,ireg] * dofy[itime]) 
        
        # calculate annual mean value of area weighted mean
        reg_mean_vals[ireg] = pert_tot[ireg] / 365.0 # To get annual mean value
        
    return reg_mean_vals

#------------------------------------------------
def calc_glob_mean(surf_data,area):
    '''
    calculate Globally weighted mean value
    '''
    glob_mon_tot = np.zeros(12,dtype='f')
    glob_mon_tot_wgt = np.zeros(12,dtype='f')
    
    for itime in range(12): 
        glob_mon_tot[itime] = np.sum(surf_data[itime,:,:]*area[:,:])
        glob_mon_tot_wgt[itime] = glob_mon_tot[itime] / np.sum(area[:,:])
            
    glob_ann_mean_tot_wgt = np.mean(glob_mon_tot_wgt[:])
    glob_ann_mean_tot_wgt_ppbv = glob_ann_mean_tot_wgt*1e9
    
    return glob_ann_mean_tot_wgt_ppbv

#------------------------------------------------

def output_txt_file_reg_individ_mod(out_fname,header,years_arr,reg_arrs,reg_vals,glob_val,mod_num):
    '''
    Output text file of regional mean concentrations of O3 response for individual models
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        for (ireg_h1,reg_h1) in enumerate(reg_arrs):
            cur_line_dat = []
            for line_dat in reg_vals[mod_num,iyr,ireg_h1,:]:
                cur_line_dat.append(str(line_dat))
                    
            glob_mean_val = str(glob_val[mod_num,iyr,ireg_h1])
                    
            cur_line_str = ','.join(cur_line_dat)     
            df.write(yr+','+reg_h1+','+cur_line_str+','+glob_mean_val+'\n')
    df.close()
    
#------------------------------------------------

def output_txt_file_burd_mmm(out_fname,header,years_arr,reg_arrs,base_burd,base_rf,burd,burd_sd,burd_var,rf,rf_sd,rf_var):
    '''
    Output text file of global mean response of O3 burden for multi-model means
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        for (ireg_h1,reg_h1) in enumerate(reg_arrs):
            cur_line_dat = []
            if iyr == 0: 
                cur_month_dat = [base_burd,0.0,0.0,base_rf,0.0,0.0]
            else:
                cur_month_dat = [burd[iyr-1,ireg_h1],burd_sd[iyr-1,ireg_h1],burd_var[iyr-1,ireg_h1],rf[iyr-1,ireg_h1],rf_sd[iyr-1,ireg_h1],rf_var[iyr-1,ireg_h1]]
                
            for j in cur_month_dat:
                cur_line_dat.append(str(j))
                    
            cur_line_str = ','.join(cur_line_dat)     
            if yr == 0:
                df.write(yr+',BASE,'+cur_line_str+'\n')
            else:
                df.write(yr+','+reg_h1+','+cur_line_str+'\n')
    df.close()

#------------------------------------------------

def output_txt_file_reg_burd_mmm(out_fname,header,val_name,years_arr,reg_arrs,base_burd,base_rf,burd,burd_sd,burd_var,rf,rf_sd,rf_var,
                                glob_burd,glob_rf,exp_glob_burd,exp_glob_burd_sd,exp_glob_burd_var,exp_glob_rf,exp_glob_rf_sd,exp_glob_rf_var):
    '''
    Output text file of regional response of O3 burden for multi-model means
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    blk = np.zeros(16)
    base_list       = [base_burd,blk,blk,base_rf,blk,blk]
    exp_list        = [burd,burd_sd,burd_var,rf,rf_sd,rf_var]
    glob_base_list  = [glob_burd,0.0,0.0,glob_rf,0.0,0.0]
    glob_exp_list   = [exp_glob_burd,exp_glob_burd_sd,exp_glob_burd_var,exp_glob_rf,exp_glob_rf_sd,exp_glob_rf_var]
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        for (ireg_h1,reg_h1) in enumerate(reg_arrs): #emission source region
            for (ivar,var) in enumerate(val_name): # variable name
                cur_base        = base_list[ivar]
                cur_exp         = exp_list[ivar]
                cur_base_glo    = glob_base_list[ivar]
                cur_exp_glo     = glob_exp_list[ivar]
                cur_line_dat = []
                if iyr == 0: 
                    
                    cur_month_dat = cur_base[:]
                    glob_mean_val = str(cur_base_glo)
                else:
                    cur_month_dat   = cur_exp[iyr-1,ireg_h1,:]
                    glob_mean_val  = str(cur_exp_glo[iyr-1,ireg_h1])
                
                for j in cur_month_dat:
                    cur_line_dat.append(str(j))
                    
                cur_line_str = ','.join(cur_line_dat)     
                if yr == 0:
                    df.write(yr+',BASE,'+var+','+cur_line_str+','+glob_mean_val+'\n')
                else:
                    df.write(yr+','+reg_h1+','+var+','+cur_line_str+','+glob_mean_val+'\n')
    df.close()


#------------------------------------------------

def output_txt_file_reg_mmm(out_fname,header,years_arr,reg_arrs,reg_vals,glob_val):
    '''
    Output text file of regional mean concentrations of O3 response for multi-model means
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        for (ireg_h1,reg_h1) in enumerate(reg_arrs):
            cur_line_dat = []
            for line_dat in reg_vals[iyr,ireg_h1,:]:
                cur_line_dat.append(str(line_dat))
                    
            glob_mean_val = str(glob_val[iyr,ireg_h1])
                    
            cur_line_str = ','.join(cur_line_dat)     
            df.write(yr+','+reg_h1+','+cur_line_str+','+glob_mean_val+'\n')
    df.close()

#------------------------------------------------

def output_txt_file_reg_mmm_comb(out_fname,header,years_arr,reg_vals,glob_val):
    '''
    Output text file of regional mean concentrations of O3 response for multi-model means from HTAP1 and HTAP2
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        #for (ireg_h1,reg_h1) in enumerate(reg_arrs):
        cur_line_dat = []
        for line_dat in reg_vals[iyr,:]:
            cur_line_dat.append(str(line_dat))
                    
        glob_mean_val = str(glob_val[iyr])
                    
        cur_line_str = ','.join(cur_line_dat)     
        df.write(yr+','+cur_line_str+','+glob_mean_val+'\n')
    df.close()

#------------------------------------------------

def output_txt_file_burd_mmm_comb(out_fname,header,years_arr,base_burd,base_rf,burd,rf,burd_diff,rf_diff):
    '''
    Output text file of global mean O3 burden response for multi-model means from HTAP1 and HTAP2 combined
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        
            cur_line_dat = []
            if iyr == 0: 
                cur_month_dat = [base_burd,0.0,base_rf,0.0]
            else:
                cur_month_dat = [burd[iyr-1],burd_diff[iyr-1],rf[iyr-1],rf_diff[iyr-1]]
                
            for j in cur_month_dat:
                cur_line_dat.append(str(j))
                    
            cur_line_str = ','.join(cur_line_dat)     
            
            df.write(yr+','+cur_line_str+'\n')
    df.close()
    
#------------------------------------------------

def output_txt_file_reg_burd_mmm_comb(out_fname,header,var_names,years_arr,base_burd,base_rf,base_glob_burd,base_glob_rf,
                                      burd_reg,rf_reg,burd_reg_diff,rf_reg_diff,exp_glob_burd,exp_glob_burd_diff,exp_glob_rf,exp_glob_rf_diff):
    '''
    Output text file of regional mean O3 burden response for multi-model means HTAP1 and HTAP2 combined
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    blk = np.zeros(16)
    base_list       = [base_burd,blk,base_rf,blk]
    exp_list        = [burd_reg,burd_reg_diff,rf_reg,rf_reg_diff]
    glob_base_list  = [base_glob_burd,0.0,base_glob_rf,0.0]
    glob_exp_list   = [exp_glob_burd,exp_glob_burd_diff,exp_glob_rf,exp_glob_rf_diff]
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        for (ivar,var) in enumerate(var_names): # variable name
            cur_base        = base_list[ivar]
            cur_exp         = exp_list[ivar]
            cur_base_glo    = glob_base_list[ivar]
            cur_exp_glo     = glob_exp_list[ivar]
            cur_line_dat    = []
            if iyr == 0: 
                cur_month_dat = cur_base
                glob_mean_val = str(cur_base_glo)
            else:
                cur_month_dat  = cur_exp[iyr-1,:]
                glob_mean_val  = str(cur_exp_glo[iyr-1])
                
            for j in cur_month_dat:
                cur_line_dat.append(str(j))
            
            cur_line_str = ','.join(cur_line_dat)     
            if yr == 0:
                df.write(yr+','+var+','+cur_line_str+','+glob_mean_val+'\n')
            else:
                df.write(yr+','+var+','+cur_line_str+','+glob_mean_val+'\n')
                    
    df.close()

#------------------------------------------------

def plot_reg_changes_all_years_comb(out_fname_pl,yrs_all,reg_means_all_yrs,h2_regions):
    
    #### CHANGE THIS TO ALL HTAP2 REGIONS ####
    
    '''
    Plot up Multi-model mean response to Emission reductions over Different years across
    Additional ALL 16 HTAP2 RECPTOR Regions (EUR,NAM,EAS,SAS,'MDE','RBU','NAF','MCA','SAM','CAS','SAF','PAN','SEA',NOP,SOP,OCN)
    H2 Region order is below
    HTAP2_RECP_REGS = {
          'Ocean': 2,'North America': 3,'Europe': 4, 'South Asia': 5, 'East Asia': 6, 'South East Asia': 7,'Pacific Aus NZ': 8, 'North Africa':9,
          'South Africa': 10,'Middle East': 11, 'Central America': 12, 'South America': 13, 'Rus Bel Ukr': 14,'Central Asia': 15, 'North Pole': 16, 'South Pole': 17          
          }
    '''
    #colours = ['blue','red','purple','green']#,'gold']
    #REGS = ['Europe','North America','East Asia','South Asia']#,'Global']
    colours = ['blue','red','purple','green','gold','black','olive','darkorange','lightcoral',
               'cyan','magenta','yellow','pink','grey','darkseagreen','turquoise']
    REGS = ['Europe','North America','East Asia','South Asia','Rus Bel Ukr','Middle East','Central Asia','North Africa','South Africa',
            'Pacific Aus NZ','South East Asia','South America','Central America','North Pole','South Pole','Ocean']
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.25,left=0.2,top=0.9,wspace=0.2,hspace=0.4) # wspace (width for white space), hspace (height for white space)
    nyrs = len(yrs_all)
    
    for ireg,reg_name in enumerate(REGS):
        print( 'plot up {}'.format(reg_name))
        plt.subplot(4,4,ireg+1)
        
        for i,reg in enumerate(sorted(h2_regions.keys())):
            if reg == reg_name: ind =  i
        
        cur_data = reg_means_all_yrs[:,ind] # for years of data for particular region
        
        # Get min max values to use to set axis for plots
        min_val = np.floor(np.amin(cur_data))
        max_val = np.ceil(np.amax(cur_data)) 
        n_pnts = (max_val - min_val) + 1.0
        
        plt.plot(yrs_all, cur_data, color = colours[ireg], linewidth=2.0)
        plt.title(reg_name,fontsize = 8)
        plt.minorticks_on()
        plt.ylim(min_val,max_val) 
        plt.yticks(np.linspace(min_val,max_val,n_pnts,endpoint=True)) # (start,stop, num pnts)
        
        ax = plt.gca()
        xmin = yrs_all[0]
        xmax = yrs_all[-1]
        if n_pnts/10 <= 5: idiv = 5
        if n_pnts/10 > 5: idiv = 10
        xpnts = (np.float(xmax) - np.float(xmin))/idiv
        print( xpnts)
        xlab = []
        for i in range(np.int(xpnts)+1):
            if i % 2:
                xlab.append('')
            else:
                pnt = np.int(xmin) + (idiv * i) 
                pnt_str = str(pnt)
                xlab.append(pnt_str)
        print( xlab)
        ax.xaxis.set_ticklabels(xlab)
        
        ax.yaxis.grid(b=True,which='major', color = '0.75',linestyle='--', linewidth=0.5) # use 0.75 for grey shades
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(6)
    
    ### CREATE MASTER AXIS FOR WHOLE FIGURE ####
    #plot separate yaxis title for figure
    fig.text(0.15,0.6,'Surface O3 change (ppb)',horizontalalignment='center',
             verticalalignment='center',rotation='vertical',transform=ax.transAxes,fontsize=10)
    #fig.text(0.05,0.5,'Surface O3 change (ppb)',horizontalalignment='center',
    #         verticalalignment='center',rotation='vertical',transform=ax.transAxes,fontsize=10)
    
    plt.savefig(out_fname_pl,bbox_inches='tight')
    print( 'Finished Regional Plots')
    
#---------------------------------------------------
  
def output_file_mod_H1_H2(out_fname,mod_name,emis_scn,nlevs,nlons,nlats,ntime,nyrs,levs,lons,lats,time,years,
                    o3_3d_dat,srf_o3_dat,o3_3d_dat_tot,srf_o3_dat_tot):
    '''
    To make netcdf output of scaled model output for combined HTAP1 and HTAP2 output
    '''
    # NO REGIONS HERE OR SD HERE#
    
    #str_out = np.array([regions], dtype='object')
    #ncdat_out = Dataset(out_fname, 'w', format='NETCDF4_CLASSIC')
    ncdat_out = Dataset(out_fname, 'w')
    #define dimensions
    ncdat_out.createDimension('level_out', nlevs) 
    ncdat_out.createDimension('lat_out', nlats)
    ncdat_out.createDimension('lon_out', nlons) 
    ncdat_out.createDimension('mon_out', ntime)
    #ncdat_out.createDimension('reg_out', nregs+1) 
    ncdat_out.createDimension('years_out', nyrs)
    
    #create variables for output
    mon_out = ncdat_out.createVariable('mon_out', np.float64, ('mon_out',)) 
    years_out = ncdat_out.createVariable('years_out', np.float64, ('years_out',)) 
    level_out = ncdat_out.createVariable('level_out', np.float32, ('level_out',))
    lat_out = ncdat_out.createVariable('lat_out', np.float32, ('lat_out',))
    lon_out = ncdat_out.createVariable('lon_out', np.float32, ('lon_out',))
    #reg_out = ncdat_out.createVariable('reg_out', 'str', ('reg_out',))
    o3_3d_out = ncdat_out.createVariable('o3_3d_out', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out'))
    o3_srf_out = ncdat_out.createVariable('o3_srf_out', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out'))
    o3_3d_out_tot = ncdat_out.createVariable('o3_3d_out_tot', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out')) # fill_value=??
    o3_srf_out_tot = ncdat_out.createVariable('o3_srf_out_tot', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out'))
    
    # Assign data to variables
    mon_out[:] = np.arange(1,ntime+1)
    level_out[:] = levs
    lat_out[:] = lats
    lon_out[:] = lons
    #reg_out[:] = str_out#regions
    years_out[:] = years
    o3_3d_out[:,:,:,:,:] = o3_3d_dat
    o3_srf_out[:,:,:,:] = srf_o3_dat
    o3_3d_out_tot[:,:,:,:,:] = o3_3d_dat_tot
    o3_srf_out_tot[:,:,:,:] = srf_o3_dat_tot
    
    # Assign attributes to variables
    mon_out.setncatts({'long_name': u"month", 'units': u"month"})
    level_out.setncatts({'long_name': u"model level", 'units': u"hPa"})
    lat_out.setncatts({'long_name': u"Latitude", 'units': u"degree_north"})
    lon_out.setncatts({'long_name': u"Longitude", 'units': u"degree_east"})
    years_out.setncatts({'long_name': u"years", 'units': u"years"})
    o3_3d_out_tot.setncatts({'long_name': u"ozone 3D concentration", 'units': u"vmr"})
    o3_3d_out.setncatts({'long_name': u"ozone 3D concentration response", 'units': u"vmr"})
    o3_srf_out.setncatts({'long_name': u"ozone surface concentration response", 'units': u"vmr"})
    o3_srf_out_tot.setncatts({'long_name': u"ozone surface concentration", 'units': u"vmr"})
    
    # Assign Global Attributes 
    ncdat_out.description = 'Total change in ozone concentrations from both HTAP-1 and HTAP-2 '+mod_name+' models scaled in reponse to an imposed emission change from '+emis_scn+' over multiple years'  
    
    #close the file and write information
    ncdat_out.close()
    print( 'Written scaled emission files to netcdf file {}'.format(out_fname))
    
#---------------------------------------------------
  
def output_file_surf_mod_H1_H2(out_fname,mod_name,emis_scn,nlevs,nlons,nlats,ntime,nyrs,levs,lons,lats,time,years,
                    srf_o3_dat,srf_o3_dat_tot):
    '''
    To make netcdf output of scaled model output for combined HTAP1 and HTAP2 output
    '''
    # NO REGIONS HERE OR SD HERE#
    
    #str_out = np.array([regions], dtype='object')
    #ncdat_out = Dataset(out_fname, 'w', format='NETCDF4_CLASSIC')
    ncdat_out = Dataset(out_fname, 'w')
    #define dimensions
    #ncdat_out.createDimension('level_out', nlevs) 
    ncdat_out.createDimension('lat_out', nlats)
    ncdat_out.createDimension('lon_out', nlons) 
    ncdat_out.createDimension('mon_out', ntime)
    #ncdat_out.createDimension('reg_out', nregs+1) 
    ncdat_out.createDimension('years_out', nyrs)
    
    #create variables for output
    mon_out = ncdat_out.createVariable('mon_out', np.float64, ('mon_out',)) 
    years_out = ncdat_out.createVariable('years_out', np.float64, ('years_out',)) 
    #level_out = ncdat_out.createVariable('level_out', np.float32, ('level_out',))
    lat_out = ncdat_out.createVariable('lat_out', np.float32, ('lat_out',))
    lon_out = ncdat_out.createVariable('lon_out', np.float32, ('lon_out',))
    #reg_out = ncdat_out.createVariable('reg_out', 'str', ('reg_out',))
    #o3_3d_out = ncdat_out.createVariable('o3_3d_out', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out'))
    o3_srf_out = ncdat_out.createVariable('o3_srf_out', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out'))
    #o3_3d_out_tot = ncdat_out.createVariable('o3_3d_out_tot', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out')) # fill_value=??
    o3_srf_out_tot = ncdat_out.createVariable('o3_srf_out_tot', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out'))
    
    # Assign data to variables
    mon_out[:] = np.arange(1,ntime+1)
    #level_out[:] = levs
    lat_out[:] = lats
    lon_out[:] = lons
    #reg_out[:] = str_out#regions
    years_out[:] = years
    #o3_3d_out[:,:,:,:,:] = o3_3d_dat
    o3_srf_out[:,:,:,:] = srf_o3_dat
    #o3_3d_out_tot[:,:,:,:,:] = o3_3d_dat_tot
    o3_srf_out_tot[:,:,:,:] = srf_o3_dat_tot
    
    # Assign attributes to variables
    mon_out.setncatts({'long_name': u"month", 'units': u"month"})
    #level_out.setncatts({'long_name': u"model level", 'units': u"hPa"})
    lat_out.setncatts({'long_name': u"Latitude", 'units': u"degree_north"})
    lon_out.setncatts({'long_name': u"Longitude", 'units': u"degree_east"})
    years_out.setncatts({'long_name': u"years", 'units': u"years"})
    #o3_3d_out_tot.setncatts({'long_name': u"ozone 3D concentration", 'units': u"vmr"})
    #o3_3d_out.setncatts({'long_name': u"ozone 3D concentration response", 'units': u"vmr"})
    o3_srf_out.setncatts({'long_name': u"ozone surface concentration response", 'units': u"vmr"})
    o3_srf_out_tot.setncatts({'long_name': u"ozone surface concentration", 'units': u"vmr"})
    
    # Assign Global Attributes 
    ncdat_out.description = 'Total change in ozone concentrations from both HTAP-1 and HTAP-2 '+mod_name+' models scaled in reponse to an imposed emission change from '+emis_scn+' over multiple years'  
    
    #close the file and write information
    ncdat_out.close()
    print( 'Written scaled emission files to netcdf file {}'.format(out_fname))

    
