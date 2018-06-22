#!/usr/bin/env python2.7

'''

Collection of Functions to be used within the HTAP parameterisation for processing HTAP2 model data

Created on Jun 12, 2017
@author: sturnock
'''

import numpy as np
from netCDF4 import Dataset

#################################################

#------------------------------------------------

def read_H2_model_data(fname,cur_scn,srf_nam,col_nam,spec):
    '''
    Read in the HTAP-2 model scenario data
    '''
    ncid = Dataset(fname,'r')
    #extract data from netcdf file
    lats = ncid.variables['lat_out'][:] #extract/copy the data
    lons = ncid.variables['lon_out'][:]
    levs = ncid.variables['levs_out'][:]
    time = ncid.variables['time_out'][:]
    srf_o3_data = ncid.variables[cur_scn+'_'+srf_nam][:] # ozone data for specific model and scenario
    o3_3d_data  = ncid.variables[cur_scn+'_'+col_nam][:]  # 3d ozone data 
    
    #close the input NetCDF file 
    ncid.close()
    
    return lats,lons,levs,time,srf_o3_data,o3_3d_data

#------------------------------------------------

def calc_scal_data_H2(lin_scal_f,non_lin_scal_g,scenario,data_surf,data_3d,ntime,nlevs,nlats,nlons):
    '''
    Applies scaling factor to HTAP2 model data to calculate an O3 response field to the new emission scenario
    '''
    data_surf_sing_reg = np.zeros((ntime,nlats,nlons), dtype='f')
    data_3d_sing_reg = np.zeros((ntime,nlevs,nlats,nlons), dtype='f')
    
    if lin_scal_f != 0.0: #only apply if scaling factor not equal to zero (i.e. not SR1 or SR6 scenarios)
        if scenario != 'nox_resp': # if species is not NOX or linear scaling factor only required (i.e. not Nox or CH4)
            print 'Adjust Response using LINEAR scaling factor of {:.3f}'.format(lin_scal_f)
            
            # Scale regional response fields where there is valid data (i.e. not -999.0)
            # Surface fields
            data_surf_sing_reg[data_surf != -999.0] = (data_surf[data_surf != -999.0] * lin_scal_f) 
            # 3D fields (these are masked arrays so only apply scaling factor to valid points)
            data_3d_sing_reg = (data_3d.filled(0) * lin_scal_f)
        
        else: # NOx Case
            print 'For NOX emissions use Non-linear Emission Scaling factor g of {:.3f}'.format(non_lin_scal_g)
            
            # To capture specific cases for Titration effects specify different scaling factor 
            if lin_scal_f < 0: # NOx increases relative to a original 20% reduction so why scaling factor below 0
                nox_scal_fact = lin_scal_f # Use linear scaling for emission increases and ozone increases
                print 'For NOX increases use Linear NOX scaling factor for Titration Regime of {:3f}'.format(lin_scal_f)
            else:
                nox_scal_fact = ((2.0 * lin_scal_f) - non_lin_scal_g) # Reduced scaling factor for where NOx emission decreases relative to 20% and ozone increases occur 
                print 'For NOX decreases use Adjusted NON-Linear NOX scaling factor for Titration regime of {:.3f}'.format(nox_scal_fact)
            
            # Apply scaling factor to Surface fields (all numpy arrays)            
            # For O3 decrease with NOX decrease
            data_surf_sing_reg[(data_surf < 0.0) & (data_surf != -999.0)] = (data_surf[(data_surf < 0.0) & (data_surf != -999.0)]* non_lin_scal_g)
            # For Titration regimes (i.e. ozone increase for emission change use either linear (for emission increase) or 2f-g for emission decrease (i.e. titration))                                   
            data_surf_sing_reg[(data_surf >= 0.0) & (data_surf != -999.0)] = (data_surf[(data_surf >= 0.0) & (data_surf != -999.0)]* nox_scal_fact)
            
            # Apply scaling factor to 3D fields (all numpy masked arrays)
            # For O3 decrease with NOX decrease
            data_3d_sing_reg[data_3d < 0.0] = (data_3d[data_3d < 0.0].filled(0) * non_lin_scal_g)
            # For Titration regimes
            data_3d_sing_reg[data_3d >= 0.0] = (data_3d[data_3d >= 0.0].filled(0) * nox_scal_fact)
            
    else: 
        print 'Modelled fields are not scaled as linear scaling factor is Zero'
    
    return data_surf_sing_reg,data_3d_sing_reg

#------------------------------------------------

def output_txt_file_reg_individ_mod_h2(out_fname,header,years_arr,h2_reg,reg_vals,glob_val,mod_num):
    '''
    Output text file of regional mean concentrations of O3 response for individual HTAP2 models
    '''
    # Open the file for writing
    df = open(out_fname, 'w')
    df.write(header+'\n') # Write Header Line across top of file
    # Write out regional concentration data line by line
    for (iyr,yr) in enumerate(years_arr): #convert original list of years into an array of years to be used
        #for (ireg_h1,reg_h1) in enumerate(reg_arrs):
        cur_line_dat = []
        for line_dat in reg_vals[mod_num,iyr,:]:
            cur_line_dat.append(str(line_dat))
                    
        glob_mean_val = str(glob_val[mod_num,iyr])
                    
        cur_line_str = ','.join(cur_line_dat)     
        df.write(yr+','+h2_reg+','+cur_line_str+','+glob_mean_val+'\n')
    df.close()

#---------------------------------------------------

def output_file_h2_mmm(out_fname,mod_name,emis_scn,nlevs,nlons,nlats,ntime,nyrs,nregs,levs,lons,lats,time,regions,years,
                    o3_3d_dat_reg,srf_o3_dat_reg,o3_3d_dat_reg_tot,srf_o3_dat_reg_tot,
                    o3_3d_dat_reg_sd,srf_o3_dat_reg_sd,o3_3d_dat_reg_tot_sd,srf_o3_dat_reg_tot_sd):
    '''
    To make netcdf output of scaled model output for the multi-model mean of HTAP2 models
    '''
    
    str_out = np.array([regions], dtype='object')
    #ncdat_out = Dataset(out_fname, 'w', format='NETCDF4_CLASSIC')
    ncdat_out = Dataset(out_fname, 'w')
    #define dimensions
    ncdat_out.createDimension('level_out', nlevs) 
    ncdat_out.createDimension('lat_out', nlats)
    ncdat_out.createDimension('lon_out', nlons) 
    ncdat_out.createDimension('mon_out', ntime)
    ncdat_out.createDimension('reg_out', nregs) 
    ncdat_out.createDimension('years_out', nyrs)
    
    #create variables for output
    mon_out = ncdat_out.createVariable('mon_out', np.float64, ('mon_out',)) 
    years_out = ncdat_out.createVariable('years_out', np.float64, ('years_out',)) 
    level_out = ncdat_out.createVariable('level_out', np.float32, ('level_out',))
    lat_out = ncdat_out.createVariable('lat_out', np.float32, ('lat_out',))
    lon_out = ncdat_out.createVariable('lon_out', np.float32, ('lon_out',))
    reg_out = ncdat_out.createVariable('reg_out', 'str', ('reg_out',))
    
    o3_3d_reg_out = ncdat_out.createVariable('o3_3d_reg_out', np.float32, ('years_out','mon_out', 'level_out', 'lat_out', 'lon_out','reg_out'))
    o3_srf_reg_out = ncdat_out.createVariable('o3_srf_reg_out', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out','reg_out'))
    o3_3d_reg_out_tot = ncdat_out.createVariable('o3_3d_reg_out_tot', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out','reg_out')) # fill_value=??
    o3_srf_reg_out_tot = ncdat_out.createVariable('o3_srf_reg_out_tot', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out','reg_out'))
    
    o3_3d_reg_out_sd = ncdat_out.createVariable('o3_3d_reg_out_sd', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out','reg_out'))
    o3_srf_reg_out_sd = ncdat_out.createVariable('o3_srf_reg_out_sd', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out','reg_out'))
    o3_3d_reg_out_tot_sd = ncdat_out.createVariable('o3_3d_reg_out_tot_sd', np.float32, ('years_out', 'mon_out', 'level_out', 'lat_out', 'lon_out','reg_out')) # fill_value=??
    o3_srf_reg_out_tot_sd = ncdat_out.createVariable('o3_srf_reg_out_tot_sd', np.float32, ('years_out', 'mon_out', 'lat_out', 'lon_out','reg_out'))
    
    # Assign data to variables
    mon_out[:] = np.arange(1,ntime+1)
    level_out[:] = levs
    lat_out[:] = lats
    lon_out[:] = lons
    reg_out[:] = str_out#regions
    years_out[:] = years
    
    o3_3d_reg_out[:,:,:,:,:,:] = o3_3d_dat_reg
    o3_srf_reg_out[:,:,:,:,:] = srf_o3_dat_reg
    o3_3d_reg_out_tot[:,:,:,:,:,:] = o3_3d_dat_reg_tot
    o3_srf_reg_out_tot[:,:,:,:,:] = srf_o3_dat_reg_tot
    
    o3_3d_reg_out_sd[:,:,:,:,:,:] = o3_3d_dat_reg_sd
    o3_srf_reg_out_sd[:,:,:,:,:] = srf_o3_dat_reg_sd
    o3_3d_reg_out_tot_sd[:,:,:,:,:,:] = o3_3d_dat_reg_tot_sd
    o3_srf_reg_out_tot_sd[:,:,:,:,:] = srf_o3_dat_reg_tot_sd
    
    # Assign attributes to variables
    mon_out.setncatts({'long_name': u"month", 'units': u"month"})
    level_out.setncatts({'long_name': u"model level", 'units': u"hPa"})
    lat_out.setncatts({'long_name': u"Latitude", 'units': u"degree_north"})
    lon_out.setncatts({'long_name': u"Longitude", 'units': u"degree_east"})
    reg_out.setncatts({'long_name': u"List of Regional Contributions to Total"})
    years_out.setncatts({'long_name': u"years", 'units': u"years"})
    
    o3_3d_reg_out_tot.setncatts({'long_name': u"regional ozone 3D concentration", 'units': u"vmr"})
    o3_3d_reg_out.setncatts({'long_name': u"regional ozone 3D concentration response", 'units': u"vmr"})
    o3_srf_reg_out.setncatts({'long_name': u"regional ozone surface concentration response", 'units': u"vmr"})
    o3_srf_reg_out_tot.setncatts({'long_name': u"regional ozone surface concentration", 'units': u"vmr"})
    
    o3_3d_reg_out_tot_sd.setncatts({'long_name': u"Standard deviation regional ozone 3D concentration", 'units': u"vmr"})
    o3_3d_reg_out_sd.setncatts({'long_name': u"Standard deviation regional ozone 3D concentration response", 'units': u"vmr"})
    o3_srf_reg_out_sd.setncatts({'long_name': u"Standard deviation regional ozone surface concentration response", 'units': u"vmr"})
    o3_srf_reg_out_tot_sd.setncatts({'long_name': u"Standard deviation regional ozone surface concentration", 'units': u"vmr"})
    
    # Assign Global Attributes 
    ncdat_out.description = 'Multi-model mean change in ozone concentrations from HTAP2 model values that have been scaled in response to an imposed emission change from '+emis_scn+' over different years'   
    
    #close the file and write information
    ncdat_out.close()
    print 'Written scaled emission files to netcdf file {}'.format(out_fname)
  
#----------------------------------------------------------------------------    