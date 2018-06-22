#!/usr/bin/env python3
'''

New version of O3 parametric model set up to use Scaled baseline of 2010
Individual HTAP-I modelled O3 response for Europe, North America, South Asia and East Asia,
scaled to the HTAP-II source regions to account for the change in size of source regions.
Additional HTAP2 regions are included based on the available HTAP2 model data 
for each particular regional perturbation scenario

Ozone response is output onto a 1x1 degree grid for conversion into mean values over regions defined in TF-HTAP2

ARGUMENTS:
Link to 2010 baseline from parametric model
Link to gridded O3 response from HTAP-I model output on 1x1 grid (21 vertical) and already scaled to account for change in source region (HTAP 1 to 2)
Link to gridded O3 response from HTAP-2 model output on 1x1 grid (21 vertical) for additional HTAP2 source regional perturbations

Fractional Change in emissions from a specific scenario across all HTAP2 Source regions (including new emission source regions)
Two files are required for emission changes:
1 - for HTAP1 Source regions (EUR,NAM,EAS,SAS), 
2 - for all other HTAP2 Source regions have data for

All additional functions are now called from separate HTAP_function modules (see import below for htap_functions, H1_funcs and H2_funcs)
Main htap_functions module requires link to:
- the IDL sav file containing the ACCMIP forcing relationship
- the netcdf file defining the HTAP2 regions

OUTPUT - Various Options (see switches below)
Multi-model scaled O3 response including contributions from both HTAP-I and HTAP2 models to new emission change scenario 
Output is a 3D grid at 1x1 degrees with 21 standard pressure levels in the vertical 
Surface output is also processed to produce regional mean values for the 16 HTAP2 Receptor regions
The output from the parameterisation can be set as follows:
1. Text file containing multi-model total ozone response (HTAP1 and 2 combined) from all emission source region perturbations
2. Text files containing multi-model ozone response (+ S.D.) to emission perturbations for individual source regions (both HTAP1 and 2)
3. Netcdf file containing gridded monthly 3D (12x21x180x360) multi-model total ozone response (HTAP1 and 2 combined) from all emission source region perturbations   
4. Netcdf file containing gridded monthly 3D (12x21x180x360) multi-model ozone response (+ S.D.) to emission perturbations for individual source regions (both HTAP1 and 2)
5. Time series graph of multi-model total ozone response (HTAP1 and 2 combined) from all emission source region perturbations (potentially available for HTAP1 and 2 source regions)

Reproduced and added additional features to the method described in Wild et al., 2012 ACP 12,2037-2054.

Created on May 8, 2017
@author: Steven Turnock

Details of Developments for publication (pre-repository version control)

#### UPDATE 07/09/2017 ####
Set up to now combine scaled parametric response to CH4 from both HTAP1 and HTAP2 models
Also look at including CH4INC from HTAP2 as larger number of models

#### UPDATE OCtober 2017 #####
To include calculation of ozone burden and ozone radiative forcing for each model and total

#### UPDATE November 2017 ####
Update O3 radiative forcing calculations to use 2D map of ACCMIP multi-model mean normalised forcing response

#### UPDATE DECEMBER 2017 ####
Included O3 response fields from the EMEP model, specifically include the Ocean as a source region as left out before

Include contributions from 5 models that supplied CH4INC simulations for 2010 to parameterisation as another source to make 21 models for CH4 in total)

#### UPDATE June 2018 #####
Cleaned up and slightly edited (not technically) main python script and function modules for upload to the internal Met Office repository at 
(http://fcm9/projects/utils). This will become the version 1.0 of the parameterisation and associated with the ACP paper.

#### UPDATE May 2020 #####
Included if statements to correct combined response output files to account for a for zero CH4 change

### UPDATE November 2021 ###
Modified to work with Python 3

'''

# Python Modules to import
import glob
import numpy as np
import numpy.ma as ma
import matplotlib
# use the Agg environment to generate an image rather than outputting to screen 
# (FOR USE WHEN SUBMITTING TO MET OFFICE SPICE SYSTEM - remove if not necessary)
matplotlib.use('Agg')

# Specific functions related to the parameterisation
import functions.htap_functions as ht_fn
import functions.H1_funcs as ht1_fn
import functions.H2_funcs as ht2_fn

################################################

# SPECIFY DETAILS FOR PARAMETERISATION

# BASELINE FILE (ALWAYS KEEP SAME - unless changing baseline)
BASE_FILE_PATH  = '/data/users/sturnock/HTAP_files/para_o3_model/HTAP_2000_2010_scaling/' # Path to 2010 Baseline Param O3 concs
BASE_CONC_FNAME = 'Multi_HTAP-I_model_mean_plus_stdev_o3_2010_HTAP-II_2010_invent_emis_change_New_Baseline_V1_emis_change_1x1_grid.nc' # 1x1 2010 baseline concentrations file

#-----------------------------------------------

# ACCMIP Burden to forcing Relationship
# 2D array of normalised O3 RF from ACCMIP multi-model mean (Fig 3d in Stevenson et al., 2013)
ACCMIP_FILE = '/data/users/sturnock/ACCMIP_data/normalised_o3t_rf_ACCMIP_March2013.sav'

#-----------------------------------------------

# EMISSION SCENARIO
EMIS_SCN  = 'Emis_scn_CMIP6_SSP5_85_2015_base'   # used for naming output files
EMISSIONS = ['NOx','VOC','CO']       # Emissions precursors to be perturbed
SPEC      = 'o3'                     # output species to be scaled
YEARS_SCN = ['2020','2030','2040','2050']                 # Specify years considered within specific scenario relative to 2010
# For Historical Years
#YEARS_RAW = np.arange(1771,1781,1)
#YEARS_SCN = []
#for iyr,yr in enumerate(YEARS_RAW):
#    #print yr
#    YEARS_SCN.append(str(yr))

# EMISSION FILES  ## ALWAYS USE FRACTIONAL EMISSIONS (REL TO BASE) ###
EMIS_FILE_PATH = '/home/h06/sturnock/Python/HTAP/para_o3_model/emission_change_input/combined_H1_H2/'  # Path to Emission scenario (fractional change considered)
EMIS_SCN_FNAME_H1 = 'H1_emis_CMIP6_SSP5_85_2015_2020_2030_2040_2050.dat'                           # Emission changes for HTAP1 source regions
EMIS_SCN_FNAME_H2 = 'H2_emis_CMIP6_SSP5_85_2015_2020_2030_2040_2050.dat'                           # Emission changes for HTAP2 Source Regions

#-----------------------------------------------

# SOURCE REGIONS
# Define HTAP1 and HTAP2 source regions to process different models

# HTAP-1 regions of interest: EU (Europe), NA (North America), SA (South Asia), EA (East Asia), RW (Rest of World)
H1_SRC_REGIONS = ['EUR','NAM','SAS','EAS'] 
# 20% HTAP-1 Emission reductions scenarios SR1 - control, SR2 - CH4, SR3 - NOX, SR4 - NMVOC, SR5 - CO, SR6 - ALL + Aerosols
# H1 Source Regions Scaled to H2 Equivalent regions of EU - Europe, NA - North America, SA - South Asia, EA - East Asia, RW - Rest of World
H1_RUNID = ['SR2','SR3EU','SR3NA','SR3SA','SR3EA','SR4EU','SR4NA','SR4SA','SR4EA','SR5EU','SR5NA','SR5SA','SR5EA']
# REMOVE SR1 (BASE), ALL RW (rest of world) and ALL SR6 as do not use them in parameterisation
NRUNS_H1 = len(H1_RUNID)
# Filepath for HTAP1 Models O3 response (scaled to represent HTAP2 regions and converted onto standard grid 12x21x180x360)
MOD_FILE_PATH_H1 = '/data/users/sturnock/HTAP_files/para_o3_model/HTAP_1_individ_models_1x1_grid_src_scaled_o3_resp/'
SRF_NAME_H1       = 'srf'   # Surface data ID
COL_NAME_H1       = 'vmr'   # 3D data ID

# HTAP2 Source regions: Europe, North America, South Asia, East Asia, Middle East, Russia Belarus Ukraine, North Africa, Central America, South America, Central Asia, South Africa, Pacific Australia New Zealand, South East Asia, Oceans
H2_SRC_REGIONS = ['CH4','MDE','RBU','NAF','MCA','SAM','CAS','SAF','PAN','SEA','OCN']      
# CH4 Perturbations are separate single scenario for HTAP2
# 20% HTAP-2 Emission reductions - Theses are all in Separate Files for each model and each regional perturbation scenario (need to process these for each file)
H2_SCN_ID = ['CH4ALL','MDEALL','RBUALL','NAFALL','MCAALL','SAMALL','CASALL','SAFALL','PANALL','SEAALL','OCNALL']       
H2_RUNID = ['nox_resp','nmvoc_resp','co_resp']
NRUNS_H2 = len(H2_RUNID)
# Filepath for HTAP2 models for the scenarios listed above (converted onto standard grid 12x21x180x360)
MOD_FILE_PATH_H2 = '/data/users/sturnock/HTAP_files/HTAP2_models/'

#-----------------------------------------------

# RECEPTOR DEFINITION
# HTAP2 Receptor Regions
HTAP2_RECP_REGS = {
          'Ocean': 2,'North America': 3,'Europe': 4, 'South Asia': 5, 'East Asia': 6, 'South East Asia': 7,'Pacific Aus NZ': 8, 'North Africa':9,
          'South Africa': 10,'Middle East': 11, 'Central America': 12, 'South America': 13, 'Rus Bel Ukr': 14,'Central Asia': 15, 'North Pole': 16, 'South Pole': 17          
          }
HTAP2_RECP_REGS_H1 = {'North America': 3,'Europe': 4, 'South Asia': 5, 'East Asia': 6}
HTAP2_RECP_REGS_H2 = {'Ocean': 2,'South East Asia': 7,'Pacific Aus NZ': 8, 'North Africa':9,'South Africa': 10,'Middle East': 11, 'Central America': 12, 'South America': 13, 'Rus Bel Ukr': 14,'Central Asia': 15}

#-----------------------------------------------

# OUTPUT DIRECTORIES
OUT_TXT_FILE_PATH   = '/home/h06/sturnock/Python/HTAP/para_o3_model/CMIP6_HTAP_param/txt_file_out/'  # Path for regional mean output
OUT_FILE_PATH       = '/data/users/sturnock/HTAP_files/para_o3_model/CMIP6_new_scns/'                        # Path for netcdf output of scaled O3 response
PLOT_DIR            = '/home/h06/sturnock/Python/HTAP/para_o3_model/CMIP6_HTAP_param/image_out/'     # Path for regional change plots

#-----------------------------------------------

# SETTINGS AND SWITCHES
IOUT_CALC_MMM_H1 = 1    # set to specifically calculate Regional mean responses from HTAP-1 multi-model values
IOUT_CALC_MMM_H2 = 1    # set to specifically calculate Regional mean responses from HTAP-2 multi-model values
IOUT_NCF_MMM_H1  = 0    # to output gridded values of the multi-model response from ONLY HTAP-1 models to a netcdf file (SLOWS DOWN CODE)
IOUT_NCF_MMM_H2  = 0    # to output gridded values of the multi-model response from ONLY HTAP-2 models to a netcdf file for all regional emis scenarios
IOUT_TXT_MMM_H1  = 1    # output text file for regional HTAP1 multi-model mean response + std devs
IOUT_TXT_MMM_H2  = 1    # output text file for regional HTAP2 multi-model mean response + std devs to all regional emis scenarios   
IOUT_NCF_INDIVID_H1 = 0 # set to output netcdf files of INDIVIDUAL HTAP1 MODEL responses
IOUT_TXT_INDIVID_H1 = 1 # output text file for regional responses from each INDIVIDUAL HTAP1 model 
IOUT_TXT_INDIVID_H2 = 1 # output text file for regional responses from each INDIVIDUAL HTAP2 model
CALC_REG_HTAP2_H1   = 1 # perform calculation of regional concentrations from HTAP1 models
CALC_REG_HTAP2_H2   = 1 # perform calculation of regional concentrations from HTAP2 models
IPRINT_REG          = 0 # print out regional responses for individual models to Screen
IOUT_NCF_MMM_COMB   = 0 # to output gridded values of the multi-model response from COMBINED HTAP-1 and HTAP-2 models to a netcdf file
IPLOT_REG_MMM_COMB  = 0 # Plot up total O3 response values across all HTAP2 receptors

#-----------------------------------------------

# SCALE FACTOR SETTINGS FOR METHANE (based on HTAP1 concentrations as all HTAP2 CH4 responses have been pre-converted to a 20% reduction equivalent)
# CH4 change from base value (1760 ppb - value fixed in HTAP-I SR1 scenarios) to -20% reduction (1408 ppb)
CH4_CHANGE_20_H1 = (1760.0 - 1408.0) 
# For HTAP2 CH4DEC is a change from base value (1798 ppb - value fixed in 2010 for HTAP-2 Scenarios) to -13% reduction (1562 ppb) RCP2.6 in 2030
# For HTAP2 CH4INC hange from base value (1798 ppb - value fixed in 2010 for HTAP-2 Scenarios) to +18% reduction (2121 ppb) RCP8.5 in 2030
# Both of these scenarios for HTAP2 have been converted to represent the abundance change above

# Emission scaling coefficients from equation for non-linear scaling f = 0.95r + 0.05r^2
A = 0.05 #0.95
B = 0.95 #0.05
# Whether to solely use a linear scaling factor or not - for O3 set to false as need to take account of CH4 lifetime effect and NOx. Other species True
LINEAR = 0 # Set to FALSE == 0 # THIS SETTING IS NOT USED IN THE CURRENT PARAMETERISATION BUT HAS BEEN INCLUDED FOR FUTURE PROOFING

#-----------------------------------------------

# CONSTANTS
NYRS  = len(YEARS_SCN)          # number of Years
NREGS_H1 = len(H1_SRC_REGIONS)  # number of H1 source regions
NREGS_H2 = len(H2_SRC_REGIONS)  # number of H2 source regions
NREGS_H2_OUT = len(HTAP2_RECP_REGS.items()) # Number of HTAP2 receptor regions

################################################

### MAIN CODE OF PARAMETERISATION ###

if __name__ == '__main__':
    
    print( 'Calculate {} Response to {} scenario'.format(SPEC,EMIS_SCN))
    
    #-----------------------------------------------
    ##### BASELINE ######
    
    # Read in Gridded Ozone Data for Baseline Scenario
    print( 'Read in 2010 Baseline concentrations from file {}'.format(BASE_CONC_FNAME))
    base_nc = BASE_FILE_PATH+BASE_CONC_FNAME
    lats,lons,levs,time,srf_o3_data_SR1,o3_3d_data_SR1_init = ht_fn.read_model_data_base(base_nc,'out_tot',SPEC,'srf')
   
    # Make O3 baseline values from numpy array back into masked array
    o3_3d_data_SR1 = ma.masked_invalid(o3_3d_data_SR1_init) 
    
    # define variables
    ntime = len(time)
    nlats = len(lats)
    nlons = len(lons)
    nlevs = len(levs)
        
    # Calculate difference between 3D pressure levels for use in burden calculation
    dp = levs[0:-1] - levs[1:]              # pressure level difference
    dp = np.hstack((dp,20)) * 1e2           # To convert from hPa to Pa (kg m-1 s-2)
    dp_3d = np.zeros(o3_3d_data_SR1.shape)
    for ilev in range(len(dp)):             # assign dp to the same shape as the O3 response arrays
        dp_3d[:,ilev,:,:] = dp[ilev]
    #print dp_3d.shape
    
    print( 'Calculate grid box areas')
    area2d_global       = ht_fn.calc_area_any(nlats,nlons,lats,lons)
    grid_areas_mon_3d   = np.ones(o3_3d_data_SR1.shape) * area2d_global # reshape to be the same as O3 response arrays
    
    print( 'Calculate baseline Global surface O3 Conc')
    srf_o3_data_SR1_glob = ht_fn.calc_glob_mean(srf_o3_data_SR1,area2d_global) 
    print( 'Baseline Surface global annual mean O3 conc = {:.2f} ppbv'.format(srf_o3_data_SR1_glob))
    
    # Load in 2d map of ACCMIP normalised O3 RF
    print( 'read in ACCMIP normalised O3 radiative forcing')
    accmip_2d_rf = ht_fn.read_pro_ACCMIP_rf(ACCMIP_FILE,lats,lons)
    
    print( 'Calculate Annual mean Total Column Burden and radiative Effect for BASE')
    # only calculate column O3 burden where O3 concentrations < 150ppb (definition of tropopause)
    # base_burd (kg) = vmr * (kg / kg * m s-2) * Pa (kg m-1 s-2) * m2
    base_burd, base_burd_dob, base_rf_glob, base_burd_htap2_regs, base_rf_htap2_regs = ht_fn.calc_burd_ACCMIP(o3_3d_data_SR1,o3_3d_data_SR1,dp_3d,grid_areas_mon_3d,accmip_2d_rf,NREGS_H2_OUT,HTAP2_RECP_REGS)
    
    # Test if baseline surface ozone field is numpy array or masked array (i.e. missing values) (EMEP model is problematic otherwise
    if isinstance(srf_o3_data_SR1, (np.ma.core.MaskedArray)):
        srf_o3_data_SR1 = srf_o3_data_SR1.filled() # copy of masked array and makes numpy array with masked values filled with -999.0
    
    # Produce baseline arrays for total O3 concentrations
    srf_o3_data_SR1_reg     = np.zeros((ntime,nlats,nlons,NREGS_H1+1), dtype='f')
    o3_3d_data_SR1_all_reg  = np.zeros((ntime,nlevs,nlats,nlons,NREGS_H1+1), dtype='f')
    
    # put baseline arrays onto same shape as O3 reseponse fields for use in calculations later
    for ireg in range(NREGS_H1+1):
        srf_o3_data_SR1_reg[:,:,:,ireg]         = srf_o3_data_SR1[:,:,:]
        o3_3d_data_SR1_all_reg[:,:,:,:,ireg]    = o3_3d_data_SR1.filled(-999.0)
    
    # Make a masked array again
    o3_3d_data_SR1_all_reg_ms = ma.masked_equal(o3_3d_data_SR1_all_reg, -999.0)
        
    print( '### Finished reading in 2010 base scenario of {} ###'.format(BASE_CONC_FNAME))
    
    #-----------------------------------------------
    ###### FRACTIONAL EMISSION CHANGE #######
    
    # Read in Fractional Emission Change data
    print( 'Read in emissions data for {} scenario'.format(EMIS_SCN))
    # read in emission file for a particular scenario for both HTAP1 and HTAP2 source regions 
    # (that contains the fractional emission change from future year relative to base year 2010) 
    # Read and Process emission file to look for right year and find fractional emission change for each region
    
    # Process HTAP1 equivalent Source Regions (yrs,species,regs)
    ch4_base_out_h1,ch4_fut_out_h1,emis_data_fract_h1 = ht_fn.read_process_emis_data(EMIS_FILE_PATH+EMIS_SCN_FNAME_H1,NYRS,NREGS_H1,YEARS_SCN)
    # Process additional HTAP2 source regions (yrs,emis,regs))
    ch4_base_out_h2,ch4_fut_out_h2,emis_data_fract_h2 = ht_fn.read_process_emis_data(EMIS_FILE_PATH+EMIS_SCN_FNAME_H2,NYRS,NREGS_H2-1,YEARS_SCN)
        
    #-----------------------------------------------
    ###### CALCULATE EMISSION SCALING FACTORS #######
        
    print( 'Calculate Emission scaling factors for {} scenario across all regions and all species'.format(EMIS_SCN))
    
    # Process scaling factors for H1 equivalent source regions
    emis_scal_f_h1, emis_scal_g_h1 = ht_fn.calc_emis_scal(emis_data_fract_h1,ch4_base_out_h1,ch4_fut_out_h1,YEARS_SCN,EMISSIONS,H1_SRC_REGIONS,NRUNS_H1,CH4_CHANGE_20_H1,A,B,LINEAR,'H1')
    print( emis_scal_f_h1[0,:])
    
    # Process scaling factors for H2 additional source regions
    emis_scal_f_h2, emis_scal_g_h2 = ht_fn.calc_emis_scal(emis_data_fract_h2,ch4_base_out_h2,ch4_fut_out_h2,YEARS_SCN,EMISSIONS,H2_SRC_REGIONS,NRUNS_H2,CH4_CHANGE_20_H1,A,B,LINEAR,'H2')
    #### NO BASE or SR or RW in HTAP1 now
    #### Order of emiss array comes out as CH4, NOX, NMVOC, CO (and then repeat for each region in H1 but not H2)
    #### HTAP1 array (year,runid)
    #### HTAP2 array stays as (year,emis ,reg (inc CH4 as first point)) so process emission factors base on regions here as separate files now
    #### All H1 scenarios in same model file. ALL H2 scenarios are different files for each perturbation
    print( '#### Finished processing Emission Scaling factors ####')
    
    #-----------------------------------------------
    ###### APPLY EMISSION SCALING FACTORS TO MODELLED O3 FIELDS FROM HTAP #######

    # Apply scaling factor to original HTAP-I and HTAP2 ozone fields to generate new response fields
    print( 'Apply Emission scaling factors to Ozone Response from each HTAP scenario')
    
    ###### HTAP 1 MODELS ###### 
    
    # Need to process HTAP-1 Models and Scenarios separately to HTAP2
    print( '#### First Process HTAP-1 Multi-Model Responses #####')
    MODEL_FNAMES_H1    =  glob.glob(MOD_FILE_PATH_H1+'*.nc')
    NMODS_H1 = len(MODEL_FNAMES_H1)
    print( 'List of HTAP-1 Models to Process')
    print( MODEL_FNAMES_H1)

    ################################
    
    # INITIALISE OUTPUT ARRAYS
    print( 'Initialise Output Arrays')
    # Initialise Scaled Output Arrays to contain data for all HTAP-I models
    o3_3d_reg_all_yrs_all_mod_H1        = np.zeros((NMODS_H1,NYRS,ntime,nlevs,nlats,nlons,NREGS_H1+1), dtype='f')
    o3_srf_reg_all_yrs_all_mod_H1       = np.zeros((NMODS_H1,NYRS,ntime,nlats,nlons,NREGS_H1+1), dtype='f')
    o3_3d_reg_tot_all_yrs_all_mod_H1    = np.zeros((NMODS_H1,NYRS,ntime,nlevs,nlats,nlons,NREGS_H1+1), dtype='f')
    o3_srf_reg_tot_all_yrs_all_mod_H1   = np.zeros((NMODS_H1,NYRS,ntime,nlats,nlons,NREGS_H1+1), dtype='f')
    
    # Initialise arrays for global mean values
    global_mean_val_H1      = np.zeros((NMODS_H1,NYRS,NREGS_H1+1), dtype='f')
    global_mean_val_act_H1  = np.zeros((NMODS_H1,NYRS,NREGS_H1+1), dtype='f')
        
    # Global burden, dobs and RF arrays
    o3_burd_all_mod_H1      = np.zeros((NMODS_H1,NYRS,NREGS_H1+1), dtype='f')
    rf_glob_all_mod_H1      = np.zeros((NMODS_H1,NYRS,NREGS_H1+1), dtype='f')
    reg_o3_burd_all_mod_H1  = np.zeros((NMODS_H1,NYRS,NREGS_H1+1,NREGS_H2_OUT), dtype='f')
    reg_o3_rf_all_mod_H1    = np.zeros((NMODS_H1,NYRS,NREGS_H1+1,NREGS_H2_OUT), dtype='f')
    
    # Initialise Regional mean value array
    # Regional mean values in response to each emission perturbation
    reg_mean_vals_H1        = np.zeros((NMODS_H1,NYRS,NREGS_H1+1,NREGS_H2_OUT), dtype='f')
    reg_mean_vals_act_H1    = np.zeros((NMODS_H1,NYRS,NREGS_H1+1,NREGS_H2_OUT), dtype='f')
    
    ################################
    
    # PROCESS HTAP1 MODELS
    print( '###### Process HTAP1 Models ######')
    # Process each HTAP1 model file in turn for EUR, NAM, SAS and EAS response
    for (imod,mod_file) in enumerate(MODEL_FNAMES_H1):
        
        # setting not for Rest of World field now as not using them but to identify any missing values in data that need to be filled with multi-model mean
        no_rw = 0 
        
        file_txt            = mod_file.split('/')
        cur_mod             = '_'.join(file_txt[7:8])
        cur_mod_short_txt   = cur_mod.split('_')
        cur_mod_short       = '_'.join(cur_mod_short_txt[0:1])
        print( 'Calculate {} Response to emission scenario {} from HTAP-I model {}'.format(SPEC,EMIS_SCN,cur_mod_short))
        
        ################################
        
        # Calculate changes for each future year of scenario
        for (iyr,year) in enumerate(YEARS_SCN):
            print( 'Calculate Emission changes for {}'.format(year))
            
            # Initialise new arrays to store the output scaled Ozone responses for each model in
            print( 'Initialise Output arrays')
            # Individual regional contribution response fields
            srf_o3_data_reg     = np.zeros((ntime,nlats,nlons,NREGS_H1+1), dtype='f') # individual regional contributions from all species and for CH4 separately
            o3_3d_data_all_reg  = np.zeros((ntime,nlevs,nlats,nlons,NREGS_H1+1), dtype='f')
            srf_o3_data_reg_tot = np.zeros((ntime,nlats,nlons,NREGS_H1+1), dtype='f')
    
            
            # Take each point individually from the ozone change of each scenario 
            # (base - 20% reduction for each species and region) and then apply relevant scaling factor 
            # for this emission change. Total up to get total ozone response
            # also need to loop through all model runs done and apply scaling factor in order
            
            ################################
            
            # loop through each HTAP1 perturbation scenario
            for (irun,run) in enumerate(H1_RUNID): #loop through each scenario of ozone changes (base - 20% reduction)
                print( 'Processing scenario {}'.format(run))
                cur_scen,ireg,reg = ht_fn.find_reg(run) # Find out which region to use
                
                # Load in surface and 3D Ozone reponse field for each scenario and for each model in order  
                lats,lons,levs,time,srf_o3_data,o3_3d_data = ht1_fn.read_H1_model_data(mod_file,run,SRF_NAME_H1,COL_NAME_H1,SPEC)
                print( 'Read in HTAP-1 Model data for scenario {} and {}'.format(run,reg))
                
                # Test if surface ozone field is numpy array or masked array (i.e. missing values)
                if isinstance(srf_o3_data, (np.ma.core.MaskedArray)):
                    print( 'Surface array is a masked array so create a numpy array')
                    srf_o3_data = srf_o3_data.filled() # copy of masked array and makes numpy array with masked values filled with -999.0
                    no_rw = 1 # If surface is masked array then has missing values and need to use rest of the world to fill in
                
                # Test if 3D array is a numpy array as should be a masked array (EMEP model 3D files are numpy arrays)
                if isinstance(o3_3d_data, (np.ndarray)):
                    print( '3D array is a numpy array so create a masked array')
                    mask_arr    = np.isnan(o3_3d_data) # create mask of boolean values for where points are NaN
                    o3_3d_data  = ma.masked_array(o3_3d_data,mask=mask_arr)
                
                print( 'Apply future emission Scaling factor to individual model HTAP-I O3 response for scenario {}'.format(run))
                # use HTAP1 O3 response to 20% emission perturbation and scale by new emission fraction
                srf_o3_data_sing_reg,o3_3d_data_sing_reg = ht1_fn.calc_scal_data(emis_scal_f_h1[iyr,irun],emis_scal_g_h1[iyr,irun],cur_scen,ireg,cur_mod,no_rw,
                                                                                srf_o3_data,o3_3d_data,ntime,nlevs,nlats,nlons)
                
                # Add Single regional contributions into separate arrays (but total of NOX, CO and VOC emis)
                srf_o3_data_reg[:,:,:,ireg]         += srf_o3_data_sing_reg[:,:,:]
                o3_3d_data_all_reg[:,:,:,:,ireg]    += o3_3d_data_sing_reg[:,:,:,:] 
                
                no_rw = 0 # reset no data flag for next model
            
            ################################    
            
            # NEED TO ADD BACK ON BASE 2010 CONCENTRATIONS TO GET NEW OZONE CONCS and not just change in Ozone (Not done above as first run should have scaling factor of zero and is ignored)                   
            print( 'Add year 2010 Parameterised baseline concentrations to get total ozone changes')
            # 3D Data
            no_vals                     = np.where(o3_3d_data_SR1_all_reg_ms.mask) # find where no values in base scenario
            o3_3d_data_all_reg[no_vals] = float('nan') # Mask out grid squares where no data and replace with nan
            o3_3d_data_all_reg_tot      = o3_3d_data_all_reg + o3_3d_data_SR1_all_reg_ms.filled(float('nan')) # add on baseline values to O3 response
    
            # Surface Data
            srf_o3_data_reg_tot[srf_o3_data_SR1_reg != -999.0] = srf_o3_data_reg[srf_o3_data_SR1_reg != -999.0] + srf_o3_data_SR1_reg[srf_o3_data_SR1_reg != -999.0] # add on baseline values to O3 response
            srf_o3_data_reg_tot[srf_o3_data_SR1_reg == -999.0] = float('nan') # replace -999.0 with Nan
            
            ############################################
            
            # Collate Individual model response for each year back into overall Master arrays 
            print( 'Collate Scaled Ozone response for {} model into master array'.format(cur_mod_short))
            o3_3d_reg_all_yrs_all_mod_H1[imod,iyr,:,:,:,:,:]        = o3_3d_data_all_reg[:,:,:,:,:]
            o3_srf_reg_all_yrs_all_mod_H1[imod,iyr,:,:,:,:]         = srf_o3_data_reg[:,:,:,:]
            o3_3d_reg_tot_all_yrs_all_mod_H1[imod,iyr,:,:,:,:,:]    = o3_3d_data_all_reg_tot[:,:,:,:,:]
            o3_srf_reg_tot_all_yrs_all_mod_H1[imod,iyr,:,:,:,:]     = srf_o3_data_reg_tot[:,:,:,:]
            
            ############################################
            
            # OUTPUT individual HTAP1 model responses to netcdf file
            if IOUT_NCF_INDIVID_H1 == 1:
                # Output Individual-model global mean values to a netcdf file 
                out_fname = '{}_HTAP-I_model_mean_{}_response_in_{}_to_{}_change.nc'.format(cur_mod,SPEC,year,EMIS_SCN)
                print( 'Output Individual HTAP1 model {} Scaled Ozone fields to Netcdf file {}'.format(cur_mod,out_fname))
                out_fname_path = OUT_FILE_PATH+out_fname
                ht1_fn.output_file_mod_H1(out_fname_path,cur_mod,EMIS_SCN,nlevs,nlons,nlats,ntime,NREGS_H1,levs,lons,lats,time,['CH4']+H1_SRC_REGIONS,year,
                        o3_3d_data_all_reg,srf_o3_data_reg,o3_3d_data_all_reg_tot,srf_o3_data_reg_tot)
            
            ############################################
            
            # CALCULATE REGIONAL SURFACE CHANGES DUE TO REGIONAL EMISSION PERTURBATIONS FOR INDIVIDUAL MODEL RESPONSES TO VERIFY OUTPUT
            if (CALC_REG_HTAP2_H1 == 1):
            
            ######### NEED TO WAIT UNTIL ADDED TOGEHER HTAP1 AND HTAP2 CONTRIBUTIONS TO GET PROPER REGIONAL MEANS #####   
                # For first model to process calculate global mean area
                if imod == 0: 
                    print( 'Calculate grid box areas')
                    area2d_global = ht_fn.calc_area_any(nlats,nlons,lats,lons)
                               
                print( 'Calculate Annual Mean O3 response values over HTAP2 receptor regions for HTAP1 Models for emission change scenario')
                # Calc reg changes doese calculation to include number of models as well now
                print( 'Calculate Regional Changes for model {} in {}'.format(cur_mod_short,year))
                
                for (ireg_h1,reg_h1) in enumerate(['CH4']+H1_SRC_REGIONS):
                    # O3 response regional mean vlaues
                    reg_mean_vals_yr                        = ht_fn.calc_H2_reg_response(ntime,srf_o3_data_reg[:,:,:,ireg_h1],area2d_global,NREGS_H2_OUT,HTAP2_RECP_REGS)
                    reg_mean_vals_H1[imod,iyr,ireg_h1,:]    = reg_mean_vals_yr[:]
                    # Calculate Total O3 response values
                    reg_mean_vals_act_yr                    = ht_fn.calc_H2_reg_response(ntime,srf_o3_data_reg_tot[:,:,:,ireg_h1],area2d_global,NREGS_H2_OUT,HTAP2_RECP_REGS)
                    reg_mean_vals_act_H1[imod,iyr,ireg_h1,:]= reg_mean_vals_act_yr[:]
                    
                    # CALCULATE GLOBAL MEAN VALUES
                    print( 'Calculate Global Mean Responses')
                    global_mean_val_H1[imod,iyr,ireg_h1]    = ht_fn.calc_glob_mean(srf_o3_data_reg[:,:,:,ireg_h1],area2d_global)
                    global_mean_val_act_H1[imod,iyr,ireg_h1]= ht_fn.calc_glob_mean(srf_o3_data_reg_tot[:,:,:,ireg_h1],area2d_global)
                    print( 'Global annual mean change in response to {} emission decrease = {:.3f}'.format(reg_h1,global_mean_val_H1[imod,iyr,ireg_h1]))
                    
                    ############################################
                    # Calculate burden and Radiative Forcing for each H1 source region of current model
                    print( 'Calculate Annual mean Global Total Column Burden for model {}'.format(cur_mod_short))
                    # calculate annual mean Global Total column burden for each model to use to find Variance in HTAP1 for RF error         
                    o3_mod_burd, mod_burd_dob, mod_rf_glob, mod_burd_htap2_regs, mod_rf_htap2_regs = ht_fn.calc_burd_ACCMIP(o3_3d_data_all_reg[:,:,:,:,ireg_h1],o3_3d_data_SR1,dp_3d,grid_areas_mon_3d,accmip_2d_rf,NREGS_H2_OUT,HTAP2_RECP_REGS)
                    
                    o3_burd_all_mod_H1[imod,iyr,ireg_h1]       = o3_mod_burd
                    rf_glob_all_mod_H1[imod,iyr,ireg_h1]       = mod_rf_glob
                    reg_o3_burd_all_mod_H1[imod,iyr,ireg_h1,:] = mod_burd_htap2_regs[:]
                    reg_o3_rf_all_mod_H1[imod,iyr,ireg_h1,:]   = mod_rf_htap2_regs[:]
                    
                    if IPRINT_REG == 1:
                        for ireg_2,values in enumerate(sorted(HTAP2_RECP_REGS)):
                            print( 'Area weighted annual mean change for HTAP 2 region {} in response to {} emission decrease = {:.3f}'.format(values,reg_h1,reg_mean_vals_H1[imod,iyr,ireg_h1,ireg_2]))
                            print( 'Area weighted total annual mean concentration for HTAP 2 region {} in response to {} emission decrease = {:.2f}'.format(values,reg_h1,reg_mean_vals_act_H1[imod,iyr,ireg_h1,ireg_2]))
            
        #########################################
        
        if IOUT_TXT_INDIVID_H1 ==1:
            # Write out text file for inidividual model responses to an emission change
            print( 'Write out regional changes to file for HTAP1 model {}'.format(cur_mod_short))
            header_out          = ['Year','Emis_scn','CAM','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
            header_out_str      = ','.join(header_out)
            out_fname_act_conc  = OUT_TXT_FILE_PATH+'individ_mods/'+cur_mod_short+'_H1_model_regional_average_'+SPEC+'_concentrations_to_each_emis_pert_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
            out_fname_resp      = OUT_TXT_FILE_PATH+'individ_mods/'+cur_mod_short+'_H1_model_regional_RESPONSE_in_'+SPEC+'_concentrations_to_each_emis_pert_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
            # Output text file of regional means
            ht_fn.output_txt_file_reg_individ_mod(out_fname_resp,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_mean_vals_H1,global_mean_val_H1,imod)
            ht_fn.output_txt_file_reg_individ_mod(out_fname_act_conc,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_mean_vals_act_H1,global_mean_val_act_H1,imod)
            print( '#### Finished Writing Regional Response for HTAP1 Individual Model {} ####'.format(cur_mod_short))
            
    ############################################
    # Calculate multi_model mean values and also standard deviation of results for both surf, 3D and regional mean values
    #### THESE ARE MMM FOR TOTAL O3 RESPONSE TO EUR, NAM, EAS AND SAS EMISSION CHANGE (H1 MODELS)
    if IOUT_CALC_MMM_H1 == 1:
        print( 'Calculate Multi-model Mean and Standard Deviations for HTAP1 Models across Regions {}'.format(['CH4']+H1_SRC_REGIONS))
        
        print( 'Calculate HTAP1 Multi-Model output arrays')
        print( o3_3d_reg_all_yrs_all_mod_H1.shape)
        o3_3d_reg_all_yrs_H1_mmm        = np.nanmean(o3_3d_reg_all_yrs_all_mod_H1, axis=0)
        o3_srf_reg_all_yrs_H1_mmm       = np.nanmean(o3_srf_reg_all_yrs_all_mod_H1, axis=0)
        o3_3d_reg_tot_all_yrs_H1_mmm    = np.nanmean(o3_3d_reg_tot_all_yrs_all_mod_H1, axis=0)
        o3_srf_reg_tot_all_yrs_H1_mmm   = np.nanmean(o3_srf_reg_tot_all_yrs_all_mod_H1, axis=0)
        
        # Calculate standard deviation of multi-model mean
        o3_3d_reg_all_yrs_H1_mmm_stdev      = np.nanstd(o3_3d_reg_all_yrs_all_mod_H1, axis=0)
        o3_srf_reg_all_yrs_H1_mmm_stdev     = np.nanstd(o3_srf_reg_all_yrs_all_mod_H1, axis=0)
        o3_3d_reg_tot_all_yrs_H1_mmm_stdev  = np.nanstd(o3_3d_reg_tot_all_yrs_all_mod_H1, axis=0)
        o3_srf_reg_tot_all_yrs_H1_mmm_stdev = np.nanstd(o3_srf_reg_tot_all_yrs_all_mod_H1, axis=0)
        
        # Separate out O3 response to Methane from HTAP1 models to combined later on with those from HTAP2 models
        print( 'Separate out Methane responses for H1 model')
        #print np.min(o3_srf_reg_all_yrs_all_mod_H1[:,:,:,:,:,0])*1e9, np.mean(o3_srf_reg_all_yrs_all_mod_H1[:,:,:,:,:,0])*1e9, np.max(o3_srf_reg_all_yrs_all_mod_H1[:,:,:,:,:,0])*1e9
        ch4_srf_all_yrs_H1          = o3_srf_reg_all_yrs_all_mod_H1[:,:,:,:,:,0]
        ch4_3d_reg_all_yrs_H1       = o3_3d_reg_all_yrs_all_mod_H1[:,:,:,:,:,:,0]
        global_mean_val_H1_ch4      = global_mean_val_H1[:,:,0]
        global_mean_val_act_H1_ch4  = global_mean_val_act_H1[:,:,0]
        reg_mean_val_H1_ch4         = reg_mean_vals_H1[:,:,0,:]
        reg_mean_val_act_H1_ch4     = reg_mean_vals_act_H1[:,:,0,:]
        o3_burd_glob_H1_ch4         = o3_burd_all_mod_H1[:,:,0]
        o3_rf_glob_H1_ch4           = rf_glob_all_mod_H1[:,:,0]
        o3_burd_reg_H1_ch4          = reg_o3_burd_all_mod_H1[:,:,0,:]
        o3_rf_reg_H1_ch4            = reg_o3_rf_all_mod_H1[:,:,0,:]
        
        print( 'Calculate Variation in Global total column burdens between models')
        o3_burd_all_mod_H1_mmm = np.nanmean(o3_burd_all_mod_H1,axis=0)
        o3_burd_all_mod_H1_sd  = np.nanstd(o3_burd_all_mod_H1, axis=0)
        o3_burd_all_mod_H1_var = np.nanvar(o3_burd_all_mod_H1, axis=0)
        
        rf_glob_all_mod_H1_mmm = np.nanmean(rf_glob_all_mod_H1,axis=0)
        rf_glob_all_mod_H1_sd  = np.nanstd(rf_glob_all_mod_H1,axis=0)
        rf_glob_all_mod_H1_var = np.nanvar(rf_glob_all_mod_H1,axis=0)
        
        reg_o3_burd_all_mod_H1_mmm = np.nanmean(reg_o3_burd_all_mod_H1,axis=0)
        reg_o3_burd_all_mod_H1_sd  = np.nanstd(reg_o3_burd_all_mod_H1, axis=0)
        reg_o3_burd_all_mod_H1_var = np.nanvar(reg_o3_burd_all_mod_H1, axis=0)
        
        reg_o3_rf_all_mod_H1_mmm = np.nanmean(reg_o3_rf_all_mod_H1,axis=0)
        reg_o3_rf_all_mod_H1_sd  = np.nanstd(reg_o3_rf_all_mod_H1,axis=0)
        reg_o3_rf_all_mod_H1_var = np.nanvar(reg_o3_rf_all_mod_H1,axis=0)
        
        ##### OUTPUT FILES LATER ALONG WITH HTAP2 MODEL RESULTS #######
        
        if IOUT_NCF_MMM_H1 == 1: # Only if set for separate NetCDF file
            # Output multi-model global mean values and standard deviations to a netcdf file 
            out_fname = 'Multi_model_mean_H1_plus_stdev_for_{}_response_to_{}_change.nc'.format(SPEC,EMIS_SCN)
            print( 'Output Scaled Ozone fields to Netcdf file {}'.format(out_fname))
            out_fname_path = OUT_FILE_PATH+out_fname
            ht1_fn.output_file_h1_mmm(out_fname_path,'Multi-model',EMIS_SCN,nlevs,nlons,nlats,ntime,NYRS,NREGS_H1,levs,lons,lats,time,['CH4']+H1_SRC_REGIONS,YEARS_SCN,
                                #o3_3d_all_yrs_mmm,o3_srf_all_yrs_mmm,o3_3d_tot_all_yrs_mmm,o3_srf_tot_all_yrs_mmm,
                                o3_3d_reg_all_yrs_H1_mmm,o3_srf_reg_all_yrs_H1_mmm,o3_3d_reg_tot_all_yrs_H1_mmm,o3_srf_reg_tot_all_yrs_H1_mmm,
                                #o3_3d_all_yrs_mmm_stdev,o3_srf_all_yrs_mmm_stdev,o3_3d_tot_all_yrs_mmm_stdev,o3_srf_tot_all_yrs_mmm_stdev,
                                o3_3d_reg_all_yrs_H1_mmm_stdev,o3_srf_reg_all_yrs_H1_mmm_stdev,o3_3d_reg_tot_all_yrs_H1_mmm_stdev,o3_srf_reg_tot_all_yrs_H1_mmm_stdev)
        
        print( 'Calculate Regional HTAP1 Multi-model regional changes (and standard deviations)')
        reg_mean_vals_mmm_H1     = np.nanmean(reg_mean_vals_H1, axis = 0)
        reg_mean_vals_act_mmm_H1 = np.nanmean(reg_mean_vals_act_H1, axis = 0)
        
        reg_mean_vals_mmm_stdev_H1      = np.nanstd(reg_mean_vals_H1, axis=0, dtype=np.float64)
        reg_mean_vals_act_mmm_stdev_H1  = np.nanstd(reg_mean_vals_act_H1,axis=0, dtype=np.float64)
                    
        global_mean_val_mmm_H1           = np.nanmean(global_mean_val_H1, axis=0)
        global_mean_val_act_mmm_H1       = np.nanmean(global_mean_val_act_H1, axis=0)
        global_mean_val_mmm_stdev_H1     = np.nanstd(global_mean_val_H1, axis=0, dtype=np.float64)
        global_mean_val_act_mmm_stdev_H1 = np.nanstd(global_mean_val_act_H1, axis=0, dtype=np.float64)
        
    ############################################
    
    # Regional Multi-model Means
    if  IOUT_TXT_MMM_H1 == 1:
        # Output multi-model regional mean values along with standard deviation to a text file     
        print( 'Write out HTAP1 Multi-model regional changes (and standard deviations) to file')
            
        header_out              = ['Year','Emis_scn','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
        header_out_str          = ','.join(header_out)
        out_fname_act_conc_mmm  = OUT_TXT_FILE_PATH+'Multi-model_H1_regional_average_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        out_fname_resp_mmm      = OUT_TXT_FILE_PATH+'Multi-model_H1_regional_average_RESPONSE_in_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        #out_fname_act_conc_sd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H1_STD_DEV_in_regional_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        out_fname_resp_sd_mmm   = OUT_TXT_FILE_PATH+'Multi-model_H1_STD_DEV_in_regional_average_RESPONSE_in_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        # Output text file of regional means
        ht_fn.output_txt_file_reg_mmm(out_fname_resp_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_mean_vals_mmm_H1,global_mean_val_mmm_H1)
        ht_fn.output_txt_file_reg_mmm(out_fname_act_conc_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_mean_vals_act_mmm_H1,global_mean_val_act_mmm_H1)
        ht_fn.output_txt_file_reg_mmm(out_fname_resp_sd_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_mean_vals_mmm_stdev_H1,global_mean_val_mmm_stdev_H1)
        #ht_fn.output_txt_file_reg_mmm(out_fname_act_conc_sd_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_mean_vals_act_mmm_stdev_H1,global_mean_val_act_mmm_stdev_H1)
        
        #print 'Write out to file HTAP1 multi-model change in Global Total O3 column Burden and RE'
        #header_out = ['Year','Emis_scn','Global Col Burd (Tg)','+/- S.D.','+/- Var','O3 RF W m-2','+/- S.D.','+/- Var']
        #header_out_str = ','.join(header_out)
        #out_fname_o3_burd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H1_global_ann_mean_RESPONSE_'+SPEC+'_col_burd_and_RE_for_'+EMIS_SCN+'_src_regs.txt'
        #ht_fn.output_txt_file_burd_mmm(out_fname_o3_burd_mmm,header_out_str,['2010']+YEARS_SCN,['CH4']+H1_SRC_REGIONS,base_burd,base_rf_glob,
        #                               o3_burd_all_mod_H1_mmm,o3_burd_all_mod_H1_sd,o3_burd_all_mod_H1_var,rf_glob_all_mod_H1_mmm,rf_glob_all_mod_H1_sd,rf_glob_all_mod_H1_var)
        
        print( 'Write out to file HTAP1 multi-model change in REGIONAL Total O3 column Burden and RE')
        #header_out_reg              = ['Year','Emis_scn','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
        #header_out_reg_str          = ','.join(header_out_reg)
        #out_fname_reg_o3_burd_mmm   = OUT_TXT_FILE_PATH+'Multi-model_H1_GLOBAL_and_REGIONAL_ann_mean_RESPONSE_'+SPEC+'_col_burd_and_RE_for_'+EMIS_SCN+'_src_regs.txt'
        #ht_fn.output_txt_file_reg_burd_mmm(out_fname_reg_o3_burd_mmm,header_out_reg_str,header_out[2:],['2010']+YEARS_SCN,['CH4']+H1_SRC_REGIONS,base_burd_htap2_regs,base_rf_htap2_regs,
        #                                   reg_o3_burd_all_mod_H1_mmm,reg_o3_burd_all_mod_H1_sd,reg_o3_burd_all_mod_H1_var,reg_o3_rf_all_mod_H1_mmm,reg_o3_rf_all_mod_H1_sd,reg_o3_rf_all_mod_H1_var,
        #                                   base_burd,base_rf_glob,o3_burd_all_mod_H1_mmm,o3_burd_all_mod_H1_sd,o3_burd_all_mod_H1_var,rf_glob_all_mod_H1_mmm,rf_glob_all_mod_H1_sd,rf_glob_all_mod_H1_var)
        
        out_fname_reg_o3_burd_mmm    = OUT_TXT_FILE_PATH+'Multi-model_H1_regional_average_RESPONSE_'+SPEC+'_col_burd_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        out_fname_reg_o3_rf_mmm      = OUT_TXT_FILE_PATH+'Multi-model_H1_regional_average_RESPONSE_'+SPEC+'_Radiative_Forcing_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        out_fname_reg_o3_burd_sd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H1_STD_DEV_in_regional_average_RESPONSE_'+SPEC+'_col_burd_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        out_fname_reg_o3_rf_sd_mmm   = OUT_TXT_FILE_PATH+'Multi-model_H1_STD_DEV_in_regional_average_RESPONSE_'+SPEC+'_Radiative_Forcing_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_burd_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_o3_burd_all_mod_H1_mmm,o3_burd_all_mod_H1_mmm)
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_rf_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_o3_rf_all_mod_H1_mmm,rf_glob_all_mod_H1_mmm)
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_burd_sd_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_o3_burd_all_mod_H1_sd,o3_burd_all_mod_H1_sd)
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_rf_sd_mmm,header_out_str,YEARS_SCN,['CH4']+H1_SRC_REGIONS,reg_o3_rf_all_mod_H1_sd,rf_glob_all_mod_H1_sd)
        
        
        print( '#### Finished Writing Regional Response for HTAP1 Multi Model Mean Values ####')
    
    ################################
    ################################  
    
    # PROCESS HTAP2 MODELS
    # Process HTAP-2 Scenarios and individual models (including separate one for Methane)
    print( '###### PROCESS HTAP2 MODELS FOR ADDITIONAL SCENARIOS ########')
    
    # INITIALISE OUTPUT ARRAYS
    print( 'Initialise Output Arrays to collect all data from HTAP2 models')
    # Initialise Scaled Output Arrays to contain data for all HTAP-2 models
    # As these are regional scenario based then need to collate on regions before totalling up
    o3_3d_reg_all_yrs_MMM_H2        = np.zeros((NYRS,ntime,nlevs,nlats,nlons,NREGS_H2), dtype='f')
    o3_srf_reg_all_yrs_MMM_H2       = np.zeros((NYRS,ntime,nlats,nlons,NREGS_H2), dtype='f')
    o3_3d_reg_tot_all_yrs_MMM_H2    = np.zeros((NYRS,ntime,nlevs,nlats,nlons,NREGS_H2), dtype='f')
    o3_srf_reg_tot_all_yrs_MMM_H2   = np.zeros((NYRS,ntime,nlats,nlons,NREGS_H2), dtype='f')
    
    o3_3d_reg_all_yrs_MMM_STDEV_H2      = np.zeros((NYRS,ntime,nlevs,nlats,nlons,NREGS_H2), dtype='f')
    o3_srf_reg_all_yrs_MMM_STDEV_H2     = np.zeros((NYRS,ntime,nlats,nlons,NREGS_H2), dtype='f')
    o3_3d_reg_tot_all_yrs_MMM_STDEV_H2  = np.zeros((NYRS,ntime,nlevs,nlats,nlons,NREGS_H2), dtype='f')
    o3_srf_reg_tot_all_yrs_MMM_STDEV_H2 = np.zeros((NYRS,ntime,nlats,nlons,NREGS_H2), dtype='f')
    
    # Initialise Regional mean value array
    reg_mean_vals_mmm_H2             = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_mean_vals_act_mmm_H2         = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_mean_vals_mmm_stdev_H2       = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_mean_vals_act_mmm_stdev_H2   = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    global_mean_val_mmm_H2           = np.zeros((NYRS,NREGS_H2), dtype='f')
    global_mean_val_act_mmm_H2       = np.zeros((NYRS,NREGS_H2), dtype='f')
    global_mean_val_mmm_stdev_H2     = np.zeros((NYRS,NREGS_H2), dtype='f')
    global_mean_val_act_mmm_stdev_H2 = np.zeros((NYRS,NREGS_H2), dtype='f')
    
    # Global burden, dobs and RF arrays
    o3_burd_all_mod_H2_mmm = np.zeros((NYRS,NREGS_H2), dtype='f')
    o3_burd_all_mod_H2_sd  = np.zeros((NYRS,NREGS_H2), dtype='f')
    o3_burd_all_mod_H2_var = np.zeros((NYRS,NREGS_H2), dtype='f')
    rf_glob_all_mod_H2_mmm = np.zeros((NYRS,NREGS_H2), dtype='f')
    rf_glob_all_mod_H2_sd  = np.zeros((NYRS,NREGS_H2), dtype='f')
    rf_glob_all_mod_H2_var = np.zeros((NYRS,NREGS_H2), dtype='f')
    
    reg_o3_burd_all_mod_H2_mmm = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_o3_burd_all_mod_H2_sd  = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_o3_burd_all_mod_H2_var = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    
    reg_o3_rf_all_mod_H2_mmm = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_o3_rf_all_mod_H2_sd  = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    reg_o3_rf_all_mod_H2_var = np.zeros((NYRS,NREGS_H2,NREGS_H2_OUT), dtype='f')
    
    print( H2_SCN_ID)
    # Process each HTAP2 model for each regional emission perturbation Scenario
    # SCENARIOS - HTAP2 regional Emission perturbation Sccenario order is 
    #['CH4ALL', 'MDEALL', 'RBUALL', 'NAFALL', 'MCAALL', 'SAMALL', 'CASALL', 'SAFALL', 'PANALL', 'SEAALL', 'OCNALL']#
    for (iscn,H2_scn) in enumerate(H2_SCN_ID):
        print( 'Process HTAP2 Models for Scenario {}'.format(H2_scn))
    
        MODEL_FNAMES_H2    =  glob.glob(MOD_FILE_PATH_H2+H2_scn+'/contributions/*.nc')
        NMODS_H2_SCN = len(MODEL_FNAMES_H2)
        print( 'List of HTAP-2 Models to Process for {} scenario'.format(H2_scn))
        print( MODEL_FNAMES_H2)
        
        # Initialise new arrays to store the output scaled Ozone responses for each model in
        print( 'Initialise Output arrays to collect data for each HTAP2 Regional Perturbation Scenario')
        # Individual regional contribution response fields
        srf_o3_data_reg_H2          = np.zeros((NMODS_H2_SCN,NYRS,ntime,nlats,nlons), dtype='f') # individual regional contributions from all species and for CH4 separately
        o3_3d_data_all_reg_H2       = np.zeros((NMODS_H2_SCN,NYRS,ntime,nlevs,nlats,nlons), dtype='f')
        srf_o3_data_reg_tot_H2      = np.zeros((NMODS_H2_SCN,NYRS,ntime,nlats,nlons), dtype='f')
        o3_3d_data_all_reg_tot_H2   = np.zeros((NMODS_H2_SCN,NYRS,ntime,nlevs,nlats,nlons), dtype='f')
        
        # Global and Regional mean values for all models for each HTAP2 scenario
        global_mean_val_H2_scn      = np.zeros((NMODS_H2_SCN,NYRS),dtype='f')
        global_mean_val_act_H2_scn  = np.zeros((NMODS_H2_SCN,NYRS),dtype='f')
        reg_mean_vals_H2_scn        = np.zeros((NMODS_H2_SCN,NYRS,NREGS_H2_OUT),dtype='f')
        reg_mean_vals_act_H2_scn    = np.zeros((NMODS_H2_SCN,NYRS,NREGS_H2_OUT),dtype='f')
        
        o3_burd_all_mod_H2 = np.zeros((NMODS_H2_SCN,NYRS),dtype='f')
        rf_glob_all_mod_H2 = np.zeros((NMODS_H2_SCN,NYRS),dtype='f')
        
        reg_o3_burd_all_mod_H2  = np.zeros((NMODS_H2_SCN,NYRS,NREGS_H2_OUT),dtype='f')
        reg_o3_rf_all_mod_H2    = np.zeros((NMODS_H2_SCN,NYRS,NREGS_H2_OUT),dtype='f')
        
        ################################
        
        # MODELS - Process each HTAP2 model file in turn for each regional scenario response
        for (imod_2,mod_file_2) in enumerate(MODEL_FNAMES_H2):
            # setting not for Rest of World field now as not using them but to identify any missing values in data that need to be filled with multi-model mean
            no_rw = 0 
        
            file_txt = mod_file_2.split('/')
            cur_mod_2 = '_'.join(file_txt[8:9])
            cur_mod_short_txt = cur_mod_2.split('_')
            cur_mod_2_short = '_'.join(cur_mod_short_txt[0:1])
            add_scn = '_'.join(cur_mod_short_txt[1:2])
            print( 'Calculate {} Response to emission scenario {} from HTAP-2 model {} for {}'.format(SPEC,EMIS_SCN,cur_mod_2_short,add_scn))
            
            ################################
            
            # YEARS - Calculate changes for each future year of scenario
            for (iyr,year) in enumerate(YEARS_SCN):
                print( 'Calculate Emission changes for {}'.format(year))
                
                # Take each point individually from the ozone change of each scenario 
                # (base - 20% reduction for each species and region) and then apply relevant scaling factor 
                # for this emission change. Total up to get total ozone response
                # also need to loop through all model runs done and apply scaling factor in order
                
                ################################
                
                # For Methane Scenario
                if (H2_scn == 'CH4ALL'):
                    print( iyr,iscn)
                    # CH4 has no contributions from CO, NOX and VOCs so just process scaling of data once
                    run_2 = 'ch4_resp'
                    print( 'Processing scenario {}'.format(H2_scn))
                    print( 'Read in HTAP-2 Model data for scenario {} and {}'.format(H2_scn,'ch4_resp'))
                    # read in HTAP2 scenario data
                    lats,lons,levs,time,srf_o3_data_H2,o3_3d_data_H2 = ht2_fn.read_H2_model_data(mod_file_2,'ch4_resp','srf','3d',SPEC)
                        
                    # Test if surface ozone field is numpy array or masked array (i.e. missing values)
                    if isinstance(srf_o3_data_H2, (np.ma.core.MaskedArray)):
                        srf_o3_data_H2 = srf_o3_data_H2.filled() # copy of masked array and makes numpy array with masked values filled with -999.0
                        no_rw = 1 # If surface is masked array then has missing values and need to use rest of the world to fill in
                    
                    # Test if 3D array is a numpy array as should be a masked array (EMEP model 3D files are numpy arrays)
                    if isinstance(o3_3d_data_H2, (np.ndarray)):
                        print( '3D array is only a numpy array so create a masked array')
                        mask_arr = np.isnan(o3_3d_data_H2) # create mask of boolean values for where points are NaN
                        o3_3d_data_H2 = ma.masked_array(o3_3d_data_H2,mask=mask_arr)
                        
                    print( 'Apply future emission Scaling factor to individual model HTAP-I O3 response for scenario {}'.format(H2_scn))
                    # Scaling factors for methane are all in iyr,0,0 point of array (CH4 inc. as source region)        
                    srf_o3_data_sing_reg_H2,o3_3d_data_sing_reg_H2 = ht2_fn.calc_scal_data_H2(emis_scal_f_h2[iyr,0,iscn],emis_scal_g_h2[iyr,0,iscn],run_2,
                                                                                              srf_o3_data_H2,o3_3d_data_H2,ntime,nlevs,nlats,nlons)
                            
                    # Add Single emission contribution (CH4) into arrays that collate total response for particular regional scenario 
                    srf_o3_data_reg_H2[imod_2,iyr,:,:,:]        += srf_o3_data_sing_reg_H2[:,:,:]
                    o3_3d_data_all_reg_H2[imod_2,iyr,:,:,:,:]   += o3_3d_data_sing_reg_H2[:,:,:,:] 
                
                # For all other HTAP2 regional scenarios
                else:
                    #loop through each precursor emission for current scenario of ozone changes (base - 20% reduction)
                    for (irun_2,run_2) in enumerate(H2_RUNID): 
                        print( iyr,irun_2,iscn)
                        print( 'Processing scenario {}'.format(H2_scn))
                        # Load in surface and 3D Ozone reponse field for each scenario and for each model in order  
                        print( 'Read in HTAP-2 Model data for scenario {} and {}'.format(H2_scn,run_2))
                        lats,lons,levs,time,srf_o3_data_H2,o3_3d_data_H2 = ht2_fn.read_H2_model_data(mod_file_2,run_2,'srf','3d',SPEC)
                        
                        # Test if surface ozone field is numpy array or masked array (i.e. missing values)
                        if isinstance(srf_o3_data_H2, (np.ma.core.MaskedArray)):
                            print( 'Surface array is masked so change to numpy')
                            srf_o3_data_H2 = srf_o3_data_H2.filled() # copy of masked array and makes numpy array with masked values filled with -999.0
                            no_rw = 1 # If surface is masked array then has missing values and need to use rest of the world to fill in
                        
                        # Test if 3D array is a numpy array as should be a masked array (EMEP model 3D files are numpy arrays)
                        if isinstance(o3_3d_data_H2, (np.ndarray)):
                            print( '3D array is only a numpy array so create a masked array')
                            mask_arr = np.isnan(o3_3d_data_H2) # create mask of boolean values for where points are NaN
                            o3_3d_data_H2 = ma.masked_array(o3_3d_data_H2,mask=mask_arr)
                         
                        print('Apply future emission Scaling factor to individual model HTAP-I O3 response for scenario {}'.format(run_2))
                        print( emis_scal_f_h2[iyr,irun_2,iscn],emis_scal_g_h2[iyr,irun_2,iscn])
                        # Emission factors for NOX,VOC,CO all in iyr,:,iscn  
                        srf_o3_data_sing_reg_H2,o3_3d_data_sing_reg_H2 = ht2_fn.calc_scal_data_H2(emis_scal_f_h2[iyr,irun_2,iscn],emis_scal_g_h2[iyr,irun_2,iscn],run_2,
                                                                                                  srf_o3_data_H2,o3_3d_data_H2,ntime,nlevs,nlats,nlons)
                        
                        # Add Single emission contribution (NOx,VOC,CO) into total array for regional scenario (but += for total of NOX, CO and VOC emis)
                        srf_o3_data_reg_H2[imod_2,iyr,:,:,:]        += srf_o3_data_sing_reg_H2[:,:,:]
                        o3_3d_data_all_reg_H2[imod_2,iyr,:,:,:,:]   += o3_3d_data_sing_reg_H2[:,:,:,:] 
                                         
                    no_rw = 0 # rest of world flag for next model
                
                #############################################
                
                print( 'Calculate Annual mean Global Total Column Burden for model {}'.format(cur_mod_short))
                # calculate annual mean Global Total column burden for each model to use to find Variance in HTAP1 for RF error         
                o3_mod_burd, mod_burd_dob, mod_rf_glob, mod_burd_htap2_regs, mod_rf_htap2_regs = ht_fn.calc_burd_ACCMIP(o3_3d_data_all_reg_H2[imod_2,iyr,:,:,:,:],o3_3d_data_SR1,dp_3d,grid_areas_mon_3d,accmip_2d_rf,NREGS_H2_OUT,HTAP2_RECP_REGS)
                
                o3_burd_all_mod_H2[imod_2,iyr]      = o3_mod_burd
                rf_glob_all_mod_H2[imod_2,iyr]      = mod_rf_glob
                reg_o3_burd_all_mod_H2[imod_2,iyr,:]= mod_burd_htap2_regs[:]
                reg_o3_rf_all_mod_H2[imod_2,iyr,:]  = mod_rf_htap2_regs[:]
                
                ################################    
                
                # NEED TO ADD BACK ON BASE 2010 CONCENTRATIONS TO GET NEW OZONE CONCS and not just change in Ozone (Not done above as first run should have scaling factor of zero and is ignored)                   
                print( 'Add year 2010 Parameterised baseline concentrations to get total ozone changes')
                
                # 3D DATA
                ### need to change shape of baseline array to fit (mod,yr,time,lat,lon) whereas currently (time,lat,lon,H1_reg)
                ## broadcast standard array into extra dimenstions?
                no_vals                                         = np.where(o3_3d_data_SR1.mask)  # find where no values in base scenario
                o3_3d_data_all_reg_H2_tmp                       = o3_3d_data_all_reg_H2[imod_2,iyr,:,:,:,:]
                o3_3d_data_all_reg_H2_tmp[no_vals]              = float('nan') # Mask out grid squares where no data and replace with nan
                o3_3d_data_all_reg_H2[imod_2,iyr,:,:,:,:]       = o3_3d_data_all_reg_H2_tmp[:,:,:,:] # Mask out grid squares where no data in origianl array
                o3_3d_data_all_reg_tot_H2[imod_2,iyr,:,:,:,:]   = o3_3d_data_all_reg_H2[imod_2,iyr,:,:,:,:] + o3_3d_data_SR1.filled(float('nan')) # add on baseline values to O3 response
                
                # Surface Data
                srf_o3_data_reg_H2_tmp                   = srf_o3_data_reg_H2[imod_2,iyr,:,:,:] 
                tmp_arr_tot                              = np.zeros((ntime,nlats,nlons), dtype='f')
                tmp_arr_tot[srf_o3_data_SR1 == -999.0]   = float('nan') # replace -999.0 with Nan
                tmp_arr_tot[srf_o3_data_SR1 != -999.0]   = srf_o3_data_reg_H2_tmp[srf_o3_data_SR1 != -999.0] + srf_o3_data_SR1[srf_o3_data_SR1 != -999.0] # add on baseline values to O3 response
                srf_o3_data_reg_tot_H2[imod_2,iyr,:,:,:] = tmp_arr_tot[:,:,:]
                
                ############################################
                
                # CALCULATE REGIONAL SURFACE CHANGES DUE TO REGIONAL EMISSION PERTURBATIONS FOR INDIVIDUAL MODEL RESPONSES 
                if (CALC_REG_HTAP2_H2 == 1):
                        
                    # Calculate Global mean area for first time
                    if iscn == 0: 
                        print( 'Calculate grid box areas')
                        area2d_global = ht_fn.calc_area_any(nlats,nlons,lats,lons)
                        
                    # CALCULATE GLOBAL MEAN VALUES
                    print( 'Calculate Global Mean Response to current regional perturbation from each model')
                    global_mean_val_H2_scn[imod_2,iyr]      = ht_fn.calc_glob_mean(srf_o3_data_reg_H2[imod_2,iyr,:,:,:],area2d_global)
                    global_mean_val_act_H2_scn[imod_2,iyr]  = ht_fn.calc_glob_mean(srf_o3_data_reg_tot_H2[imod_2,iyr,:,:,:],area2d_global)
                      
                    print( 'Calculate Annual Mean O3 response values for HTAP2 Models across RELEVANT HTAP2 receptor regions')
                    # Calc reg changes doese calculation to include number of models as well now
                    print( 'Calculate Regional Changes for model {} in {}'.format(cur_mod_2_short,year))
                    
                    # O3 response regional mean vlaues
                    reg_mean_vals_yr                    = ht_fn.calc_H2_reg_response(ntime,srf_o3_data_reg_H2[imod_2,iyr,:,:,:],area2d_global,NREGS_H2_OUT,HTAP2_RECP_REGS)
                    reg_mean_vals_H2_scn[imod_2,iyr,:]  = reg_mean_vals_yr[:]
                    # Calculate Total O3 response values
                    reg_mean_vals_act_yr                    = ht_fn.calc_H2_reg_response(ntime,srf_o3_data_reg_tot_H2[imod_2,iyr,:,:,:],area2d_global,NREGS_H2_OUT,HTAP2_RECP_REGS)
                    reg_mean_vals_act_H2_scn[imod_2,iyr,:]  = reg_mean_vals_act_yr[:]
                    
                    if IPRINT_REG == 1:
                        for ireg_2,values in enumerate(sorted(HTAP2_RECP_REGS)): 
                            print( 'Area weighted annual mean change for HTAP 2 region {} = {:.3f}'.format(values,reg_mean_vals_H2_scn[imod_2,iyr,ireg_2]))
                            print( 'Area weighted total annual mean concentration for HTAP 2 region {} = {:.2f}'.format(values,reg_mean_vals_act_H2_scn[imod_2,iyr,ireg_2]))
                
            #########################################
            
            if IOUT_TXT_INDIVID_H2 ==1:
                # Write out text file for inidividual model responses to an emission change
                print( 'Write out regional changes to file for HTAP2 model {} in scenario {}'.format(cur_mod_2_short,H2_scn))
                header_out          = ['Year','Emis_scn','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
                header_out_str      = ','.join(header_out)
                out_fname_act_conc  = OUT_TXT_FILE_PATH+'individ_mods/'+cur_mod_2_short+'_'+add_scn+'_H2_model_regional_average_'+SPEC+'_concentrations_from_'+H2_scn+'_reg_emis_scenario_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
                out_fname_resp      = OUT_TXT_FILE_PATH+'individ_mods/'+cur_mod_2_short+'_'+add_scn+'_H2_model_regional_RESPONSE_in_'+SPEC+'_concentrations_from_'+H2_scn+'_reg_emis_scenario_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
                # Output text file of regional means
                ht2_fn.output_txt_file_reg_individ_mod_h2(out_fname_resp,header_out_str,YEARS_SCN,H2_scn,reg_mean_vals_H2_scn,global_mean_val_H2_scn,imod_2)
                ht2_fn.output_txt_file_reg_individ_mod_h2(out_fname_act_conc,header_out_str,YEARS_SCN,H2_scn,reg_mean_vals_act_H2_scn,global_mean_val_act_H2_scn,imod_2)
                print( '#### Finished Writing Regional Response for HTAP2 Individual Model {} for scenario {} ####'.format(cur_mod_2_short,H2_scn))
                
        ############################################
        
        # Collate Individual model response for each year back into overall Master arrays for all scenarios 
        #### THESE ARE MMM FOR TOTAL O3 RESPONSE TO HTAP2 SOURCE REGION EMISSION CHANGE (H2 MODELS)
        if IOUT_CALC_MMM_H2 == 1:
            #if IOUT_NCF_MMM_H2 == 1:
            print( 'Collate Scaled Ozone response for scenario {} into master array by calculating MMM and S.D.'.format(H2_scn))
            o3_3d_reg_all_yrs_MMM_H2[:,:,:,:,:,iscn]    = np.nanmean(o3_3d_data_all_reg_H2, axis=0)
            o3_srf_reg_all_yrs_MMM_H2[:,:,:,:,iscn]     = np.nanmean(srf_o3_data_reg_H2, axis=0)
            o3_3d_reg_tot_all_yrs_MMM_H2[:,:,:,:,:,iscn]= np.nanmean(o3_3d_data_all_reg_tot_H2, axis=0)
            o3_srf_reg_tot_all_yrs_MMM_H2[:,:,:,:,iscn] = np.nanmean(srf_o3_data_reg_tot_H2, axis=0)
            
            if iscn == 0:
                print( 'Separate our Methane Responses for CH4DEC')
                ch4_srf_all_yrs_H2          = srf_o3_data_reg_H2[:,:,:,:,:]
                global_mean_val_H2_ch4      = global_mean_val_H2_scn[:,:]
                global_mean_val_act_H2_ch4  = global_mean_val_act_H2_scn[:,:]
                reg_mean_val_H2_ch4         = reg_mean_vals_H2_scn[:,:,:]
                reg_mean_val_act_H2_ch4     = reg_mean_vals_act_H2_scn[:,:,:]
                ch4_3d_reg_all_yrs_H2       = o3_3d_data_all_reg_H2[:,:,:,:,:,:]
                o3_burd_glob_H2_ch4         = o3_burd_all_mod_H2[:,:]
                o3_rf_glob_H2_ch4           = rf_glob_all_mod_H2[:,:]
                o3_burd_reg_H2_ch4          = reg_o3_burd_all_mod_H2[:,:,:]
                o3_rf_reg_H2_ch4            = reg_o3_rf_all_mod_H2[:,:,:]
                
            o3_3d_reg_all_yrs_MMM_STDEV_H2[:,:,:,:,:,iscn]      = np.nanstd(o3_3d_data_all_reg_H2, axis=0)
            o3_srf_reg_all_yrs_MMM_STDEV_H2[:,:,:,:,iscn]       = np.nanstd(srf_o3_data_reg_H2, axis=0)
            o3_3d_reg_tot_all_yrs_MMM_STDEV_H2[:,:,:,:,:,iscn]  = np.nanstd(o3_3d_data_all_reg_tot_H2, axis=0)
            o3_srf_reg_tot_all_yrs_MMM_STDEV_H2[:,:,:,:,iscn]   = np.nanstd(srf_o3_data_reg_tot_H2, axis=0)
            # arrays are now (yr,time,lat,lon,src_reg)
        
            ############################################
            
            # Regional Multi-model Means
            # Calculate multi-model regional mean values along with standard deviation     
            print( 'Calculate Regional HTAP2 Multi-model regional changes (and standard deviations)')
            #(NYRS,NREGS_H2,NREGS_H2_OUT)
            
            reg_mean_vals_mmm_H2[:,iscn,:]           = np.nanmean(reg_mean_vals_H2_scn, axis = 0)
            reg_mean_vals_act_mmm_H2[:,iscn,:]       = np.nanmean(reg_mean_vals_act_H2_scn, axis = 0)
            reg_mean_vals_mmm_stdev_H2[:,iscn,:]     = np.nanstd(reg_mean_vals_H2_scn, axis=0, dtype=np.float64)
            reg_mean_vals_act_mmm_stdev_H2[:,iscn,:] = np.nanstd(reg_mean_vals_act_H2_scn,axis=0, dtype=np.float64)
                
            global_mean_val_mmm_H2[:,iscn]           = np.nanmean(global_mean_val_H2_scn, axis=0)
            global_mean_val_act_mmm_H2[:,iscn]       = np.nanmean(global_mean_val_act_H2_scn, axis=0)
            global_mean_val_mmm_stdev_H2[:,iscn]     = np.nanstd(global_mean_val_H2_scn, axis=0, dtype=np.float64)
            global_mean_val_act_mmm_stdev_H2[:,iscn] = np.nanstd(global_mean_val_act_H2_scn, axis=0, dtype=np.float64)
            
            print( 'Calculate Variation in Global total column burdens between models')
            o3_burd_all_mod_H2_mmm[:,iscn] = np.nanmean(o3_burd_all_mod_H2,axis=0)
            o3_burd_all_mod_H2_sd[:,iscn]  = np.nanstd(o3_burd_all_mod_H2, axis=0)
            o3_burd_all_mod_H2_var[:,iscn] = np.nanvar(o3_burd_all_mod_H2, axis=0)
        
            rf_glob_all_mod_H2_mmm[:,iscn] = np.nanmean(rf_glob_all_mod_H2,axis=0)
            rf_glob_all_mod_H2_sd[:,iscn]  = np.nanstd(rf_glob_all_mod_H2,axis=0)
            rf_glob_all_mod_H2_var[:,iscn] = np.nanvar(rf_glob_all_mod_H2,axis=0)
            
            reg_o3_burd_all_mod_H2_mmm[:,iscn,:] = np.nanmean(reg_o3_burd_all_mod_H2,axis=0)
            reg_o3_burd_all_mod_H2_sd[:,iscn,:]  = np.nanstd(reg_o3_burd_all_mod_H2, axis=0)
            reg_o3_burd_all_mod_H2_var[:,iscn,:] = np.nanvar(reg_o3_burd_all_mod_H2, axis=0)
        
            reg_o3_rf_all_mod_H2_mmm[:,iscn,:] = np.nanmean(reg_o3_rf_all_mod_H2,axis=0)
            reg_o3_rf_all_mod_H2_sd[:,iscn,:]  = np.nanstd(reg_o3_rf_all_mod_H2,axis=0)
            reg_o3_rf_all_mod_H2_var[:,iscn,:] = np.nanvar(reg_o3_rf_all_mod_H2,axis=0)
    
    ############################################
            
    if IOUT_NCF_MMM_H2 == 1:
        # Output multi-model global mean values and standard deviations to a netcdf file 
        out_fname_mmm_h2 = 'Multi_model_mean_H2_plus_stdev_for_{}_response_to_{}_change.nc'.format(SPEC,EMIS_SCN)
        print( 'Output HTAP2 multi-model scaled Ozone fields to Netcdf file {}'.format(out_fname_mmm_h2))
        out_fname_path_h2_mmm = OUT_FILE_PATH+out_fname_mmm_h2
        ht2_fn.output_file_h2_mmm(out_fname_path_h2_mmm,'Multi-model',EMIS_SCN,nlevs,nlons,nlats,ntime,NYRS,NREGS_H2,levs,lons,lats,time,H2_SRC_REGIONS,YEARS_SCN,
                                #o3_3d_all_yrs_mmm,o3_srf_all_yrs_mmm,o3_3d_tot_all_yrs_mmm,o3_srf_tot_all_yrs_mmm,
                                o3_3d_reg_all_yrs_MMM_H2,o3_srf_reg_all_yrs_MMM_H2,o3_3d_reg_tot_all_yrs_MMM_H2,o3_srf_reg_tot_all_yrs_MMM_H2,
                                #o3_3d_all_yrs_mmm_stdev,o3_srf_all_yrs_mmm_stdev,o3_3d_tot_all_yrs_mmm_stdev,o3_srf_tot_all_yrs_mmm_stdev,
                                o3_3d_reg_all_yrs_MMM_STDEV_H2,o3_srf_reg_all_yrs_MMM_STDEV_H2,o3_3d_reg_tot_all_yrs_MMM_STDEV_H2,o3_srf_reg_tot_all_yrs_MMM_STDEV_H2)
    
    ############################################
            
    if IOUT_TXT_MMM_H2 == 1:
        # Output multi-model regional surface concentration responses at each HTAP2 receptor to each HTAP2 regional emission perturbation scenario            
        print( 'Output regional mean values from HTAP2 multi-model scaled Ozone fields')
        header_out              = ['Year','Emis_scn','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
        header_out_str          = ','.join(header_out)
        out_fname_act_conc_mmm  = OUT_TXT_FILE_PATH+'Multi-model_H2_regional_average_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        out_fname_resp_mmm      = OUT_TXT_FILE_PATH+'Multi-model_H2_regional_average_RESPONSE_in_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        #out_fname_act_conc_sd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H2_STD_DEV_in_regional_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        out_fname_resp_sd_mmm   = OUT_TXT_FILE_PATH+'Multi-model_H2_STD_DEV_in_regional_average_RESPONSE_in_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
        # Output text file of regional means
        ht_fn.output_txt_file_reg_mmm(out_fname_resp_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_mean_vals_mmm_H2,global_mean_val_mmm_H2)
        ht_fn.output_txt_file_reg_mmm(out_fname_act_conc_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_mean_vals_act_mmm_H2,global_mean_val_act_mmm_H2)
        ht_fn.output_txt_file_reg_mmm(out_fname_resp_sd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_mean_vals_mmm_stdev_H2,global_mean_val_mmm_stdev_H2)
        #ht_fn.output_txt_file_reg_mmm(out_fname_act_conc_sd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_mean_vals_act_mmm_stdev_H2,global_mean_val_act_mmm_stdev_H2)
        
        #print 'Write out to file HTAP2 multi-model change in Global Total O3 column Burden and RE'
        #header_out = ['Year','Emis_scn','Global Col Burd (Tg)','+/- S.D.','+/- Var','O3 RF W m-2','+/- S.D.','+/- Var']
        #header_out_str = ','.join(header_out)
        #out_fname_o3_burd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H2_global_ann_mean_RESPONSE_'+SPEC+'_col_burd_and_RE_for_'+EMIS_SCN+'_src_regs.txt'
        #ht_fn.output_txt_file_burd_mmm(out_fname_o3_burd_mmm,header_out_str,['2010']+YEARS_SCN,H2_SRC_REGIONS,base_burd,base_rf_glob,o3_burd_all_mod_H2_mmm,o3_burd_all_mod_H2_sd,o3_burd_all_mod_H2_var,rf_glob_all_mod_H2_mmm,rf_glob_all_mod_H2_sd,rf_glob_all_mod_H2_var)      
        
        # Regional Burden and RF changes
        #print 'Write out to file HTAP1 multi-model change in REGIONAL Total O3 column Burden and RE'
        #header_out_reg              = ['Year','Emis_scn','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
        #header_out_reg_str          = ','.join(header_out_reg)
        #out_fname_reg_o3_burd_mmm   = OUT_TXT_FILE_PATH+'Multi-model_H2_GLOBAL_and_REGIONAL_ann_mean_RESPONSE_'+SPEC+'_col_burd_and_RE_for_'+EMIS_SCN+'_src_regs.txt'
        #ht_fn.output_txt_file_reg_burd_mmm(out_fname_reg_o3_burd_mmm,header_out_reg_str,header_out[2:],['2010']+YEARS_SCN,H2_SRC_REGIONS,base_burd_htap2_regs,base_rf_htap2_regs,
        #                                   reg_o3_burd_all_mod_H2_mmm,reg_o3_burd_all_mod_H2_sd,reg_o3_burd_all_mod_H2_var,reg_o3_rf_all_mod_H2_mmm,reg_o3_rf_all_mod_H2_sd,reg_o3_rf_all_mod_H2_var,
        #                                   base_burd,base_rf_glob,o3_burd_all_mod_H2_mmm,o3_burd_all_mod_H2_sd,o3_burd_all_mod_H2_var,rf_glob_all_mod_H2_mmm,rf_glob_all_mod_H2_sd,rf_glob_all_mod_H2_var)
        
        print( 'Write out to file HTAP2 multi-model change in REGIONAL Total O3 column Burden and RE')
        
        out_fname_reg_o3_burd_mmm    = OUT_TXT_FILE_PATH+'Multi-model_H2_regional_average_RESPONSE_'+SPEC+'_col_burd_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        out_fname_reg_o3_rf_mmm      = OUT_TXT_FILE_PATH+'Multi-model_H2_regional_average_RESPONSE_'+SPEC+'_Radiative_Forcing_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        out_fname_reg_o3_burd_sd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H2_STD_DEV_in_regional_average_RESPONSE_'+SPEC+'_col_burd_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        out_fname_reg_o3_rf_sd_mmm   = OUT_TXT_FILE_PATH+'Multi-model_H2_STD_DEV_in_regional_average_RESPONSE_'+SPEC+'_Radiative_Forcing_for_'+EMIS_SCN+'_on_HTAP2_receptors.txt'
        
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_burd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_o3_burd_all_mod_H2_mmm,o3_burd_all_mod_H2_mmm)
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_rf_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_o3_rf_all_mod_H2_mmm,rf_glob_all_mod_H2_mmm)
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_burd_sd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_o3_burd_all_mod_H2_sd,o3_burd_all_mod_H2_sd)
        ht_fn.output_txt_file_reg_mmm(out_fname_reg_o3_rf_sd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_o3_rf_all_mod_H2_sd,rf_glob_all_mod_H2_sd)
        
        
        print( '#### Finished Writing Regional Response for HTAP2 Multi Model Mean Values ####')
        
    ################################
    ################################  
    
    # NOW NEED TO RECOMBINE BOTH HTAP1 AND HTAP2 MULTI-MODEL RESULTS TO GET AN OVERALL RESPONSE
    print( '###### COMBINE HTAP1 AND HTAP2 RESULTS TOGETHER #######')
    
    print( 'Combine HTAP1 and HTAP2 Methane response together into a H1 and H2 array')
    # Join together H1 and H2 methane response gridded output
    ch4_srf_all_yrs_H1_H2_comb          = np.concatenate((ch4_srf_all_yrs_H1,ch4_srf_all_yrs_H2),axis=0)
    ch4_3d_reg_all_yrs_H1_H2_comb       = np.concatenate((ch4_3d_reg_all_yrs_H1,ch4_3d_reg_all_yrs_H2),axis=0)
    # Global values
    global_mean_val_H1_H2_ch4_comb      = np.concatenate((global_mean_val_H1_ch4,global_mean_val_H2_ch4),axis=0)
    global_mean_val_act_H1_H2_ch4_comb  = np.concatenate((global_mean_val_act_H1_ch4,global_mean_val_act_H2_ch4),axis=0)
    global_burd_resp_H1_H2_ch4_comb     = np.concatenate((o3_burd_glob_H1_ch4,o3_burd_glob_H2_ch4),axis=0)
    global_rf_resp_H1_H2_ch4_comb       = np.concatenate((o3_rf_glob_H1_ch4,o3_rf_glob_H2_ch4),axis=0)
    # Regional values
    reg_mean_val_H1_H2_ch4_comb         = np.concatenate((reg_mean_val_H1_ch4,reg_mean_val_H2_ch4),axis=0)
    reg_mean_val_H1_H2_ch4_comb[reg_mean_val_H1_H2_ch4_comb == 0.0]     = np.nan # where there is no change (or no values assign as nan for meaning purposes)
    reg_mean_val_act_H1_H2_ch4_comb     = np.concatenate((reg_mean_val_act_H1_ch4,reg_mean_val_act_H2_ch4),axis=0)
    reg_mean_val_act_H1_H2_ch4_comb[reg_mean_val_H1_H2_ch4_comb == 0.0] = np.nan
    reg_burd_resp_H1_H2_ch4_comb        = np.concatenate((o3_burd_reg_H1_ch4,o3_burd_reg_H2_ch4),axis=0)
    reg_burd_resp_H1_H2_ch4_comb[reg_burd_resp_H1_H2_ch4_comb == 0.0]   = np.nan
    reg_rf_resp_H1_H2_ch4_comb          = np.concatenate((o3_rf_reg_H1_ch4,o3_rf_reg_H2_ch4),axis=0)
    reg_rf_resp_H1_H2_ch4_comb[reg_rf_resp_H1_H2_ch4_comb == 0.0]       = np.nan
    
    print( 'Calculate multi-model mean values to CH4 response for both HTAP1 and HTAP2 models')
    # Global gridded arrays
    ch4_srf_all_yrs_H1_H2_comb_mmm              = np.mean(ch4_srf_all_yrs_H1_H2_comb,axis=0)
    ch4_3d_reg_all_yrs_H1_H2_comb_mmm           = np.mean(ch4_3d_reg_all_yrs_H1_H2_comb,axis=0)
    # Global value
    global_mean_val_H1_H2_ch4_comb_mmm          = np.mean(global_mean_val_H1_H2_ch4_comb,axis=0)
    global_mean_val_act_H1_H2_ch4_comb_mmm      = np.mean(global_mean_val_act_H1_H2_ch4_comb,axis=0)
    global_mean_burd_resp_H1_H2_ch4_comb_mmm    = np.mean(global_burd_resp_H1_H2_ch4_comb,axis=0)
    global_mean_rf_resp_H1_H2_ch4_comb_mmm      = np.mean(global_rf_resp_H1_H2_ch4_comb,axis=0)
    # Regional values
    reg_mean_val_H1_H2_ch4_comb_mmm             = np.nanmean(reg_mean_val_H1_H2_ch4_comb,axis=0)
    reg_mean_val_act_H1_H2_ch4_comb_mmm         = np.nanmean(reg_mean_val_act_H1_H2_ch4_comb,axis=0)
    reg_mean_burd_resp_H1_H2_ch4_comb_mmm       = np.nanmean(reg_burd_resp_H1_H2_ch4_comb,axis=0)  
    reg_mean_rf_resp_H1_H2_ch4_comb_mmm         = np.nanmean(reg_rf_resp_H1_H2_ch4_comb,axis=0)
    
    print( 'Global CH4 MMM'      , global_mean_val_H1_H2_ch4_comb_mmm)
    print( 'Global Burd CH4 MMM' , global_mean_burd_resp_H1_H2_ch4_comb_mmm)
    print( 'Global RF CH4 MMM'   , global_mean_rf_resp_H1_H2_ch4_comb_mmm)
    
    ################################  
    
    print( 'Initialise New combined arrays')
    # Initialise new combined arrays for multi-model means
    # Global values
    global_mean_val_mmm_comb_H1_H2_ch4      = np.zeros(NYRS,dtype='f')
    global_mean_val_act_mmm_comb_H1_H2_ch4  = np.zeros(NYRS,dtype='f')
    o3_burd_all_mod_mmm_comb_H1_H2_ch4      = np.zeros(NYRS,dtype='f')
    o3_burd_diff_all_mod_mmm_comb_H1_H2_ch4 = np.zeros(NYRS,dtype='f')
    rf_glob_all_mod_mmm_comb_H1_H2_ch4      = np.zeros(NYRS,dtype='f')
    rf_glob_diff_all_mod_mmm_comb_H1_H2_ch4 = np.zeros(NYRS,dtype='f')
    # regional values
    reg_mean_vals_mmm_comb_H1_H2_ch4        = np.zeros((NYRS,NREGS_H2_OUT),dtype='f')
    reg_mean_vals_act_mmm_comb_H1_H2_ch4    = np.zeros((NYRS,NREGS_H2_OUT),dtype='f')
    o3_burd_reg_all_mod_mmm_comb_H1_H2_ch4  = np.zeros((NYRS,NREGS_H2_OUT),dtype='f')
    rf_reg_all_mod_mmm_comb_H1_H2_ch4       = np.zeros((NYRS,NREGS_H2_OUT),dtype='f')
    # Gridded arrays
    o3_srf_tot_all_yrs_MMM_comb_H1_H2_ch4   = np.zeros((NYRS,ntime,nlats,nlons),dtype='f')
    o3_3d_tot_all_yrs_MMM_comb_H1_H2_ch4    = np.zeros((NYRS,ntime,nlevs,nlats,nlons),dtype='f')
    
    # Create output with CH4 removed for both surface and 3D
    print( ' Seprate CH4 response from H1 and H2 fields')
    o3_srf_reg_all_yrs_MMM_H2_no_ch4 = o3_srf_reg_all_yrs_MMM_H2[:,:,:,:,1:] #np.zeros((NYRS,ntime,nlats,nlons),dtype='f')
    o3_3d_reg_all_yrs_MMM_H2_no_ch4  = o3_3d_reg_all_yrs_MMM_H2[:,:,:,:,:,1:] #np.zeros((NYRS,ntime,nlevs,nlats,nlons),dtype='f')
    o3_srf_reg_all_yrs_MMM_H1_no_ch4 = o3_srf_reg_all_yrs_H1_mmm[:,:,:,:,1:]  #np.zeros((NYRS,ntime,nlats,nlons),dtype='f')
    o3_3d_reg_all_yrs_MMM_H1_no_ch4  =  o3_3d_reg_all_yrs_H1_mmm[:,:,:,:,:,1:] #np.zeros((NYRS,ntime,nlevs,nlats,nlons),dtype='f')
    
    # Add combined CH4 fields back onto H1 and H2 fields with no CH4 response
    print( 'Add back together combined Mathane responses to H1 and H2 responses')
    # Sum up O3 response over all source regions to give total response here for H1 src regions + H2 src egions + Ch4 response
    o3_srf_all_yrs_MMM_comb_H1_H2_ch4 = np.sum(o3_srf_reg_all_yrs_MMM_H2_no_ch4, axis=4) + np.sum(o3_srf_reg_all_yrs_MMM_H1_no_ch4, axis=4) + ch4_srf_all_yrs_H1_H2_comb_mmm
    o3_3d_all_yrs_MMM_comb_H1_H2_ch4  = np.sum(o3_3d_reg_all_yrs_MMM_H2_no_ch4, axis=5) + np.sum(o3_3d_reg_all_yrs_MMM_H1_no_ch4, axis=5) + ch4_3d_reg_all_yrs_H1_H2_comb_mmm
    
    ################################  
    
    # for Total O3 to be used in burden and RF calculations combine back together for each year
    for (iyr,yr) in enumerate(YEARS_SCN):
        print( 'Combine HTAP1 and HTAP2 results for year {}'.format(yr))
        # Careful Check not double counting CH4 contribution
        # Total up Global response values to all source region emission perturbations (for H1 and H2) and add methane contributions back on
        global_mean_val_mmm_comb_H1_H2_ch4[iyr] = np.sum(global_mean_val_mmm_H2[iyr,1:]) + np.sum(global_mean_val_mmm_H1[iyr,1:]) + global_mean_val_H1_H2_ch4_comb_mmm[iyr]
                
        # Check if this way giving the same answer as calculating directly
        print( 'H2 ',np.sum(global_mean_val_mmm_H2[iyr,1:]))
        print( 'H1 ',np.sum(global_mean_val_mmm_H1[iyr,1:]))
        print( 'CH4 ',global_mean_val_H1_H2_ch4_comb_mmm[iyr])
        print( 'Total ',global_mean_val_mmm_comb_H1_H2_ch4[iyr])
        print( 'Global direct ',ht_fn.calc_glob_mean(o3_srf_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:],area2d_global))
        
        print( 'Add baseline back onto for total O3 response')
        o3_srf_tot_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:]  = o3_srf_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:] + srf_o3_data_SR1[:,:,:]
        o3_3d_tot_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:,:] = o3_3d_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:,:] + o3_3d_data_SR1.filled(float('nan'))
        
        # Combine the scaled response of Global burdens and RF (response not total burdens or RF) from each scenario
        o3_burd_all_mod_mmm_comb_H1_H2_ch4[iyr] = np.sum(o3_burd_all_mod_H2_mmm[iyr,1:]) + np.sum(o3_burd_all_mod_H1_mmm[iyr,1:]) + global_mean_burd_resp_H1_H2_ch4_comb_mmm[iyr]
        rf_glob_all_mod_mmm_comb_H1_H2_ch4[iyr] = np.sum(rf_glob_all_mod_H2_mmm[iyr,1:]) + np.sum(rf_glob_all_mod_H1_mmm[iyr,1:]) + global_mean_rf_resp_H1_H2_ch4_comb_mmm[iyr]
        
        print( 'GLOB O3 burd RESP'   , o3_burd_all_mod_mmm_comb_H1_H2_ch4[iyr])
        print( 'GLOB O3 RF RESP'     , rf_glob_all_mod_mmm_comb_H1_H2_ch4[iyr])
        
        # For each year of scenario now combine results together for each HTAP2 receptor region
        for (ireg_h2,reg_h2) in enumerate(sorted(HTAP2_RECP_REGS)):
            print( 'Combine HTAP1 and HTAP2 results for HTAP2 Receptor Region {}'.format(reg_h2))
            # can only combine O3 responses like this
            # Total up regional response values to all source region emission perturbations (for H1 and H2) and add methane contributions back on
            
            # statement to account for zero change in CH4 in scenario
            if np.isnan(reg_mean_val_H1_H2_ch4_comb_mmm[iyr,ireg_h2]): # if no change in CH4 then do no need to include in totals as is NaN values
                print('No CH4 change so do not add surf change from CH4 response to combined response')
                reg_mean_vals_mmm_comb_H1_H2_ch4[iyr,ireg_h2] = np.sum(reg_mean_vals_mmm_H2[iyr,1:,ireg_h2]) + np.sum(reg_mean_vals_mmm_H1[iyr,1:,ireg_h2])
            else:
                reg_mean_vals_mmm_comb_H1_H2_ch4[iyr,ireg_h2] = np.sum(reg_mean_vals_mmm_H2[iyr,1:,ireg_h2]) + np.sum(reg_mean_vals_mmm_H1[iyr,1:,ireg_h2]) + reg_mean_val_H1_H2_ch4_comb_mmm[iyr,ireg_h2]
            
            # can't combine total O3 response otherwise end up with 3x as much ppbv of O3
                        
            # Combine the scaled response of Regional burdens and RF from each scenario
            # statement to account for zero change in CH4 in scenario
            if np.isnan(reg_mean_burd_resp_H1_H2_ch4_comb_mmm[iyr,ireg_h2]):
                print('No CH4 change so do not add burden change from CH4 response to combined response')
                o3_burd_reg_all_mod_mmm_comb_H1_H2_ch4[iyr,ireg_h2] = np.sum(reg_o3_burd_all_mod_H2_mmm[iyr,1:,ireg_h2]) + np.sum(reg_o3_burd_all_mod_H1_mmm[iyr,1:,ireg_h2])
            else: 
                o3_burd_reg_all_mod_mmm_comb_H1_H2_ch4[iyr,ireg_h2] = np.sum(reg_o3_burd_all_mod_H2_mmm[iyr,1:,ireg_h2]) + np.sum(reg_o3_burd_all_mod_H1_mmm[iyr,1:,ireg_h2]) + reg_mean_burd_resp_H1_H2_ch4_comb_mmm[iyr,ireg_h2]
            
            # statement to account for zero change in CH4 in scenario
            if np.isnan(reg_mean_rf_resp_H1_H2_ch4_comb_mmm[iyr,ireg_h2]):
                print('No CH4 change so do not add RF change from Ch4 response to combined response')
                rf_reg_all_mod_mmm_comb_H1_H2_ch4[iyr,ireg_h2]      = np.sum(reg_o3_rf_all_mod_H2_mmm[iyr,1:,ireg_h2]) + np.sum(reg_o3_rf_all_mod_H1_mmm[iyr,1:,ireg_h2])
            else:
                rf_reg_all_mod_mmm_comb_H1_H2_ch4[iyr,ireg_h2]      = np.sum(reg_o3_rf_all_mod_H2_mmm[iyr,1:,ireg_h2]) + np.sum(reg_o3_rf_all_mod_H1_mmm[iyr,1:,ireg_h2]) + reg_mean_rf_resp_H1_H2_ch4_comb_mmm[iyr,ireg_h2]
            
        # Recalculate total O3 response based on combined mmm surface fields
        reg_mean_vals_act_mmm_comb_H1_H2_ch4[iyr,:] = ht_fn.calc_H2_reg_response(ntime,o3_srf_tot_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:],area2d_global,NREGS_H2_OUT,HTAP2_RECP_REGS)
        global_mean_val_act_mmm_comb_H1_H2_ch4[iyr] = ht_fn.calc_glob_mean(o3_srf_tot_all_yrs_MMM_comb_H1_H2_ch4[iyr,:,:,:],area2d_global)
    
    print( 'Calculate Total Burdens and RF by adding baseline values back on')
    # Calculate total burden now (response + base)
    o3_burd_tot_all_mod_mmm_comb_H1_H2_ch4      = o3_burd_all_mod_mmm_comb_H1_H2_ch4 + base_burd
    rf_glob_tot_all_mod_mmm_comb_H1_H2_ch4      = rf_glob_all_mod_mmm_comb_H1_H2_ch4 + base_rf_glob
    o3_burd_tot_reg_all_mod_mmm_comb_H1_H2_ch4  = o3_burd_reg_all_mod_mmm_comb_H1_H2_ch4 + base_burd_htap2_regs
    rf_reg_tot_all_mod_mmm_comb_H1_H2_ch4       = rf_reg_all_mod_mmm_comb_H1_H2_ch4 + base_rf_htap2_regs
    
    ################################  
    
    ##### OUTPUT SECTION (BOTH TEXT FILES AND NETCDF) #####
    
    #print 'Output Combined total multi-model mean O3 response Fields to a netcdf file'
    #out_fname_mmm_comb_H1_ch4 = 'Multi_model_mean_{}_response_from_H1_H2_models_to_{}_change_H1_CH4_used.nc'.format(SPEC,EMIS_SCN)
    if IOUT_NCF_MMM_COMB == 1:
        print( 'Output HTAP1 and HTAP2 combined multi-model scaled Ozone fields to Netcdf files')
        out_fname_mmm_comb_H1_H2_ch4 = 'Multi_model_mean_{}_response_from_H1_H2_models_to_{}_change_H1_H2_comb_CH4_used.nc'.format(SPEC,EMIS_SCN)
        out_fname_path_h2_mmm        = OUT_FILE_PATH+out_fname_mmm_comb_H1_H2_ch4
        ht_fn.output_file_mod_H1_H2(out_fname_path_h2_mmm,'Combined_H1_H2_CH4',EMIS_SCN,nlevs,nlons,nlats,ntime,NYRS,levs,lons,lats,time,YEARS_SCN,
                                    o3_3d_all_yrs_MMM_comb_H1_H2_ch4,o3_srf_all_yrs_MMM_comb_H1_H2_ch4,o3_3d_tot_all_yrs_MMM_comb_H1_H2_ch4,o3_srf_tot_all_yrs_MMM_comb_H1_H2_ch4)
    
    # Output multi-model regional concentration responses at each HTAP2 receptor to each HTAP2 regional emission perturbation scenario            
    print( 'Output regional multi-model mean values from combined HTAP1 and HTAP2 multi-model scaled Ozone fields')
    header_out                                  = ['Year','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
    header_out_str                              = ','.join(header_out)
    out_fname_resp_mmm_comb_H1_H2_ch4_comb      = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_regional_average_RESPONSE_in_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors_H1_H2_CH4_comb.txt'
    out_fname_act_conc_mmm_comb_H1_H2_ch4_comb  = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_regional_average_TOTAL_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors_H1_H2_CH4_comb.txt'
    #out_fname_act_conc_sd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H2_STD_DEV_in_regional_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
    #out_fname_resp_sd_mmm = OUT_TXT_FILE_PATH+'Multi-model_H2_STD_DEV_in_regional_average_RESPONSE_in_'+SPEC+'_concentrations_for_'+EMIS_SCN+'_on_HTAP_2_receptors.txt'
    # Output text file of regional means
    #ht_fn.output_txt_file_reg_mmm(out_fname_resp_sd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_mean_vals_mmm_stdev_H2,global_mean_val_mmm_stdev_H2)
    #ht_fn.output_txt_file_reg_mmm(out_fname_act_conc_sd_mmm,header_out_str,YEARS_SCN,H2_SRC_REGIONS,reg_mean_vals_act_mmm_stdev_H2,global_mean_val_act_mmm_stdev_H2)
    ht_fn.output_txt_file_reg_mmm_comb(out_fname_resp_mmm_comb_H1_H2_ch4_comb,header_out_str,YEARS_SCN,reg_mean_vals_mmm_comb_H1_H2_ch4,global_mean_val_mmm_comb_H1_H2_ch4)
    ht_fn.output_txt_file_reg_mmm_comb(out_fname_act_conc_mmm_comb_H1_H2_ch4_comb,header_out_str,YEARS_SCN,reg_mean_vals_act_mmm_comb_H1_H2_ch4,global_mean_val_act_mmm_comb_H1_H2_ch4)
    
    #print 'Output Global Total Column Burden and Radiative Forcing values from combined HTAP1 and HTAP2 multi-model scaled Ozone fields'
    #header_out = ['Year','Total Column Burden (Tg)','Diff to Base','Radiative Forcing (W m-2)','Diff to Base']
    #header_out_str = ','.join(header_out)
    #out_fname_burd_mmm_comb_H1_H2_ch4_comb = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_Global_Total_column_'+SPEC+'_burden_and_Radiative_forc_for_'+EMIS_SCN+'_H1_H2_CH4_comb.txt'
    #ht_fn.output_txt_file_burd_mmm_comb(out_fname_burd_mmm_comb_H1_H2_ch4_comb,header_out_str,['2010']+YEARS_SCN,base_burd,base_rf_glob,o3_burd_all_mod_mmm_comb_H1_H2_ch4,rf_glob_all_mod_mmm_comb_H1_H2_ch4,o3_burd_diff_all_mod_mmm_comb_H1_H2_ch4,rf_glob_diff_all_mod_mmm_comb_H1_H2_ch4)
    
    print( 'Output Global and HTAP2 Regional Total Column Burden and Radiative Forcing RESPONSE values from combined HTAP1 and HTAP2 multi-model scaled Ozone fields')
    header_reg_out                          = ['Year','MCA','CAS','EAS','EUR','MDE','NAF','NAM','NOP','OCN','PAN','RBU','SAF','SAM','SAS','SEA','SOP','GLO']
    header_reg_out_str                      = ','.join(header_reg_out)
    # output burden and RF RESPONSE
    out_fname_burd_mmm_comb_H1_H2_ch4_comb  = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_reg_mean_RESPONSE_'+SPEC+'_burden_for_'+EMIS_SCN+'_on_H2_receptors_H1_H2_CH4_comb.txt'
    ht_fn.output_txt_file_reg_mmm_comb(out_fname_burd_mmm_comb_H1_H2_ch4_comb,header_reg_out_str,YEARS_SCN,o3_burd_reg_all_mod_mmm_comb_H1_H2_ch4,o3_burd_all_mod_mmm_comb_H1_H2_ch4)
    out_fname_rf_mmm_comb_H1_H2_ch4_comb    = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_reg_mean_RESPONSE_'+SPEC+'_Radiative_Forcing_for_'+EMIS_SCN+'_on_H2_receptors_H1_H2_CH4_comb.txt'
    ht_fn.output_txt_file_reg_mmm_comb(out_fname_rf_mmm_comb_H1_H2_ch4_comb,header_reg_out_str,YEARS_SCN,rf_reg_all_mod_mmm_comb_H1_H2_ch4,rf_glob_all_mod_mmm_comb_H1_H2_ch4)
    
    # Output TOTAL Burden and TOTAL RF response
    print( 'Output Global and HTAP2 Regional Total Column Burden and Radiative Forcing TOTAL values from combined HTAP1 and HTAP2 multi-model scaled Ozone fields')
    out_fname_burd_mmm_comb_H1_H2_ch4_comb  = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_reg_mean_TOTAL_'+SPEC+'_burden_for_'+EMIS_SCN+'_on_H2_receptors_H1_H2_CH4_comb.txt'
    ht_fn.output_txt_file_reg_mmm_comb(out_fname_burd_mmm_comb_H1_H2_ch4_comb,header_reg_out_str,YEARS_SCN,o3_burd_tot_reg_all_mod_mmm_comb_H1_H2_ch4,o3_burd_tot_all_mod_mmm_comb_H1_H2_ch4)
    out_fname_rf_mmm_comb_H1_H2_ch4_comb    = OUT_TXT_FILE_PATH+'Multi-model_H1_H2_comb_reg_mean_TOTAL_'+SPEC+'_Radiative_Forcing_for_'+EMIS_SCN+'_on_H2_receptors_H1_H2_CH4_comb.txt'
    ht_fn.output_txt_file_reg_mmm_comb(out_fname_rf_mmm_comb_H1_H2_ch4_comb,header_reg_out_str,YEARS_SCN,rf_reg_tot_all_mod_mmm_comb_H1_H2_ch4,rf_glob_tot_all_mod_mmm_comb_H1_H2_ch4)
                                       
    print( '#### Finished Writing Regional Response for HTAP2 Multi Model Mean Values ####')
    
    ################################ 
    
    ## PLOT UP regional response values
    # Plot out Multi-model mean O3 responses over different years for EUR, NAM, SAS, EAS, GLOBAL
    if IPLOT_REG_MMM_COMB == 1:
        print( 'Plot up Regional HTAP1 Multi-model mean responses to Emission change across years')
        out_plot_fname = PLOT_DIR+'Regional_Multi-model_mean_H2_{}_response_between_{}_to_{}_for_{}_emis_change.png'.format(SPEC,YEARS_SCN[0],YEARS_SCN[-1],EMIS_SCN)
        #if CALC_REG_HTAP1 == 1: plot_reg_changes_mods(pert_wght_1,imod,cur_mod,iyr,year,out_plot_fname)
                    
        year_2010 = np.zeros((1,NREGS_H2_OUT), dtype='f')
        reg_mean_vals_mmm_2_plot = np.vstack((year_2010,reg_mean_vals_mmm_comb_H1_H2_ch4)) # Put zeros at start for plotting
        years_plot = ['2010'] + YEARS_SCN
        ht_fn.plot_reg_changes_all_years_comb(out_plot_fname,years_plot,reg_mean_vals_mmm_2_plot,HTAP2_RECP_REGS) # (NYRS,NREGS)
    
    print( '###### !!! FIN !!! #######')
        
    ################################ 
