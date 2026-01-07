#!/usr/bin/env python3.8
# Class for using ppe design 
# R. Sansom 8/8/22

import numpy as np
import xarray as xr
import cloud_lib as cl
import pandas as pd

def get_sst():
    SST_dict = {'surface_boundary_input_times' : np.asarray([0.0, 28800.0, 50400.0, 72000.0, 93600.0,117000.0,136800.0,158400.0,180000.0,201600.0,223200.0,243800.0,266400.0]),
                'surface_temperatures' : np.asarray([293.75, 294.16, 294.55, 295.08, 295.57, 296.1, 296.55, 297.02, 297.54, 298.06, 298.44, 298.8, 299.17])}
    return SST_dict

def pre_sorter(member, sc_thresh, cu_thresh, include_spinup=False):
    if include_spinup:
        member.cloud_fraction = member.ds.cloud_frac
        member.rwp = member.ds.rwp
    else:
        member.cloud_fraction = member.ds.cloud_frac[1:]
        member.rwp = member.ds.rwp[5:]

    sorter(member, sc_thresh, cu_thresh)

def sorter(member, sc_thresh, cu_thresh):

    # Does the simulation form Sc?
    if any(member.cloud_fraction > sc_thresh):
        member.sc = True
        # Which indices have Sc?
        sc, = np.where(member.cloud_fraction > sc_thresh)
        member.sc_ind = sc[0]
        member.sc_time = member.cloud_fraction.time_coarse[member.sc_ind].data

        # Fine resolution time index needed for RWP
        member.sc_fine_index = abs(member.rwp.time_fine - member.sc_time).argmin().data

        # transition beginning
        if sc[-1] < len(member.cloud_fraction)-1:
            sc_diff = sc[1:] - sc[:-1]
            j, = np.where(sc_diff > 1)
            if len(j) > 0:
                t_begin_ind = sc[j[0]]
            else:
                t_begin_ind = sc[-1]
    
            x = [member.cloud_fraction[t_begin_ind+1].data, member.cloud_fraction[t_begin_ind].data]
            y = [member.cloud_fraction.time_coarse[t_begin_ind+1].data, member.cloud_fraction.time_coarse[t_begin_ind].data]
    
            member.t_begin_ind = t_begin_ind
            member.t_begin_time = np.interp(sc_thresh, x, y)
            member.t_begin_time_sc = member.t_begin_time - member.sc_time
            member.t_begin_ind_fine = abs(member.rwp.time_fine - member.t_begin_time).argmin().data
        else:
            member.t_begin_ind = None
            member.t_begin_time = None
            member.t_begin_time_sc = None
            member.t_begin_ind_fine = None

        # Out of those indices, which following ones have Cu?
        cu, = np.where(member.cloud_fraction[member.sc_ind:] < cu_thresh)
        # cu_ind refers to the original simulation indices that have Cu
        cu_inds_after_sc = cu + member.sc_ind
        
        # Does Cu form and the simulation finishes in a Cu state?
        if len(cu)!=0 and member.cloud_fraction[-1] < cu_thresh:
            member.cu = True
            # Does it stay in Cu from the initial Cu formation?
            if all(member.cloud_fraction[cu_inds_after_sc[0]:] < cu_thresh):
                member.cu_ind = cu_inds_after_sc[0]
            # For the case where it recovers from Cu but will eventually return to Cu
            else:
                # Find the difference between the indices that have Cu, take the index which is the last one with a difference of more than one timestep. 
                # From this point, the differences are only one timestep so the cloud does not recover again.
                diff = cu[1:] - cu[:-1]
                i, = np.where(diff > 1)
                member.cu_ind = cu_inds_after_sc[i[-1]+1]

            member.cu_time = member.cloud_fraction.time_coarse[member.cu_ind].data
            member.transition_time = member.cu_time - member.sc_time
            member.transition_time2 = member.cu_time - member.t_begin_time

            # Fine resolution time index needed for RWP
            member.cu_fine_index = abs(member.rwp.time_fine - member.cu_time).argmin().data
            member.rwp_mean = member.rwp[member.sc_fine_index:member.cu_fine_index+1].mean()*1e3
            member.rwp_mean2 = member.rwp[member.t_begin_ind_fine:member.cu_fine_index+1].mean()*1e3

        # Where no Cu is formed or it forms but then recovers - could split this for ones that form Cu 
        else:
            member.cu = False
            member.cu_time = None
            member.cu_fine_index = None
            member.transition_time = None
            member.rwp_mean = None
            member.rwp_mean2 = None

    # Where no Sc is formed
    else:
        member.sc = False
        member.cu = False
        member.sc_ind = None
        member.sc_time = None
        member.sc_fine_index = None
        member.t_begin_ind = None
        member.t_begin_time = None
        member.t_begin_time_sc = None
        member.cu_ind = None
        member.cu_time = None
        member.cu_fine_index = None
        member.transition_time = None
        member.rwp_mean = None
        member.rwp_mean2 = None


class Member:
    def __init__(self, key, index, inputs):
        self.key = key
        self.index = index
        self.id = f"{key}{index}"
        self.inputs = inputs
        self.ds = xr.open_dataset(f"/gws/nopw/j04/carisma/eers/sct/combined_processed/sct_{key}{index}_pp.nc")


class ICE_Member:
    def __init__(self, root_key, var_num, inputs):
        self.root_key = root_key
        self.var_num = var_num
        self.id = f"{root_key}_{var_num}"
        self.inputs = inputs
        self.ds = xr.open_dataset(f"/gws/nopw/j04/carisma/eers/sct/processed/initial_condition_ensembles/{root_key}/sct_{root_key}_{var_num}_pp.nc")


class Ensemble:
    def __init__(self, sc_thresh, cu_thresh, 
                 include_spinup=False, 
                 delete_bad=True, 
                 use_sc_beginning_vals=None, 
                 find_ssts=None, 
                 use_ic_ensembles=None):

        print("Initiating ensemble...")
        self.sc_thresh = sc_thresh
        self.cu_thresh = cu_thresh
        # Assign names and labels
        self.parameter_names = ['qv_bl','inv','delt','delq','na','baut']
        self.parameter_labels = [r'$BL~q_{v}$', r'$BL~z$', r'$\Delta~\theta$', r'$\Delta~q_{v}$', r'$BL~N_{a}$', r'10^{$b_{aut}$}']
        self.axes_labels = [(7, 11), (500, 1300), (2, 21), (-7, -1), (10, 500), (10**(-2.3), 10**(-1.3))]

        # Load Latin hypercube design (including spinup or not)
        # Assign parameter ranges
        self.include_spinup = include_spinup
        if include_spinup:
            self.design = np.loadtxt("lh_design/original/SCT_full_inputs_design.csv",delimiter=',',skiprows=1)
            self.parameter_ranges = [(7,11), (500,1300), (2,21), (-7,-1), (10,500), (-2.3,-1.3)]
        else:
            self.design = np.loadtxt("lh_design/post_spinupvalues/all_post_spinup.csv",delimiter=',',skiprows=1)
            minimums = [min(column) for column in (self.design.T)]
            maximums = [max(column) for column in (self.design.T)]
            self.parameter_ranges = [(pmin,pmax) for pmin,pmax in zip(minimums, maximums)]

        if use_sc_beginning_vals == True:
            self.design = np.loadtxt("lh_design/post_spinupvalues/all_sc_beginning.csv",delimiter=',',skiprows=1)
            minimums = [min(column) for column in (self.design.T)]
            maximums = [max(column) for column in (self.design.T)]
            self.parameter_ranges = [(pmin,pmax) for pmin,pmax in zip(minimums, maximums)]

        # Create members as instances of Member
        self.member_keys = []
        self.member_sct_keys = []
        self.member_sc_keys = []
        self.member_sc_no_cu_keys = []
        self.member_bad_keys = ["em6", "em86", "em93"]
        key = "em"
        print("Loading members")
        for i, member_inputs in enumerate(self.design):
            setattr(self, f"{key}{i}", Member(key,i,member_inputs))
            member = vars(self)[f"{key}{i}"]

            if f"{key}{i}" in self.member_bad_keys:
                member.sc = False
                member.cu = False
                member.sc_ind = None
                member.sc_time = None
                member.sc_fine_index = None
                member.cu_ind = None
                member.cu_time = None
                member.cu_fine_index = None
                member.transition_time = None
                member.rwp_mean = None
            else:
                pre_sorter(member, self.sc_thresh, self.cu_thresh, include_spinup)

            self.member_keys.append(f"{key}{i}")
            if member.cu:
                self.member_sct_keys.append(f"{key}{i}")
                self.member_sc_keys.append(f"{key}{i}")
            elif member.sc:
                self.member_sc_no_cu_keys.append(f"{key}{i}")
                self.member_sc_keys.append(f"{key}{i}")

        if delete_bad:
            self.design = np.delete(self.design, (6,86,93), axis=0)
            self.member_keys = set(self.member_keys) ^ set(self.member_bad_keys)

        if find_ssts == True:
            self.find_initial_ssts()

        if use_ic_ensembles == True:
            self.load_ice_members()
            self.replace_members_with_ice_means()

        print("Ensemble initialised.")

    # def load_variable_from_merged_nc(self, member, variable_strings):
    #     if member.index < 61:
    #         key = "em"
    #         index = member.index
    #     elif member.index > 60 and member.index < 85:
    #         key = "val"
    #         index = member.index - 61
    #     elif member.index > 84:
    #         key = "xtra"
    #         index = member.index - 85

    #     ds = xr.open_dataset(f"/gws/nopw/j04/carisma/eers/sct/{key}/{key}{index}/sct_{key}{index}_merged.nc")
    #     ds = cl.ds_fix_dims(ds)
    #     list_of_variables = [ds[variable] for variable in variable_strings]
    #     list_of_variables.insert(0, vars(self)[member.id].ds)
    #     vars(self)[member.id].ds = xr.merge(list_of_variables)

    def create_training_set(self, output_string):
        column_names = self.parameter_names + [output_string]
        training_set = pd.DataFrame(index=range(40), columns=column_names)
        i=0
        for key in self.member_keys:
            member = vars(self)[key]
            if member.cu:
                row = np.append(member.inputs, vars(member)[output_string])
                training_set.loc[i] = row
                i+=1
        training_set = training_set.dropna()
        return training_set

    def find_initial_ssts(self):
        SST = get_sst()
        for key in self.member_keys:
            member = vars(self)[key]
            if member.sc:
                member.initial_sst = SST['surface_temperatures'][np.argmin(abs(SST['surface_boundary_input_times']/3600 - member.sc_time))]
            else:
                member.initial_sst = None

    def load_ice_members(self):
        self.ice_dict_keys = {"em0": [], "em34": [], "em80": [], "em85": []}

        for dict_key in self.ice_dict_keys.keys():
            var_length = 4 if dict_key=="em0" else 5
            for var_num in range(var_length):
                self.ice_dict_keys[dict_key].append(f"{dict_key}_{var_num}")
                setattr(self, f"{dict_key}_{var_num}", ICE_Member(dict_key, var_num, vars(self)[f"{dict_key}"].inputs))
                pre_sorter(vars(self)[f"{dict_key}_{var_num}"], self.sc_thresh, self.cu_thresh, self.include_spinup)

    def replace_members_with_ice_means(self):
        for member_key, ice_key_list in self.ice_dict_keys.items():
            member = vars(self)[member_key]
            cf_times = np.ones((len(ice_key_list), 65))*(-1)
            cf_timeseries = np.ones((len(ice_key_list), 65))*(-1)
            rwp_times = np.ones((len(ice_key_list), 180))*(-1)
            rwp_timeseries = np.ones((len(ice_key_list), 180))*(-1)
        
            for i, ice_key in enumerate(ice_key_list):
                ice_member = vars(self)[ice_key]
                cf_timeseries[i,:len(ice_member.cloud_fraction)] = ice_member.cloud_fraction
                cf_times[i,:len(ice_member.cloud_fraction)] = ice_member.cloud_fraction.time_coarse
                rwp_timeseries[i,:len(ice_member.rwp)] = ice_member.rwp
                rwp_times[i,:len(ice_member.rwp)] = ice_member.rwp.time_fine
        
            cf_timeseries_nan_mask = np.where(cf_timeseries>0, cf_timeseries, np.nan)
            cf_times_nan_mask = np.where(cf_times>0, cf_times, np.nan)
            cf_mean = np.nanmean(cf_timeseries_nan_mask, axis=0)
            cf_times_mean = np.nanmean(cf_times_nan_mask, axis=0)
            cf_mean_isnan, = np.where(np.isnan(cf_mean) == False)
        
            member.cloud_fraction = xr.DataArray(data=cf_mean[:cf_mean_isnan[-1]+1],
                                                 dims=["time_coarse"],
                                                 coords=dict(time_coarse=cf_times_mean[:cf_mean_isnan[-1]+1]),
                                                 attrs=dict(description="mean cloud_frac of ice"))

            rwp_timeseries_nan_mask = np.where(rwp_timeseries>0, rwp_timeseries, np.nan)
            rwp_times_nan_mask = np.where(rwp_times>0, rwp_times, np.nan)
            rwp_mean = np.nanmean(rwp_timeseries_nan_mask, axis=0)
            rwp_times_mean = np.nanmean(rwp_times_nan_mask, axis=0)
            rwp_mean_isnan, = np.where(np.isnan(rwp_mean) == False)
        
            member.rwp = xr.DataArray(data=rwp_mean[:rwp_mean_isnan[-1]+1],
                                                 dims=["time_fine"],
                                                 coords=dict(time_fine=rwp_times_mean[:rwp_mean_isnan[-1]+1]),
                                                 attrs=dict(description="mean rwp of ice"))
            sorter(member, self.sc_thresh, self.cu_thresh)

            if member.id not in self.member_sct_keys and member.transition_time is not None:
                self.member_sct_keys.insert(0, member.id)
