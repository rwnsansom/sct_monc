import xarray as xr
import sys
sys.path.append("./py_scripts")
import cloud_lib as cl
import re
import metpy.calc
from metpy.units import units

def create_dataset(member_key, ice=False, count=None):
    sct_dir = "/gws/nopw/j04/carisma/eers/sct"

    if ice:
        splits = member_key.split("_")

        print(f"Opening dataset {member_key}")
        ds = xr.open_dataset(f"{sct_dir}/ice/{splits[0]}/{member_key}/sct_{member_key}_merged.nc")
    else:
        r_key = re.compile(r'(\D*)(\d*)')
        a = r_key.search(member_key)
        key = a.group(1)
    
        print(f"Opening dataset {member_key}")
        ds = xr.open_dataset(f"{sct_dir}/{key}/{member_key}/sct_{member_key}_merged.nc")
    ds = cl.ds_fix_dims(ds)
    
    # print('Calculating LWP mask')
    # total_lwp, cloudy_lwp, cloud_frac, times, lwp_mask_2d = cl.lwp_cloud(ds)
    
    print('Calculating delta theta threshold')
    height_timeseries = []
    for tstep in range(len(ds.theta_mean)):
        subcloud_top_ind, inv_ind, diff = cl.get_deltheta_thresh(ds, tstep)
        height_timeseries.append(ds.zn[inv_ind])
    
    deltheta_thresh_heights = height_timeseries

    cloud_mass_lim = 1e-5
    # rain_mass_lim = 1e-6
    
    # print("Calculating in-cloud mean masses")
    # cloud_droplet_mass = ds.q_cloud_liquid_mass.where(ds.q_cloud_liquid_mass>cloud_mass_lim).mean(axis=(1,2,3))
    # rain_droplet_mass = ds.q_rain_mass.where(ds.q_rain_mass>rain_mass_lim).mean(axis=(1,2,3))
    # activated_aerosol_mass = ds.q_active_sol_liquid.where(ds.q_cloud_liquid_mass>cloud_mass_lim).mean(axis=(1,2,3))

    # print("Calculating in-cloud mean numbers")
    # cloud_droplet_number = ds.q_cloud_liquid_number.where(ds.q_cloud_liquid_mass>cloud_mass_lim).mean(axis=(1,2,3))
    # rain_droplet_number = ds.q_rain_number.where(ds.q_cloud_liquid_mass>cloud_mass_lim).mean(axis=(1,2,3))

    print("Calculating out-of-cloud mean masses")
    aitken_mass_bl = ds.q_aitken_sol_mass[:,:,:,:inv_ind].where(ds.q_cloud_liquid_mass[:,:,:,:inv_ind]<cloud_mass_lim).mean(axis=(1,2,3))
    accumulation_mass_bl = ds.q_accum_sol_mass[:,:,:,:inv_ind].where(ds.q_cloud_liquid_mass[:,:,:,:inv_ind]<cloud_mass_lim).mean(axis=(1,2,3))
    coarse_mass_bl = ds.q_coarse_sol_mass[:,:,:,:inv_ind].where(ds.q_cloud_liquid_mass[:,:,:,:inv_ind]<cloud_mass_lim).mean(axis=(1,2,3))
    
    aitken_mass_ft = ds.q_aitken_sol_mass[:,:,:,inv_ind:].mean(axis=(1,2,3))
    accumulation_mass_ft = ds.q_accum_sol_mass[:,:,:,inv_ind:].mean(axis=(1,2,3))
    coarse_mass_ft = ds.q_coarse_sol_mass[:,:,:,inv_ind:].mean(axis=(1,2,3))

    print("Calculating out-of-cloud mean numbers")
    aitken_number_bl = ds.q_aitken_sol_number[:,:,:,:inv_ind].where(ds.q_cloud_liquid_mass[:,:,:,:inv_ind]<cloud_mass_lim).mean(axis=(1,2,3))
    accumulation_number_bl = ds.q_accum_sol_number[:,:,:,:inv_ind].where(ds.q_cloud_liquid_mass[:,:,:,:inv_ind]<cloud_mass_lim).mean(axis=(1,2,3))
    coarse_number_bl = ds.q_coarse_sol_number[:,:,:,:inv_ind].where(ds.q_cloud_liquid_mass[:,:,:,:inv_ind]<cloud_mass_lim).mean(axis=(1,2,3))

    aitken_number_ft = ds.q_aitken_sol_number[:,:,:,inv_ind:].mean(axis=(1,2,3))
    accumulation_number_ft = ds.q_accum_sol_number[:,:,:,inv_ind:].mean(axis=(1,2,3))
    coarse_number_ft = ds.q_coarse_sol_number[:,:,:,inv_ind:].mean(axis=(1,2,3))

    # print("Calculating parameter timeseries")
    # q_accum_2d_time_z = ds.q_accum_sol_number.mean(axis=(1,2))
    # q_cloud_liquid_mass = ds.q_cloud_liquid_mass[:,:,128,:].where(ds.q_cloud_liquid_mass[:,:,128,:]>1e-5)

    print("Calculating lcl")
    temp = metpy.calc.temperature_from_potential_temperature(ds.prefn[:,1].values*units.pascal, ds.theta_mean[:,1]*units.kelvin)
    dewpoint = metpy.calc.dewpoint_from_specific_humidity(ds.prefn[:,1].values*units.pascal, ds.vapour_mmr_mean[:,1].values*units('kg/kg'))
    dewpoint_k = dewpoint*units.kelvin
    zlcl = 0.125*1000*(temp.values - dewpoint_k.magnitude)

    print("Calculating decoupling")
    samples = [abs(ds.time_mid.data - time).argmin() for time in ds.time_coarse.data]
    decoupling = (ds.clbas.where(ds.clbas>0).mean(axis=(1,2)) - zlcl[samples])/zlcl[samples]

    # rwp = ds.RWP_mean
    surface_precip = ds.surface_precip
    flux_up_SW_mean = ds.flux_up_SW_mean
    flux_down_SW_mean = ds.flux_down_SW_mean
    flux_up_LW_mean = ds.flux_up_LW_mean
    flux_down_LW_mean = ds.flux_down_LW_mean
    SW_heating_rate_mean = ds.SW_heating_rate_mean
    LW_heating_rate_mean = ds.LW_heating_rate_mean
    total_radiative_heating_rate_mean = ds.total_radiative_heating_rate_mean
    toa_up_LW_mean = ds.toa_up_LW_mean
    surface_down_LW_mean = ds.surface_down_LW_mean
    surface_up_LW_mean = ds.surface_up_LW_mean
    toa_down_SW_mean = ds.toa_down_SW_mean
    toa_up_SW_mean = ds.toa_up_SW_mean
    surface_down_SW_mean = ds.surface_down_SW_mean
    surface_up_SW_mean = ds.surface_up_SW_mean
    theta_mean = ds.theta_mean
    vapour_mmr_mean = ds.vapour_mmr_mean
    time_fine = ds.time_fine
    time_mid = ds.time_mid
    time_coarse = ds.time_coarse
    x = ds.x
    y = ds.y
    z = ds.z
    zn = ds.zn
    ds.close()

    print("Creating new xarray dataset")
    post_processed_ds = xr.Dataset({
        # "total_lwp":(["time_coarse"], total_lwp),
        # "cloudy_lwp":(["time_coarse"], cloudy_lwp),
        # "cloud_frac":(["time_coarse"], cloud_frac),
        # "lwp_mask_2d":(["time_coarse","x","y"], lwp_mask_2d),
        "inversion_height":(["time_mid"], deltheta_thresh_heights),
        "zlcl":(["time_mid"], zlcl),
        "decoupling":(["time_coarse"], decoupling.data),
        # "q_cloud_liquid_mass_3d_masked":(["time_coarse","x","z"], q_cloud_liquid_mass.data),
        # "q_accum_sol_number_2d_time_z":(["time_coarse","z"], q_accum_2d_time_z.data),
        # "rwp":(["time_fine"], rwp.data),
        "surface_precip":(["time_coarse","x","y"], surface_precip.data),
        "flux_up_SW_mean":(["time_coarse"], flux_up_SW_mean.data),
        "flux_down_SW_mean":(["time_coarse"], flux_down_SW_mean.data),
        "flux_up_LW_mean":(["time_coarse"], flux_up_LW_mean.data),
        "flux_down_LW_mean":(["time_coarse"], flux_down_LW_mean.data),
        "SW_heating_rate_mean":(["time_coarse"], SW_heating_rate_mean.data),
        "LW_heating_rate_mean":(["time_coarse"], LW_heating_rate_mean.data),
        "total_radiative_heating_rate_mean":(["time_coarse"], total_radiative_heating_rate_mean.data),
        "toa_up_LW_mean":(["time_coarse"], toa_up_LW_mean.data),
        "surface_down_LW_mean":(["time_coarse"], surface_down_LW_mean.data),
        "surface_up_LW_mean":(["time_coarse"], surface_up_LW_mean.data),
        "toa_down_SW_mean":(["time_coarse"], toa_down_SW_mean.data),
        "toa_up_SW_mean":(["time_coarse"], toa_up_SW_mean.data),
        "surface_down_SW_mean":(["time_coarse"], surface_down_SW_mean.data),
        "surface_up_SW_mean":(["time_coarse"], surface_up_SW_mean.data),
        "theta_mean":(["time_mid","zn"], theta_mean.data),
        "vapour_mmr_mean":(["time_mid","zn"], vapour_mmr_mean.data),
        # "cloud_droplet_mass":(["time_coarse"], cloud_droplet_mass.data),
        # "rain_droplet_mass":(["time_coarse"], rain_droplet_mass.data),
        # "activated_aerosol_mass":(["time_coarse"], activated_aerosol_mass.data),
        "aitken_mass_bl":(["time_coarse"], aitken_mass_bl.data),
        "accumulation_mass_bl":(["time_coarse"], accumulation_mass_bl.data),
        "coarse_mass_bl":(["time_coarse"], coarse_mass_bl.data),
        "aitken_mass_ft":(["time_coarse"], aitken_mass_ft.data),
        "accumulation_mass_ft":(["time_coarse"], accumulation_mass_ft.data),
        "coarse_mass_ft":(["time_coarse"], coarse_mass_ft.data),
        # "cloud_droplet_number":(["time_coarse"], cloud_droplet_number.data),
        # "rain_droplet_number":(["time_coarse"], rain_droplet_number.data),
        "aitken_number_bl":(["time_coarse"], aitken_number_bl.data),
        "accumulation_number_bl":(["time_coarse"], accumulation_number_bl.data),
        "coarse_number_bl":(["time_coarse"], coarse_number_bl.data),
        "aitken_number_ft":(["time_coarse"], aitken_number_ft.data),
        "accumulation_number_ft":(["time_coarse"], accumulation_number_ft.data),
        "coarse_number_ft":(["time_coarse"], coarse_number_ft.data),

        },
        coords={"time_fine": time_fine, 
                "time_mid": time_mid, 
                "time_coarse": time_coarse,
                "x": x,
                "y": y,
                "z": z,
                "zn": zn
               })

    # del lwp_mask_2d, total_lwp, cloudy_lwp, cloud_frac, deltheta_thresh_heights
    for var_ds in [ # rwp,
    #                cloud_droplet_mass,rain_droplet_mass,activated_aerosol_mass,
    #                aitken_mass,accumulation_mass,coarse_mass,
    #                cloud_droplet_number,rain_droplet_number,
    #                aitken_number,accumulation_number,coarse_number,
                   time_fine,time_mid,time_coarse,x,y]:
        var_ds.close()

    print("Saving to netcdf")
    if ice:
        post_processed_ds.to_netcdf(f"{sct_dir}/ice/{splits[0]}/{member_key}/sct_{member_key}_pp.nc",mode="w")
    else:
        # post_processed_ds.to_netcdf(f"{sct_dir}/new_processed/sct_em{count}_pp.nc",mode="w")
        
        post_processed_ds.to_netcdf(f"{sct_dir}/new_processed/sct_{member_key}_pp.nc",mode="w")
    post_processed_ds.close()
    print(f"Completed {member_key}!")
    

def main():
    # Process main ensemble
    
    # index_range = [1,4,5,6,8]
    # key_iterator = [f"{key}{index}" for index in index_range]
    # for member_key in key_iterator:
    #     create_dataset(member_key)

    key = "em"
    index_range = range(0, 61)
    key_iterator = [f"{key}{index}" for index in index_range]
    for member_key in key_iterator:
        if key_iterator in ["em27", "em10"]:
            continue
        else:
            create_dataset(member_key)

    # key = "val"
    # index_range = range(24)
    # key_iterator = [f"{key}{index}" for index in index_range]
    # for member_key in key_iterator:
    #     create_dataset(member_key, count=count)
    #     count += 1

    # key = "xtra"
    # index_range = range(12)
    # key_iterator = [f"{key}{index}" for index in index_range]
    # for member_key in key_iterator:
    #     create_dataset(member_key, count=count)
    #     count += 1

    # key_iterator = [#"em0_0", "em0_1", "em0_2", "em0_3", 
    #                 #"em34_1", "em34_2", "em34_3", "em34_4",
    #                 "em80_0", "em80_1", "em80_2", "em80_3", "em80_4",
    #                "em85_0", "em85_1", "em85_2", "em85_3", "em85_4"]
    # for member_key in key_iterator:
    #     create_dataset(member_key, ice=True)

if __name__=="__main__":
    main()