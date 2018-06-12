"""
Interpolate HYDRAD results to a uniform spatial grid at each timestep and output as HDF5
"""
import os
import argparse

import numpy as np
from scipy.interpolate import splev,splrep
import h5py
import astropy.units as u
import astropy.constants as const

from hydrad_tools.parse import Strand


@u.quantity_input
def interpolate_to_uniform_grid(hydrad_root, delta_s: u.cm,):
    """
    Interpolate temperature, density, and velocity to 
    uniform grid
    """
    # Create the strand
    s = Strand(hydrad_root, read_amr=False)
    # Create uniform coordinate
    s_uniform = np.arange(0,s.loop_length.to(u.cm).value, delta_s.to(u.cm).value)*u.cm
    # Preallocate space for arrays
    electron_temperature = np.zeros(s_uniform.shape+s.time.shape)
    ion_temperature = np.zeros(s_uniform.shape+s.time.shape)
    density = np.zeros(s_uniform.shape+s.time.shape)
    velocity = np.zeros(s_uniform.shape+s.time.shape)
    # Interpolate each quantity at each timestep
    for i,_ in enumerate(s.time):
        p = s[i]
        coord = p.coordinate.to(u.cm).value
        # Temperature
        tsk = splrep(coord, p.electron_temperature.to(u.K).value,)
        electron_temperature[:,i] = splev(s_uniform.value, tsk, ext=0)
        tsk = splrep(coord, p.ion_temperature.to(u.K).value,)
        ion_temperature[:,i] = splev(s_uniform.value, tsk, ext=0)
        # Density
        tsk = splrep(coord, p.electron_density.to(u.cm**(-3)).value)
        density[:,i] = splev(s_uniform.value,tsk,ext=0)
        # Velocity
        tsk = splrep(coord, p.velocity.to(u.cm/u.s).value,)
        velocity[:,i] = splev(s_uniform.value, tsk, ext=0)
        
    return s.time, s_uniform, electron_temperature*u.K, ion_temperature*u.K, density*u.cm**(-3), velocity*u.cm/u.s


def save_to_hdf5(filename, t, s, Te, Ti, n, v):
    """
    Save electron temperature, ion_temperature, density, velocity,
    time, and coordinate to file
    """
    with h5py.File(filename, 'w') as hf:
        dset = hf.create_dataset('coordinate', data=s.value)
        dset.attrs['unit'] = s.unit.to_string()
        dset = hf.create_dataset('time', data=t.value)
        dset.attrs['unit'] = t.unit.to_string()
        dset = hf.create_dataset('electron_temperature', data=Te.value)
        dset.attrs['unit'] = Te.unit.to_string()
        dset = hf.create_dataset('ion_temperature', data=Ti.value)
        dset.attrs['unit'] = Ti.unit.to_string()
        dset = hf.create_dataset('density', data=n.value)
        dset.attrs['unit'] = n.unit.to_string()
        dset = hf.create_dataset('velocity', data=v.value)
        dset.attrs['unit'] = v.unit.to_string()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--hydrad_root', help='Path to HYDRAD directory',type=str)
    parser.add_argument('--output_file', help='Path to output file for reduced results',type=str)
    args = parser.parse_args()
    # Get delta s for Hi-C
    ds = (((0.1*u.arcsec).to(u.radian).value * (const.au - const.R_sun)).to(u.cm))
    # Interpolate results
    t,s,Te,Ti,n,v = interpolate_to_uniform_grid(args.hydrad_root, ds)
    # Save to HDF5 file
    save_to_hdf5(args.output_file, t, s, Te, Ti, n, v)
    