"""
Detector class for the Hi-C imager
"""
import os
import pickle

import toolz
import numpy as np
import h5py
from scipy.interpolate import splrep,splev,interp1d
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.console import ProgressBar
from sunpy.map import Map
import distributed

from synthesizAR.instruments import InstrumentSDOAIA,InstrumentBase
from synthesizAR.atomic import EmissionModel,Element
from synthesizAR.util import SpatialPair, get_keys, is_visible


class InstrumentHiC(InstrumentBase):
    """
    High Resolution Coronal Imager (Hi-C) instrument
    """
    
    def __init__(self, observing_time, observer_coordinate, fov={}, **kwargs):
        self._fov = fov
        self.fits_template['telescop'] = 'HiC'
        self.fits_template['detector'] = 'HiC'
        self.fits_template['waveunit'] = 'angstrom'
        self.name = 'Hi_C'
        self.channels = [{'wavelength': 171*u.angstrom, 'name': '171', 'intstrument_label': 'HiC',
                          'gaussian_width': {'x': 0.962*u.pixel, 'y': 0.962*u.pixel}},]
        self.cadence = 6.0*u.s
        self.resolution = kwargs.get('resolution',
                                     SpatialPair(x=0.1*u.arcsec/u.pixel, y=0.1*u.arcsec/u.pixel, z=None))
        super().__init__(observing_time, observer_coordinate)
        self._setup_channels()
        
    def _setup_channels(self,):
        """
        Configure the wavelength response function of the instrument
        """
        # Approximate Hi-C 171 wavelength response using Hi-C 193 response, AIA 193
        # response, and AIA 171 response
        with open('HiC_193.ins','r') as f: # Assumes instrument file is in same folder
            lines = f.readlines()
            n = int(lines[0])
            wvl_193_tmp = np.zeros((n,))
            resp_hi_c_193 = np.zeros((n,))
            for i,l in enumerate(lines[1:n+1]):
                wvl_193_tmp[i] = l.strip().split()[0]
                resp_hi_c_193[i] = l.strip().split()[1]
        delta_lambda=20
        wvl_193 = 193 + np.linspace(-delta_lambda,delta_lambda,2000)
        wvl_171 = 171 + np.linspace(-delta_lambda,delta_lambda,2000)
        resp_hi_c_193 = splev(wvl_193,splrep(wvl_193_tmp,resp_hi_c_193))
        aia = InstrumentSDOAIA([0,1]*u.s, self.observer_coordinate)
        resp_aia_193 = splev(wvl_193, aia.channels[3]['wavelength_response_spline'])
        resp_aia_171 = splev(wvl_171, aia.channels[2]['wavelength_response_spline'])
        resp_hi_c_171 = resp_hi_c_193 * resp_aia_171 / resp_aia_193
        # Set channel properties
        self.channels[0]['wavelength_range'] = None
        self.channels[0]['wavelength_response_spline'] = splrep(wvl_171,resp_hi_c_171)
        
    def _get_fov(self, ar_map):
        min_x, max_x, min_y, max_y = super()._get_fov(ar_map)
        min_x = self._fov.get('min_x', min_x)
        max_x = self._fov.get('max_x', max_x)
        min_y = self._fov.get('min_y', min_y)
        max_y = self._fov.get('max_y', max_y)
        return min_x, max_x, min_y, max_y
        
    
    def build_detector_file(self, file_template, dset_shape, chunks, *args, parallel=False):
        """
        Allocate space for counts data.
        """
        additional_fields = ['{}'.format(channel['name']) for channel in self.channels]
        super().build_detector_file(file_template, dset_shape, chunks, *args,
                                    additional_fields=additional_fields, parallel=parallel)
    
    def interpolate(self, y, loop):
        """
        Interpolate in time and write to HDF5 file.
        """
        f_t = interp1d(loop.time.value, y.value, axis=0, kind='linear', fill_value='extrapolate')
        interpolated_y = f_t(self.observing_time.value)
        return interpolated_y * y.unit
    
    def flatten_parallel(self, loops, interpolated_loop_coordinates, emission_model=None):
        """
        Compute intensity counts in parallel with Dask, save as a "flattened" column
        """
        # Setup scheduler
        client = distributed.get_client()
        start_indices = np.insert(np.array(
            [s.shape[0] for s in interpolated_loop_coordinates]).cumsum()[:-1], 0, 0)
        if emission_model is None:
            raise ValueError('Emission Model required')

        futures = {}
        for channel in self.channels:
            # Flatten emissivities for appropriate channel
            flat_emiss = client.scatter(self.flatten_emissivities(channel, emission_model))
            # Build partials for functions
            #partial_counts = toolz.curry(self.calculate_counts)(
            #    channel, emission_model=emission_model, flattened_emissivities=flat_emiss)
            #partial_write = toolz.curry(self.write_to_hdf5)(dset_name=channel['name'])
            # Map functions to iterables
            futures[channel['name']] = []
            for i,loop in enumerate(loops):
                y = client.submit(self.calculate_counts, channel, loop, emission_model, flat_emiss, pure=False)
                interp_y = client.submit(self.interpolate, y, loop, pure=False)
                futures[channel['name']].append(
                    client.submit(self.write_to_hdf5, interp_y, start_indices[i], channel['name'], pure=False))

        return futures
    
    def flatten_serial(self, loops, interpolated_loop_coordinates, hf, emission_model=None):
        if emission_model is None:
            raise ValueError('Emission model is required')
        with ProgressBar(len(self.channels)*len(loops),ipython_widget=True) as progress:
            for channel in self.channels:
                start_index = 0
                dset = hf[channel['name']]
                flattened_emissivities = self.flatten_emissivities(channel, emission_model)
                for loop, interp_s in zip(loops, interpolated_loop_coordinates):
                    c = self.calculate_counts(channel, loop, emission_model, flattened_emissivities)
                    y = self.interpolate_and_store(c, loop, interp_s)
                    self.commit(y, dset, start_index)
                    start_index += interp_s.shape[0]
                    progress.update()
    
    def calculate_counts(self, channel, loop, emission_model, flattened_emissivities):
        """
        Calculate Hi-C intensity as a function of time and space
        """
        density = loop.density
        electron_temperature = loop.electron_temperature
        counts = np.zeros(electron_temperature.shape)
        itemperature, idensity = emission_model.interpolate_to_mesh_indices(loop)
        for ion, flat_emiss in zip(emission_model, flattened_emissivities):
            if flat_emiss is None:
                continue
            ionization_fraction = emission_model.get_ionization_fraction(electron_temperature, ion)
            tmp = np.reshape(map_coordinates(flat_emiss.value, np.vstack([itemperature, idensity]), order=1, prefilter=False),
                             electron_temperature.shape)
            tmp = u.Quantity(np.where(tmp < 0., 0., tmp), flat_emiss.unit)
            counts_tmp = ion.abundance*0.83/(4*np.pi*u.steradian)*ionization_fraction*density*tmp
            if not hasattr(counts, 'unit'):
                counts = counts*counts_tmp.unit
            counts += counts_tmp
        return counts
    
    @staticmethod
    def flatten_emissivities(channel, emission_model):
        """
        Compute product between wavelength response and emissivity for all ions
        """
        flattened_emissivities = []
        for ion in emission_model:
            wavelength, emissivity = emission_model.get_emissivity(ion)
            if wavelength is None or emissivity is None:
                flattened_emissivities.append(None)
                continue
            interpolated_response = splev(wavelength.value, channel['wavelength_response_spline'],
                                          ext=1)
            em_summed = np.dot(emissivity.value, interpolated_response)
            unit = emissivity.unit*u.count/u.photon*u.steradian/u.pixel*u.cm**2
            flattened_emissivities.append(u.Quantity(em_summed, unit))

        return flattened_emissivities
    
    def detect(self, channel, i_time, header, bins, bin_range):
        """
        For a given timestep, map the intensity along the loop to the 3D field and
        return the Hi-C data product.

        Parameters
        ----------
        channel : `dict`
        i_time : `int`
        header : `~sunpy.util.metadata.MetaDict`
        bins : `~synthesizAR.util.SpatialPair`
        bin_range : `~synthesizAR.util.SpatialPair`

        Returns
        -------
        AIA data product : `~sunpy.map.Map`
        """
        with h5py.File(self.counts_file, 'r') as hf:
            weights = np.array(hf[channel['name']][i_time, :])
            units = u.Unit(get_keys(hf[channel['name']].attrs, ('unit','units')))

        hpc_coordinates = self.total_coordinates
        dz = np.diff(bin_range.z)[0].cgs / bins.z * (1. * u.pixel)
        visible = is_visible(hpc_coordinates, self.observer_coordinate)
        hist, _, _ = np.histogram2d(hpc_coordinates.Tx.value, hpc_coordinates.Ty.value,
                                    bins=(bins.x.value, bins.y.value),
                                    range=(bin_range.x.value, bin_range.y.value),
                                    weights=visible * weights * dz.value)
        header['bunit'] = (units * dz.unit).to_string()

        counts = gaussian_filter(hist.T, (channel['gaussian_width']['y'].value,
                                          channel['gaussian_width']['x'].value))
        return Map(counts.astype(np.float32), header)
    
    
class CustomEmissionModel(EmissionModel):
    
    def calculate_ionization_fraction(self, field, savefile, interface=None, **kwargs):
        """
        Compute population fractions for each ion and for each loop.

        Find the fractional ionization for each loop in the model as defined by the loop
        model interface. If no interface is provided, the ionization fractions are calculated
        assuming ionization equilibrium.

        Parameters
        ----------
        field : `~synthesizAR.Field`
        savefile : `str`
        interface : optional
            Hydrodynamic model interface

        Other Parameters
        ----------------
        log_temperature_dex : `float`, optional
        """
        self.ionization_fraction_savefile = savefile
        # Create sufficiently fine temperature grid
        dex = kwargs.get('log_temperature_dex', 0.01)
        logTmin = np.log10(self.temperature.value.min())
        logTmax = np.log10(self.temperature.value.max())
        temperature = u.Quantity(10.**(np.arange(logTmin, logTmax+dex, dex)), self.temperature.unit)
        
        if interface is not None:
            return interface.calculate_ionization_fraction(field, self, temperature=temperature,
                                                           **kwargs)
        unique_elements = list(set([ion.element_name for ion in self]))
        # Calculate ionization equilibrium for each element and interpolate to each loop
        notebook = kwargs.get('notebook', True)
        with h5py.File(self.ionization_fraction_savefile, 'a') as hf:
            dset = hf.create_dataset('temperature', data=temperature.value)
            dset.attrs['unit'] = temperature.unit.to_string()
            for el_name in unique_elements:
                element = Element(el_name, temperature)
                ioneq = element.equilibrium_ionization()
                dset = hf.create_dataset(element.element_name, data=ioneq.value)
                dset.attrs['unit'] = ioneq.unit.to_string()
                dset.attrs['description'] = 'uninterpolated equilibrium ionization fractions'

    def get_ionization_fraction(self, loop_electron_temperature, ion):
        """
        Get ionization state from the ionization balance equations.

        Get ion population fractions for a particular loop and element. This can be either the
        equilibrium or the non-equilibrium ionization fraction, depending on which was calculated.

        Parameters
        ----------
        loop_electron_temperature : `~astropy.units.quantity`
        ion : `~fiasco.Ion`
        """
        with h5py.File(self.ionization_fraction_savefile, 'r') as hf:
            temperature = hf['temperature'][:]
            tunit = get_keys(hf['temperature'].attrs, ('unit', 'units'))
            dset = hf[ion.element_name]
            ionization_fraction = dset[:, ion.charge_state]
            unit = get_keys(dset.attrs, ('unit', 'units'))
        temperature = u.Quantity(temperature, tunit)
        #dex = 0.01
        #logTmin = np.log10(self.temperature.value.min())
        #logTmax = np.log10(self.temperature.value.max())
        #temperature = u.Quantity(10.**(np.arange(logTmin, logTmax+dex, dex)), self.temperature.unit)
        #ionization_fraction = Element(ion.element_name, temperature).equilibrium_ionization()[:,ion.charge_state].value
        f_interp = interp1d(temperature.value, ionization_fraction, fill_value='extrapolate')
        ionization_fraction = f_interp(loop_electron_temperature.to(temperature.unit).value)
        ionization_fraction = np.where(ionization_fraction < 0., 0., ionization_fraction)

        return u.Quantity(ionization_fraction,unit)
    