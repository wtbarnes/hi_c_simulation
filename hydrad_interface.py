"""
Model interface for the HYDrodynamics and RADiation (HYDRAD) code
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
import sunpy.sun.constants as sun_const

from synthesizAR.interfaces.ebtel import power_law_transform
from hydrad_tools.configure import Configure


class HYDRADInterface(object):
    
    def __init__(self,base_config,hydrad_dir,output_dir):
        self.name = 'HYDRAD'
        self.base_config = base_config
        self.hydrad_dir = hydrad_dir
        self.output_dir = output_dir
        self.duration = 200.0 * u.s
        self.stress = 0.3
        self.max_grid_cell = 1e8*u.cm
    
    def configure_input(self,loop):
        config = self.base_config.copy()
        # General configuration
        config['general']['loop_length'] = loop.full_length
        # This makes sure that the chromosphere does not take up the whole loop
        config['general']['footpoint_height'] = 0.5 * min(10*u.Mm, 0.5*loop.full_length)
        config['initial_conditions']['heating_location'] = loop.full_length / 2.
        config['grid']['minimum_cells'] = int(loop.full_length / self.max_grid_cell)
        # Gravity and cross-section coefficients
        config['general']['tabulated_gravity_profile'] = self.get_gravity_coefficients(loop)
        config['general']['tabulated_cross_section_profile'] = self.get_cross_section_coefficients(loop)
        # Heating configuration
        events = []
        rates = self.get_heating_rates(loop)
        twaits = self.get_waiting_times(rates)
        cumulative_time = 0*u.s
        for r,t in zip(rates,twaits):
            events.append({
                'time_start': cumulative_time.copy(),
                'rise_duration': self.duration/2.,
                'decay_duration': self.duration/2.,
                'total_duration': self.duration,
                'location': loop.full_length / 2.,
                'scale_height': 1e300*u.cm,
                'rate': r,
            })
            cumulative_time += t + self.duration
        # Set the heating events here
        config['heating']['events'] = events
        # Setup configuration, run IC code
        c = Configure(config)
        c.setup_simulation(self.output_dir, base_path=self.hydrad_dir, name=loop.name, verbose=False)
    
    def load_results(self,loop):
        # return time, electron_temperature, ion_temperature, density, velocity
        pass
    
    def get_cross_section_coefficients(self, loop):
        s_norm = loop.field_aligned_coordinate / loop.full_length
        return np.polyfit(s_norm, loop.field_strength, 6)[::-1]
    
    def get_gravity_coefficients(self, loop):
        s_norm = loop.field_aligned_coordinate / loop.full_length
        s_hat = (np.gradient(loop.coordinates.cartesian.xyz, axis=1) 
                 / np.linalg.norm(np.gradient(loop.coordinates.cartesian.xyz, axis=1), axis=0))
        r_hat = u.Quantity(np.stack([np.sin(loop.coordinates.spherical.lat)*np.cos(loop.coordinates.spherical.lon),
                                     np.sin(loop.coordinates.spherical.lat)*np.sin(loop.coordinates.spherical.lon),
                                     np.cos(loop.coordinates.spherical.lat)]))
        g_parallel = -sun_const.surface_gravity.cgs * ((const.R_sun.cgs / loop.coordinates.spherical.distance)**2) * (r_hat * s_hat).sum(axis=0)
        return np.polyfit(s_norm, g_parallel, 6)[::-1]
    
    def get_average_waiting_time(self,loop):
        # Approximate the loop cooling time, hardcoded for intermediate frequency for now
        return loop.full_length / (80*u.Mm) * 4000*u.s 
    
    def get_number_events(self,loop):
        twait = self.get_average_waiting_time(loop)
        return int(np.ceil(self.base_config['general']['total_time'] / (self.duration + twait)))
    
    def get_heating_rates(self,loop):
        rate_max = (self.stress * loop.field_strength.max().value)**2 / (8.*np.pi) * u.erg / (u.cm**3)
        rate_max /= (self.duration / 2.)
        rate_min = rate_max / 100.0
        return power_law_transform(np.random.rand(self.get_number_events(loop)), rate_min, rate_max, -2.5) 
    
    def get_waiting_times(self, rates):
        prop_const = (self.base_config['general']['total_time'] - rates.shape[0] * self.duration) / rates.sum()
        return rates * prop_const
    