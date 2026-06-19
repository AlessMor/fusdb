"""Plasma temperature state relations."""

def calc_average_ion_temp_from_temperature_ratio(average_electron_temp, ion_to_electron_temp_ratio):
    """cfspopcon: average_ion_temp = average_electron_temp * ion_to_electron_temp_ratio."""
    return average_electron_temp * ion_to_electron_temp_ratio
