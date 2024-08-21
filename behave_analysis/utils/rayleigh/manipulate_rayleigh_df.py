def extract_compartment_values(data, column_name: str) -> tuple:
    """Extract compartment values from a polars DataFrame

    Returns:
    -- compartment values (tuple) for each cell e.g ((x1, y1), (x2, y2), ...
    first value is shelter zone, second value is threat zone"""
    first = [x[0] for x in data[column_name]]  # shelter zone
    second = [x[1] for x in data[column_name]]  # threat zone
    output = tuple(zip(first, second))
    assert len(output) == len(data), "Length of extracted compartment values does not match length of data"
    return output


def extract_firing_rates(dataframe, analyze_efizz_settings) -> list:
    """Extract firing rates and seperate them into two lists. Currently the first n in the cell are assigned to the shelter zone,
    the next n are assigned to the threat zone. This function decouples that logic"""
    bin_edges = analyze_efizz_settings.number_of_bins
    number_of_bins = bin_edges - 1
    shelter_zone_firing_rates = [cell[:number_of_bins] for cell in dataframe["angle_firing_hist"].to_numpy()]
    threat_zone_firing_rates = [cell[number_of_bins:] for cell in dataframe["angle_firing_hist"].to_numpy()]
    assert len(shelter_zone_firing_rates[0]) + len(threat_zone_firing_rates[0]) == number_of_bins * 2, "Length of extracted firing rates does not match number of bins"
    return shelter_zone_firing_rates, threat_zone_firing_rates
