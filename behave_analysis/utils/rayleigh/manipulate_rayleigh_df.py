def extract_compartment_values(data, column_name: str) -> tuple:
    """Extract compartment values from a polars DataFrame

    Returns:
    -- compartment values (tuple) for each cell e.g ((x1, y1), (x2, y2), ...
    first value is shelter zone, second value is threat zone"""
    first = [x[0] for x in data[column_name]]
    second = [x[1] for x in data[column_name]]
    output = tuple(zip(first, second))
    assert len(output) == len(
        data
    ), "Length of extracted compartment values does not match length of data"
    return output