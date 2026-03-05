import pandas as pd
from dataclasses import asdict
import numpy as np
import uuid
from loguru import logger
import ast
import os

SETTINGS_AE = ["stim_type", "redo_compute", "cluster_type", "show_plots", "condition_types", "compartment_split", "parallel_pool_linshit"]

def check_database_for_same_run(db_settings, results_csv_name, settings):

    do_analysis = True
    # check if database file exists
    results_csv = results_csv_name
    if os.path.exists(results_csv):
        database = pd.read_csv(results_csv)
        # check if there is a run with the same settings as the current ones
        matched_results = check_database_for_matched_results(database, db_settings)
        if len(matched_results) > 0:
            # if there is a match, print the name of the matched run and skip the analysis....
            logger.info(f"Found {len(matched_results)} matched results in database for current settings: {matched_results} in the folder {results_csv_name}")
            do_analysis = False
            if settings.redo_compute:
                # ... unless you have chosen to redo the analysis anyway!
                logger.info("You have chosen to redo the analysis anyway!")
                do_analysis = True
    else:
        # if database doesn't exist, create it and add the current run to the database
        logger.info(f"No existing database found for escape tuning results at {results_csv_name}, will compute escape tuning and save to new database.")
        database = pd.DataFrame([])

    # if we are doing the replay analysis, add the current run to the database with a new hexadecimal name
    if do_analysis:
        hexaname = generate_run_id()
    
    return database, do_analysis, hexaname

def check_database_for_matched_results(database: pd.DataFrame, settings_to_check: dict):
    """Check if we have already run the analysis with these settings!
    INPUTS:
        database: the pandas dataframe that is the database of runs
        settings_to_check: a dictionary of the settings we want to check for matches in the database"""

    # check if we have a row that matches all the settings
    matched_rows = find_matching_run(database, settings_to_check)
    
    # return hexadecimal name of the matched rows
    if np.sum(matched_rows) > 0:
        return database[matched_rows].name.values
    else:
        return []

def settings_to_check(settings_obj, analysis_type):
    """Given the settings object and the type of analysis,
    return a dict of the relevant settings to check for that analysis type.
    It assumes that the names of the settings for each analysis type start with the name of the analysis type (e.g. 'replay')
    INPUTS:
        settings_obj: the settings object that contains all the settings for the analysis
        analysis_type: a string or list of strings that the settings we want start with (e.g. 'replay', 'place_cell', 'LDA')"""

    settings_dict = asdict(settings_obj)

    settings_list = []
    if isinstance(analysis_type, str):
        settings_list = [s for s in settings_dict.keys() if s.startswith(analysis_type)]
    else:
        for at in analysis_type:
            settings_list.extend([s for s in settings_dict.keys() if s.startswith(at)])
    
    gen_settings_list = [s for s in settings_dict.keys() if s in SETTINGS_AE]
    settings_list.extend(gen_settings_list)

    settings_to_check_dict = {s: settings_dict[s] for s in settings_list}

    return settings_to_check_dict

def find_matching_run(database, settings_dict, saved_vars = []):
    """A function that chcks a database for rows with settings that match settings_dict.
    Optionally you can also ask to check the list of saved_vars for this row by passing a list of var names"""
    matched_rows = np.ones(len(database), dtype=bool)
    saved_vars_rows = np.ones(len(database), dtype=bool)

    # iterate over rows
    for row in range(len(database)):
        row_dict = database.iloc[row].to_dict()
        for setting_name, setting_value in settings_dict.items():
            if setting_name not in row_dict:
                matched_rows[row] = False
                break
            else:
                if row_dict[setting_name] != setting_value:
                    matched_rows[row] = False
                    break
        if len(saved_vars) > 0:
            data_vars = ast.literal_eval(row_dict["data_vars"]) if isinstance(row_dict["data_vars"], str) else row_dict["data_vars"]
            if data_vars != saved_vars:
                saved_vars_rows[row] = False

    if len(saved_vars) > 0:
        return matched_rows, saved_vars_rows 
    else:   
        return matched_rows

def add_run_to_database(dataframe, settings_dict, savepath, hexadecimal_name, saved_vars=None):
    """INPUTS:
        dataframe: the pandas dataframe that is the database of runs
        settings_dict: a dictionary of the settings for this run (best to filter the settings_obj with a settings_list of interest
        savepath: where the database csv is saved - it must exist!
        hexadecimal_name: the unique identifier for this run (e.g. a random 16 character hex string)"""
    # add a row to dataframe with the settings and the hexadecimal name
    row = {'name': hexadecimal_name}
    # Build row data
    for key, value in settings_dict.items():
        row[key] = value
    
    if not saved_vars is None:
        # add a column for variables saved to data
        row['data_vars'] = saved_vars
    
    # Add to dataframe
    new_df = pd.DataFrame([row])
    if dataframe.empty:
        dataframe = new_df
    else:
        dataframe = pd.concat([dataframe, new_df], ignore_index=True)
    
    # Save to CSV
    dataframe.to_csv(savepath, index=False)
    logger.info(f"Added run {hexadecimal_name} to tracker")


def generate_run_id() -> str:
        """Generate a random 16-character hex identifier."""
        return uuid.uuid4().hex[:16]