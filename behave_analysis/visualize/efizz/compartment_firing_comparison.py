'''In miller 2019 paper they compared the firing rate of individual neurons for
left and right route selections on a continuous t-maze task. They found that
neurons in the RSP were selective for the route selection. Here we will try to
do the same thing but for the compartment firing rates.

TODO:
-- revisit, inital plot is not very informative, will need to re add this function to the pipeline so its executed'''

import os

import polars as pl
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from behave_analysis.utils.creating_directories import make_directory
from behave_analysis.utils.heatplot_utils import add_features, filter_outside_arena_tracking_for_video_and_spike_data
from behave_analysis.analyze.filtering_data.filtering_functions import filter_video_dataframe


# Because of memory constraints, let's only keep the columns we need for this module
COLUMNS_TO_KEEP = [
    "mouse_x_position",
    "mouse_y_position",
    "spike_clusters",
    "spike_count",
    "OutofshelterIdx",
    "EscapePeriod",
    "shelter",
    "barrier_present",
    "barrier_flipped",
]

def compare_fr_compartments(video_and_spike_data: pl.DataFrame, conditions: list, save_base: str, session: object):
    '''Compare the firing rates of compartments for different conditions by neuron.
    
    Args:
        video_and_spike_data (pl.DataFrame): A DataFrame containing spike counts per frame and neuron in a single session
        conditions (list): A list of conditions to divide the heatmap into, each condition is a string
        save_base (str): The base directory to save the heatmaps to
        session (Session): A Session object containing metadata about the session

    '''
    logger.info("Comparing firing rates of compartments for different conditions")
    
    # Prepare the data
    save_path = make_directory(os.path.join(save_base, "compartment_firing_comparison"))
    data = video_and_spike_data.select(COLUMNS_TO_KEEP)
    del video_and_spike_data
    unit_ids = data["spike_clusters"].unique().to_numpy()
    data = filter_outside_arena_tracking_for_video_and_spike_data(video_and_spike_data=data, session=session)
    
    # Add new column to the dataframe to indicate which compartment the mouse is in
    data = data.with_columns([
        pl.when(data["mouse_y_position"] > 512).then(2).otherwise(1).alias("compartment"),
    ])

    clu_dictionary = {}
    # Loop through each cluster
    for clu in unit_ids:
        print(clu)
        # fig, axs = plt.subplots(nrows=1, ncols=len(conditions), figsize=(15, 7), sharey=True, sharex=True)
        con_dictionary = {}
        for _, condition in enumerate(conditions):
            filtered_df = filter_video_dataframe(dataframe=data, condition=condition)
            clu_df = filtered_df.filter(pl.col("spike_clusters") == clu)

            # Skip if no data for this unit and condition
            if clu_df.is_empty() or sum(clu_df["spike_count"]) == 0:
                continue
            
            # Count how many spikes were fired in each compartment
            compartment_counts = clu_df.groupby("compartment", maintain_order = True).agg(pl.sum("spike_count"))
            
            # Count how many frames were spent in each compartment
            compartment_frames = clu_df.groupby("compartment", maintain_order = True).agg(pl.count("spike_count"))
            compartment_frames = compartment_frames.rename({"spike_count": "total_frames"})
            
            joined = compartment_counts.join(compartment_frames, on="compartment")
            joined = joined.with_columns((pl.col("spike_count") / pl.col("total_frames")).alias("normalised_spikes"))
              
            # Convert to firing rate for c1
            c1_frames = joined.filter(pl.col("compartment") == 1)["total_frames"].to_numpy()[0]
            c1_factor = c1_frames / 40
            
            if c1_factor > 1:
                c1_hz = joined.filter(pl.col("compartment") == 1)["spike_count"].to_numpy()[0] / c1_factor
            if c1_factor <= 1:
                c1_hz = joined.filter(pl.col("compartment") == 1)["spike_count"].to_numpy()[0] * (1/c1_factor)
                
            # Convert to firing rate for c2
            c2_frames = joined.filter(pl.col("compartment") == 2)["total_frames"].to_numpy()[0]
            c2_factor = c2_frames / 40
            
            if c2_factor > 1:
                c2_hz = joined.filter(pl.col("compartment") == 2)["spike_count"].to_numpy()[0] / c2_factor
            if c2_factor <= 1:
                c2_hz = joined.filter(pl.col("compartment") == 2)["spike_count"].to_numpy()[0] * (1/c2_factor)
                
            con_dictionary[condition] = {"c1": c1_hz, "c2": c2_hz}
            
            # check hz is not negative or greater than 200
            
        clu_dictionary[clu] = con_dictionary
            
    # Now turn the dictionary of dictionaries into a DataFrame
    data = []
    for clu, conditions in clu_dictionary.items():
        for condition, values in conditions.items():
            row = {'clu': clu, 'condition': condition}
            row.update(values)
            data.append(row)
            
    # Plotting
    # Filter for "all_time" condition
    df = pd.DataFrame(data)
    df_all_time = df[df['condition'] == 'barrier_pre_flip']

    # Melt the DataFrame to long format
    df_long = df_all_time.melt(id_vars=['clu', 'condition'], value_vars=['c1', 'c2'], var_name='compartment', value_name='value')

    import seaborn as sns

    # Plotting
    sns.violinplot(data=df_long, x='compartment', y='value', color=".8")

    # Now plot each point and draw lines
    for clu in df_all_time['clu'].unique():
        # Filter data for the current cluster
        cluster_data = df_long[df_long['clu'] == clu]
        
        # Determine positions for c1 and c2 points
        c1_value = cluster_data[cluster_data['compartment'] == 'c1']['value'].values
        c2_value = cluster_data[cluster_data['compartment'] == 'c2']['value'].values
        
        # Check if both c1 and c2 have values before plotting
        if len(c1_value) > 0 and len(c2_value) > 0:
            # Plotting points
            plt.scatter(x=[0]*len(c1_value), y=c1_value, color='blue', s=10)
            plt.scatter(x=[1]*len(c2_value), y=c2_value, color='blue', s=10)
            
            # Drawing lines
            for y1, y2 in zip(c1_value, c2_value):
                plt.plot([0, 1], [y1, y2], color='grey', linewidth=1)

    plt.title('Violin plot with points and lines for "all_time" condition')
    plt.xticks([0, 1], ['c1', 'c2'])  # Set the x-ticks to correspond to c1 and c2
    plt.show()
    
            
            
            
            

    

