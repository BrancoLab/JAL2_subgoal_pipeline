import numpy as np

from behave_analysis.utils.rayleigh.load_rayleigh import extract_rayleigh_path, load_rayleigh_data
from behave_analysis.analyze.filtering_data.filtering_functions import identify_conditions, identify_angles
import matplotlib.pyplot as plt


def retrieve_specific_rayleigh_data(condition, angle, session, cluster_type, data_type="Rayleigh") -> dict:
    """For a given condition and angle, retrieve the Rayleigh data for each cell
    
    Returns:
    -- rayleigh_data (dict) with keys "one" and "two" and values of the compartment values for each cell"""
    path = extract_rayleigh_path(session, cluster_type, condition=condition, file_name=angle + "_Rayleigh.arrow")
    data = load_rayleigh_data(path)
    magnitude = extract_compartment_values(data, data_type)
    one_compartments = [x[0] for x in magnitude]
    two_compartments = [x[1] for x in magnitude]
    rayleigh_data = {"one": one_compartments, "two": two_compartments}
    return rayleigh_data

def retrieve_rayleigh_data(session, cluster_type) -> dict:
    """"A function that loops through each condition and angle and retrieves the rayleigh data for each cell in each compartment
    
    Returns:
    -- dict_of_conditions (nested dict) with:
        First key is conditions, second key as angles, third key as "one" or "two" representing the compartment
        values of the compartment values for each cell"""
    conditions = identify_conditions(session)
    angles = identify_angles(session)
    dict_of_conditions = {}
    for condition in conditions:
        dict_of_angles = {}
        for angle in angles:
            dict_of_angles[angle] = retrieve_specific_rayleigh_data(condition, angle, session, cluster_type)
        dict_of_conditions[condition] = dict_of_angles
    return dict_of_conditions

def compare_condition_pdfs(session, cluster_type)-> dict:
    raise NotImplementedError
    


    # # Define the conditions
    # conditions = ["shelter_only", "barrier_pre_flip", "barrier_post_flip"]

    # dic_one = {}
    # dic_two = {}
    # for condition in conditions:
    #     path = extract_rayleigh_path(session, cluster_type, condition=condition, file_name="hdir_Rayleigh.arrow")
    #     data = load_rayleigh_data(path)
    #     magnitude = extract_compartment_values(data, "Rayleigh")
    #     one_compartments = [x[0] for x in magnitude]
    #     two_compartments = [x[1] for x in magnitude]
    #     dic_one[condition] = one_compartments
    #     dic_two[condition] = two_compartments

    # # Calculate deltas for each compartment
    # delta_one_shelter_pre = np.asarray(dic_one["shelter_only"]) - np.asarray(dic_one["barrier_pre_flip"])
    # delta_one_pre_post = np.asarray(dic_one["barrier_pre_flip"]) - np.asarray(dic_one["barrier_post_flip"])

    # delta_two_shelter_pre = np.asarray(dic_two["shelter_only"]) - np.asarray(dic_two["barrier_pre_flip"])
    # delta_two_pre_post = np.asarray(dic_two["barrier_pre_flip"]) - np.asarray(dic_two["barrier_post_flip"])

    # # Create subplots
    # fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # # Plot for first compartment: Delta between Shelter Only and Barrier Pre Flip
    # axs[0, 0].hist(delta_one_shelter_pre[delta_one_shelter_pre >= 0], bins=100, color="b", alpha=0.7, label="Positive")
    # axs[0, 0].hist(delta_one_shelter_pre[delta_one_shelter_pre < 0], bins=100, color="r", alpha=0.7, label="Negative")
    # axs[0, 0].set_title("Compartment 1: Shelter Only vs. Barrier Pre Flip")
    # axs[0, 0].legend()

    # # Plot for first compartment: Delta between Barrier Pre Flip and Post Flip
    # axs[0, 1].hist(delta_one_pre_post[delta_one_pre_post >= 0], bins=100, color="b", alpha=0.7, label="Positive")
    # axs[0, 1].hist(delta_one_pre_post[delta_one_pre_post < 0], bins=100, color="r", alpha=0.7, label="Negative")
    # axs[0, 1].set_title("Compartment 1: Barrier Pre Flip vs. Post Flip")
    # axs[0, 1].legend()

    # # Plot for second compartment: Delta between Shelter Only and Barrier Pre Flip
    # axs[1, 0].hist(delta_two_shelter_pre[delta_two_shelter_pre >= 0], bins=100, color="b", alpha=0.7, label="Positive")
    # axs[1, 0].hist(delta_two_shelter_pre[delta_two_shelter_pre < 0], bins=100, color="r", alpha=0.7, label="Negative")
    # axs[1, 0].set_title("Compartment 2: Shelter Only vs. Barrier Pre Flip")
    # axs[1, 0].legend()

    # # Plot for second compartment: Delta between Barrier Pre Flip and Post Flip
    # axs[1, 1].hist(delta_two_pre_post[delta_two_pre_post >= 0], bins=100, color="b", alpha=0.7, label="Positive")
    # axs[1, 1].hist(delta_two_pre_post[delta_two_pre_post < 0], bins=100, color="r", alpha=0.7, label="Negative")
    # axs[1, 1].set_title("Compartment 2: Barrier Pre Flip vs. Post Flip")
    # axs[1, 1].legend()

    # # Display the plot
    # plt.tight_layout()
    # plt.show()

    # probabilities = one_compartments / np.sum(one_compartments)
    # plt.bar(range(len(probabilities)), probabilities)
    # plt.xlabel('Neuron Index')
    # plt.ylabel('Probability')
    # plt.title('Probability Distribution of Neuron Activations')
    # plt.show()


def extract_compartment_values(data, column_name: str) -> tuple:
    """Extract compartment values from a polars DataFrame

    Returns:
    -- compartment values (tuple) for each cell e.g ((x1, y1), (x2, y2), ...
    first value is shelter zone, second value is threat zone"""
    first = [x[0] for x in data[column_name]]
    second = [x[1] for x in data[column_name]]
    output = tuple(zip(first, second))
    assert len(output) == len(data), "Length of extracted compartment values does not match length of data"
    return output
