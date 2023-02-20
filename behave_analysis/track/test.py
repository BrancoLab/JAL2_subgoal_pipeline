import pandas as pd

def load_dlc_tracking(tracking_file: str):
    """
        load and unstack tracking data from a DLC file
    """
    tracking = pd.read_hdf(tracking_file)

    bodyparts = tracking.iloc[0].index.levels[1]
    scorer = tracking.iloc[0].index.levels[0]

    tracking = tracking.unstack()

    trackings = {}
    for bp in bodyparts:
        trackings[bp] = {c: tracking.loc[scorer, bp, c].values for c in ["x", "y", "likelihood"]}
    
    return trackings

file = r"D:\efizz\YT6240_23jan19\cam1DLC_resnet50_NPX_7May20shuffle1_500000.h5"


print(load_dlc_tracking(file)['left_ear'])

