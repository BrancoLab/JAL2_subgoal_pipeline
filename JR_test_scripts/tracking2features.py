import numpy as np


'''From tracking data, extract random point positions and assign them to shelter or barrier'''

def tracking_to_features(tracking_data):
    features = {}
    ybins, y = np.unique(tracking_data["randP_loc"][:, 0], return_inverse=True)
    xbins, x = np.unique(tracking_data["randP_loc"][:, 1], return_inverse=True)
    heatmap = np.zeros(shape=(len(np.unique(tracking_data["randP_loc"][:, 0])), len(np.unique(tracking_data["randP_loc"][:, 1]))))

    # find random points that belong to shelter
    shelter_x_start = np.argmin(np.abs(xbins - tracking_data['shelter_loc'][0][1]))
    shelter_x_end = np.argmin(np.abs(xbins - tracking_data['shelter_loc'][1][1]))
    shelter_y_start = np.argmin(np.abs(ybins - tracking_data['shelter_loc'][0][0]))
    shelter_y_end = np.argmin(np.abs(ybins - tracking_data['shelter_loc'][1][0]))
    features['shelter'] = np.logical_and(np.logical_and(x>=shelter_x_start,x<=shelter_x_end),np.logical_and(y>=shelter_y_start,y<=shelter_y_end))
    heatmap[x, y] = np.ones(shape=(len(features['shelter'])))
    heatmap[x[features['shelter']], y[features['shelter']]] = np.ones(shape=(np.sum(features['shelter'])))*2

    # barrier random points
    feature_names = ['preflip_barrier','postflip_barrier']
    for i in [0,1]:
        barrier_x = np.sort(np.argsort(np.abs(ybins - tracking_data['barrier_loc'][i][1]))[:2])
        barrier_y = np.sort(np.argsort(np.abs(xbins - tracking_data['barrier_loc'][i][0]))[:2])
        features[feature_names[i]] = np.logical_and(np.logical_and(x>=barrier_x[0],x<=barrier_x[1]),np.logical_and(y>=barrier_y[0],y<=barrier_y[1]))
        heatmap[x[features[feature_names[i]]], y[features[feature_names[i]]]] = np.ones(shape=(np.sum(features[feature_names[i]])))*(3+i)

    feature_names = ['left_barrier','right_barrier']
    b = [b[0] for b in tracking_data['barrier_loc']]
    b = np.sort(b[:2])
    for i in [0,1]:
        barrier_y = np.sort(np.argsort(np.abs(xbins - b[i]))[:2])
        barrier_x = np.sort(np.argsort(np.abs(ybins - tracking_data['barrier_loc'][i][1]))[:2])
        features[feature_names[i]] = np.logical_and(np.logical_and(x>=barrier_x[0],x<=barrier_x[1]),np.logical_and(y>=barrier_y[0],y<=barrier_y[1]))
        heatmap[x[features[feature_names[i]]], y[features[feature_names[i]]]] = np.ones(shape=(np.sum(features[feature_names[i]])))*(3+i)

    feature_names = ['arena_bottom','arena_top']
    for i in [0,1]:
        barrier_x = np.sort(np.argsort(np.abs(ybins - tracking_data['barrier_loc'][i][0]))[:2])
        barrier_y = np.sort(np.argsort(np.abs(xbins - tracking_data['barrier_loc'][i][1]))[:2])
        features[feature_names[i]] = np.logical_and(np.logical_and(x>=barrier_x[0],x<=barrier_x[1]),np.logical_and(y>=barrier_y[0],y<=barrier_y[1]))
        heatmap[x[features[feature_names[i]]], y[features[feature_names[i]]]] = np.ones(shape=(np.sum(features[feature_names[i]])))*(3+i)


    return features