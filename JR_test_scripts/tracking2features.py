import numpy as np
import os
import dill as pickle

from behave_analysis.process.session import get_experiment 

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

def load_prediction_accuracy(session,settings, cond, time_cond):

    coef_path = os.path.join(session.base_path,
                            session.processed_path,
                            'models',
                            'LDA',
                            settings,
                            r'good',
                            time_cond,
                            'all',
                            cond,
                            str('good_' + cond + '_LDA_pa.pkl'))
    if not os.path.exists(coef_path):
        print("path still doesn't exist: " + coef_path)
    with open(coef_path, "rb") as dill_file:
        coef = pickle.load(dill_file)
    
    return coef

def extract_pa_across_sesh(all_angles,experiments_objects, conditions, all_features, settings, bar_angles = True, time_cond = ['experimental_conditions']):

    sessy = {}
    all_names = []
    pa = {}
    avg_pa = {}
    for a in all_angles:
        pa[a] = np.empty((len(experiments_objects),len(conditions),len(time_cond)))
    for a in all_features:
        avg_pa[a] = np.empty((len(experiments_objects), len(conditions),len(time_cond)))

    if isinstance(time_cond,str):
        time_cond = [time_cond]

    for s,sesh in enumerate(experiments_objects):
        # find session
        session = get_experiment(sesh)
        
        # load tracking
        track_path = os.path.join(session.base_path,session.processed_path,r"fully_processed_tracking_data.pickle")
        with open(track_path, "rb") as dill_file:
            tracking_data = pickle.load(dill_file)
        features = tracking_to_features(tracking_data)

        # load prediction accuracy
        coef = {}
        for h_idx,h in enumerate(time_cond): # this has to be a list!
            for idx,c in enumerate(conditions):
                
                point_grid = []

                coef[c] = load_prediction_accuracy(session, settings, c, h)
                
                for a in coef[c].keys():
                    if a in all_angles:
                        pa[a][s,idx,h_idx] = coef[c][a]
                    if 'rand' in a:
                        point_grid.append(coef[c][a])
                        
                if bar_angles:
                    b = [b[0] for b in tracking_data['barrier_loc']]
                    if b[0] > b[1]: # preflip barrier on the right
                        pa['h_rightbar_a'][s,idx,h_idx] = coef[c]['h_preflipbar_a']
                        pa['h_leftbar_a'][s,idx,h_idx] = coef[c]['h_postflipbar_a']
                    else:
                        pa['h_rightbar_a'][s,idx,h_idx] = coef[c]['h_postflipbar_a']
                        pa['h_leftbar_a'][s,idx,h_idx] = coef[c]['h_preflipbar_a']
                
                # avrage random point pred.acc. for shelter and barrier
                point_grid = np.array(point_grid)
                for a in features.keys():
                    avg_pa[a][s,idx,h_idx] = np.mean(point_grid[features[a]])                        

        name = sesh.nick_name + '_' + sesh.experiment_date
        all_names.append(name)
        sessy[name] = coef

    if len(time_cond) == 1:
        for a in pa.keys():
            pa[a] = pa[a][:,:,0]
        for a in features.keys():
            avg_pa[a] = avg_pa[a][:,:,0]

    return sessy, all_names, avg_pa, pa