# OS libaries
from loguru import logger
import numpy as np
import scipy.signal as sp
import os
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import pickle
import re
import matplotlib
matplotlib.use('Agg')
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# import functions
from behave_analysis.analyze.LDA.LDAlinearshift import LinearShift
from behave_analysis.utils.open_tracking_data import open_tracking_data
from behave_analysis.analyze.filtering_data.filtering_functions  import filter_video_dataframe, generate_bin_angles

def run_LDA_model(self, settings, angles):
    """ A function that runs discriminant analysis based on user settings"""

    prediction_accuracy = {}
    LS_compiled = {}
    title = []
    self.savepath = BuildSavingFolder(self.dir, settings, self.cluster_type, self.condition)
    
    # run LDA on different angles
    for variable in angles:
        if variable != 'randP':
            logger.info(f"Processing for LDA on {variable}")
            binned_angles,frames, savename= BinDfbyAngle(self,variable, settings)
            X = ProcessPredictors(self,frames,settings)
            logger.info(f"Running LDA on {variable}")
            pa = linear_discriminant_analysis(X,
                                              Y = binned_angles.T, 
                                              discriminant_type = settings.discriminant_type, 
                                              plotting = True, 
                                              settings = settings, 
                                              self = self, 
                                              title = savename)
            prediction_accuracy.update({variable: pa})
            logger.info(f"Running linear shift on LDA on {variable}")
            if settings.linear_shift:
                LS_output = LinearShift(X, 
                                        y = binned_angles.T,
                                        stat_computation_func = linear_discriminant_analysis,
                                        size_of_central_chunk = np.round(np.shape(X)[0]/3))
                LS_compiled.update({variable: LS_output})
                del LS_output
            title = np.append(title,variable)
        else:
            for j in np.arange(self.video_df.select(pl.col('^head_randP_.*$')).width):
                logger.info(f"Processing for LDA on {variable + str(j)} of {self.video_df.select(pl.col('^head_randP_.*$')).width}")
                binned_angles,frames, savename = BinDfbyAngle(self,str('head_randP_' + str(j)), settings)
                X = ProcessPredictors(self,frames,settings)
                logger.info(f"Running LDA on {variable + str(j)} of {self.video_df.select(pl.col('^head_randP_.*$')).width}")
                pa = linear_discriminant_analysis(X,
                                                  Y = binned_angles.T, 
                                                  discriminant_type = settings.discriminant_type, 
                                                  plotting = False,
                                                  settings = settings,
                                                  self = self,
                                                  title = savename)
                prediction_accuracy.update({str(variable + str(j)): pa})
                logger.info(f"Running linear shift on LDA on {variable + str(j)}")
                if settings.linear_shift:
                    LS_output = LinearShift(X, 
                                            y = binned_angles.T,
                                            stat_computation_func = linear_discriminant_analysis,
                                            size_of_central_chunk = np.round(np.shape(X)[0]/3))
                    LS_compiled.update({str(variable + str(j)): LS_output})
                    del LS_output
                title = np.append(title,str('randP' + str(j)))
    
    # make a plot of prediction accuracy across variables
    PlotPredictionAccuracy(self, prediction_accuracy,title,settings)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + '_' + str(self.condition) + "_LDA_prediction_accuracy" + ".pkl"
    with open(filename, 'wb') as fp:
        pickle.dump(prediction_accuracy, fp) 

    # make a plot of prediction accuracy across variables with linear shift stats
    if settings.linear_shift:
        PlotLSPredictionAccuracy(self,LS_compiled,title,settings)
        filename = str(self.savepath) + "/" + str(self.cluster_type) + '_' + str(self.condition) + "_LDA_LS_prediction_accuracy" + ".pkl"
        with open(filename, 'wb') as fp:
            pickle.dump(LS_compiled, fp)  

    # map random points on arena:
    if len(list(filter(lambda x: 'randP' in x, title))) > 10:
        PredictionAccuracyMapped(self,prediction_accuracy)

#### --------- MAIN LDA FUNCS

def BinDfbyAngle(self, variable, settings):
    """
    A function that processes dataframe for discriminant analysis
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """
    # edges for binning firing rate at different angles
    bin_angles, _ = generate_bin_angles(settings.number_of_bins)

    # subselect relevant times
    filtered_video_df = filter_video_dataframe(self.video_df, self.condition)
    title = str(variable + '_' + self.condition)
    filtered_video_df = filtered_video_df.select(['frames',variable])
    frames = filtered_video_df['frames'].unique().to_numpy() - 1

    # bin angles
    binned_angles = np.array(filtered_video_df[variable].to_numpy())

    # median filter!
    binned_angles = np.arctan2(sp.medfilt(np.sin(binned_angles),41),sp.medfilt(np.cos(binned_angles),41))
    binned_angles = np.digitize(binned_angles, bin_angles)

    return binned_angles, frames, title

def binDfbyEpoch(matrix, matriy, bins, epoch_num):

    # make angle bins equally populated
    matrix, matriy = EqualAngleBins_matrix(matrix, matriy) # this step randomly subsamples!!

    # chunk data into training and test data for each angle bin!!
    epochs = np.empty_like(matriy)
    for i in bins:
        x_filt = matrix[matriy == i,:]
        binned_frames = data_chunker(x_filt,epoch_num)
        epochs[matriy == i] = binned_frames
    
    epochs = epochs[np.argsort(matrix[:,0])]
    
    return matrix, matriy, epochs

def ProcessPredictors(self,frames, settings):
    
    # select frames that have been filtered
    X = self.postprocessObject.frame_by_cluster_matrix
    X = X[frames,:]

    # remove NaN columns (empty clusters)
    nancolumns = np.where(np.sum(X == 0,axis = 0) == np.shape(X)[0])[0]
    if len(nancolumns) > 0:
        X = np.delete(X, nancolumns, axis=1)

    # normalize firing rates
    # X = X/np.amax(X,axis=0)

    # z-score firing rates
    X = (X - np.mean(X,axis=0))/np.std(X,axis=0)

    # optional: run PCA
    if settings.PCA_process:
        pca = PCA(n_components = 15)
        X = pca.fit_transform(X)

    # first column of X is frame num
    X = np.c_[frames,X]

    return X

def linear_discriminant_analysis(X,Y, discriminant_type = 'linear', plotting = False, settings = None, self = None, title = None):#self, df, title, settings, X):
    """
    A function for doing LDA on data
    """

    # initialize variables
    n_bins = len(np.unique(Y))
    epoch_num = 6
    conf_matrix_all_train = np.empty((n_bins,n_bins,epoch_num))
    conf_matrix_all_test = np.empty((n_bins,n_bins,epoch_num))

    # chunk into epochs
    X, Y, epochs = binDfbyEpoch(X, Y, np.unique(Y), epoch_num)
    X = X[:,1:] # the first column is frame id and you no longer need it

    # LDA
    for counter,i in enumerate(np.unique(epochs).astype(int)):
        test_idx = epochs == (i)
        train_idx = epochs != (i)

        # figure set up
        if plotting:
            plt.figure(figsize=(20, 16))
            plt.subplots_adjust(hspace=0.3)
        
        # make train matrix of frames x clusters
        X1 = X[train_idx,:]

        # make test matrix of frames x clusters
        X2 = X[test_idx,:]

        # train model
        y = Y[train_idx]
        if discriminant_type == 'linear':
            clf = LinearDiscriminantAnalysis()
        elif discriminant_type == 'quadratic':
            clf = QuadraticDiscriminantAnalysis()
        clf.fit(X1, y)

        # plot confusion matrix of prediction on training data
        conf_matrix_all_train[:,:,counter] = plotConfusionMatrix(y,clf.predict(X1),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

        if plotting:
            # plot histogram of frames per angle bin
            ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
            ax.hist(clf.predict(X1), np.arange(1,n_bins+2))
            ax.hist(y, np.arange(1,n_bins+2))
            ax.set_title('training data')

            # look at data side-by-side
            ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
            ax.plot(clf.predict(X1))
            ax.plot(y)
            ax.legend(["prediction","real"])
            ax.set_title("training data")
            ax.set_ylabel('binned angles')
            ax.set_xlabel('time')

        # plot confusion matrix of prediction on test data
        y = Y[test_idx]
        conf_matrix_all_test[:,:,counter] = plotConfusionMatrix(y,clf.predict(X2),'test data',plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

        if plotting:
            # plot histogram of frames per angle bin
            ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
            ax.hist(clf.predict(X2), np.arange(1,n_bins+2))
            ax.hist(y, np.arange(1,n_bins+2))
            ax.set_title('test data')

            # look at data side-by-side
            ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
            ax.plot(clf.predict(X2))
            ax.plot(y)
            ax.legend(["prediction","real"])
            ax.set_title("test data")
            ax.set_ylabel('binned angles')
            ax.set_xlabel('time')

            filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i) + ".png"
            plt.savefig(filename)
            if self.show_plots: plt.show()
            plt.close()
    
    if plotting:
        # plot average confusion matrix
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        ax = plt.subplot(1,2,1)
        ax.imshow(np.mean(conf_matrix_all_train, axis=2), cmap = "Blues", vmin = 0, vmax = 1)
        ax.set_ylabel('real')
        ax.set_xlabel('predicted')
        ax.set_title('train')

        ax = plt.subplot(1,2,2)
        ax.imshow(np.mean(conf_matrix_all_test, axis=2), cmap = "Blues", vmin = 0, vmax = 1)
        ax.set_ylabel('real')
        ax.set_xlabel('predicted')
        ax.set_title('test')

        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + ".png"
        plt.savefig(filename)
        if self.show_plots: plt.show()
        plt.close()

    prediction_accuracy = compute_prediction_accuracy(np.mean(conf_matrix_all_test, axis=2))

    return prediction_accuracy

def PlotPredictionAccuracy(self, prediction_accuracy, title,settings):
    fig = go.Figure()
    
    if len(list(filter(lambda x: 'randP' in x, title))) < 10:
        colorz = sample_colorscale('Rainbow', list(np.linspace(0,1,len(title))))
    else:
        colorz = sample_colorscale('Rainbow', list(np.linspace(0,1,len(list(filter(lambda x: 'randP' not in x, title)))+1)))
        
    for i, var in enumerate(list(filter(lambda x: 'randP' not in x, title))):
        fig.add_trace(go.Bar(x = [var],
                            y = [prediction_accuracy[var]],
                            width = .5,
                            marker = dict(color = colorz[i], opacity = .5)))
    
    if len(list(filter(lambda x: 'randP' in x, title))) < 10:
        for j, var in enumerate(list(filter(lambda x: 'randP' in x, title))):
            fig.add_trace(go.Bar(x = [var],
                                y = [prediction_accuracy[var]],
                                width = .5,
                                marker = dict(color = colorz[i+j], opacity = .5)))
    else:
        res = [val for key, val in prediction_accuracy.items() if re.search('randP', key)]
        var = 'randP'
        fig.add_trace(go.Violin(x = [var]*len(res), 
                                y = res,
                                points = 'all',jitter = .05,
                                marker = dict(size = 3,color = colorz[i+1])))

    fig.update_layout(showlegend=False)
    fig.update_yaxes(range = [0, 1])
    fig.update_yaxes(title_text = 'prediction accuracy')
    fig.update_xaxes(tickangle = -45)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + '_' + str(self.condition) + "_LDA_prediction_accuracy" + ".png"
    fig.write_image(filename)

def PlotLSPredictionAccuracy(self, LS_compiled, title, settings):
    fig = go.Figure()
    if len(title) > 10:
        colorz = sample_colorscale('Rainbow', list(np.linspace(0,1,10)))
    else:
        colorz = sample_colorscale('Rainbow', list(np.linspace(0,1,len(title))))

    for i, var in enumerate(title):
        if i >= 10:
            break
        fig.add_trace(go.Violin(x = [var]*len(LS_compiled[var].pseudo_stats), 
                                y = LS_compiled[var].pseudo_stats,
                                points = 'all',jitter = .05,
                                marker = dict(size = 3,color = colorz[i])))
        fig.add_trace(go.Scatter(x = [var],
                                 y = [LS_compiled[var].real_stat],
                                 mode="markers",
                                 marker_color='rgb(255, 0, 0)',
                                 marker = dict(size = 5, symbol = 'diamond')))
        if LS_compiled[var].reject_null:
            fig.add_trace(go.Scatter(x = [var],
                                     y = [1],
                                     mode="markers",
                                     marker_color='rgb(0, 0, 0)',
                                     marker = dict(size = 5, symbol = 'star')))

    fig.update_layout(showlegend=False)
    fig.update_yaxes(range = [0, 1.1])
    fig.update_yaxes(title_text = 'prediction accuracy')
    fig.update_xaxes(tickangle = -45)
    filename = str(self.savepath) + "/" + str(self.cluster_type) + '_' + str(self.condition) + "_LDA_LS_prediction_accuracy" + ".png"
    fig.write_image(filename)

def PredictionAccuracyMapped(self,prediction_accuracy):
    open_tracking_data(self)
    pa = [val for key, val in prediction_accuracy.items() if re.search('randP', key)]
    plt.figure(figsize=(15, 15))
    if 'h_bar_north_a' in prediction_accuracy.keys():
        plt.plot([self.tracking_data["barrier_loc"][0][0],self.tracking_data["barrier_loc"][1][0]],
                [self.tracking_data["barrier_loc"][0][1],self.tracking_data["barrier_loc"][1][1]],
                color = [1,0,0])
    if 'hsa' in prediction_accuracy.keys():
        for i in [0,1]:
            plt.plot([self.tracking_data["shelter_loc"][0][0],self.tracking_data["shelter_loc"][1][0]],
                    [self.tracking_data["shelter_loc"][i][1],self.tracking_data["shelter_loc"][i][1]],
                    color = [1,0,0])
            plt.plot([self.tracking_data["shelter_loc"][i][0],self.tracking_data["shelter_loc"][i][0]],
                    [self.tracking_data["shelter_loc"][0][1],self.tracking_data["shelter_loc"][1][1]],
                    color = [1,0,0])
    sc = plt.scatter(self.tracking_data["randP_loc"][:,0],self.tracking_data["randP_loc"][:,1], c = pa, s  =75, cmap = "Blues")
    plt.colorbar(sc)
    plt.axis('off')
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_aspect('equal')
    filename = str(self.savepath) + "/" + str(self.cluster_type) + '_' + str(self.condition) + "_LDA_prediction_accuracy_map" + ".png"
    plt.savefig(filename)
    if self.show_plots: plt.show()
    plt.close()

# Utility functions ------------------------------------------------------------------------------------------------

def BuildSavingFolder(basepath, settings, cluster_type, condition):

    if settings.discriminant_type == 'linear':
        pathh = str(basepath) + "/" + "LDA"
    elif settings.discriminant_type == 'quadratic':
        pathh = str(basepath) + "/" + "QDA"
    if len(settings.PCA_process) > 0:
        pathh = str(pathh) + "_PCA"
    if settings.use_firing_rate:
        pathh = str(pathh) + "_fr"

    pathh = str(pathh) + "/" + str(cluster_type) + "/" + str(condition)

    if not(os.path.exists(pathh)): 
        os.makedirs(pathh) 
    
    return pathh

def plotConfusionMatrix(y,x,title,axy):
    conf = confusion_matrix(y, x)
    conf = conf.astype('float64')
    conf = conf/np.sum(conf,axis=1)

    axy.imshow(conf, cmap = "Blues", vmin = 0, vmax = 1)
    axy.set_ylabel('real')
    axy.set_xlabel('predicted')
    axy.set_title(title)
    return conf

def EqualAngleBins_matrix(x,y):
    angbins, counts = np.unique(y, return_counts = True)
    samples = np.amin(counts)

    y_new = []
    for c,i in enumerate(angbins):
        x_filt = x[y == i,:]
        samplingidx = np.random.randint(0,len(x_filt),samples)
        x_filt = x_filt[samplingidx,:]
        if c == 0: x_new = x_filt
        else: x_new = np.append(x_new,x_filt, axis=0)
        y_new = np.append(y_new,np.ones(np.shape(x_filt)[0])*i)
    
    y_new = y_new[np.argsort(x_new[:,0])]
    x_new = x_new[np.argsort(x_new[:,0]),:]
    return x_new, y_new

def data_chunker(x,epoch_num):
    rows = np.arange(np.shape(x)[0])
    epoch_edge = np.round(np.linspace(np.amin(rows)-1,np.amax(rows)+1,epoch_num+1))
    binned_frames = np.digitize(rows,epoch_edge)
    return binned_frames

def compute_prediction_accuracy(matrixx):
    pos = np.floor(np.shape(matrixx)[1]/2).astype(int)
    pred_acc = np.zeros(np.shape(matrixx)[0])
    for i in np.arange(np.shape(matrixx)[0]):
        x = np.roll(matrixx[i.astype(int),:],pos-i)
        pred_acc[i] = np.sum(x[pos-1:pos+2])
    return np.mean(pred_acc)
