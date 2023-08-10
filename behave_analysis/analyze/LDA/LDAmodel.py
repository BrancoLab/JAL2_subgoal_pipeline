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
import matplotlib
import re
matplotlib.use('Agg')
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# import functions
from behave_analysis.visualize.visualize_efizz import filter_video_dataframe, generate_bin_angles 
from behave_analysis.analyze.LDA.LDAlinearshift import LinearShift
from behave_analysis.utils.open_tracking_data import open_tracking_data

def run_LDA_model(self, settings):
    """ A function that runs discriminant analysis based on user settings"""

    prediction_accuracy = {}
    LS_compiled = {}
    title = []
    self.savepath = BuildSavingFolder(self.dir, settings, self.cluster_type)
    
    # run LDA on different angles
    for variable in settings.run_LDA:
        if variable != 'randP':
            logger.info(f"Processing for LDA on {variable}")
            df, savename, clu = BinDfbyAngle(self,variable, settings)
            X = ProcessPredictors(df, clu, settings)
            logger.info(f"Running LDA on {variable}")
            pa = linear_discriminant_analysis(X,
                                              Y = (df['binned_angles'].to_numpy().T), 
                                              discriminant_type = settings.discriminant_type, 
                                              plotting = True, 
                                              settings = settings, 
                                              self = self, 
                                              title = savename)
            prediction_accuracy.update({variable: pa})
            logger.info(f"Running linear shift on LDA on {variable}")
            if settings.linear_shift:
                LS_output = LinearShift(X, 
                                        y = (df['binned_angles'].to_numpy().T),
                                        stat_computation_func = linear_discriminant_analysis,
                                        size_of_central_chunk = np.round(np.shape(X)[0]/3))
                LS_compiled.update({variable: LS_output})
                del LS_output
            title = np.append(title,variable)
        else:
            for j in np.arange(self.data_df.select(pl.col('^head_randP_.*$')).width):
                logger.info(f"Processing for LDA on {variable + str(j)} of {self.data_df.select(pl.col('^head_randP_.*$')).width}")
                df, savename, clu = BinDfbyAngle(self,str('head_randP_' + str(j)), settings)
                X = ProcessPredictors(df, clu, settings)
                logger.info(f"Running LDA on {variable + str(j)} of {self.data_df.select(pl.col('^head_randP_.*$')).width}")
                pa = linear_discriminant_analysis(X,
                                                  Y = (df['binned_angles'].to_numpy().T), 
                                                  discriminant_type = settings.discriminant_type, 
                                                  plotting = False,
                                                  settings = settings,
                                                  self = self,
                                                  title = savename)
                prediction_accuracy.update({str(variable + str(j)): pa})
                logger.info(f"Running linear shift on LDA on {variable + str(j)}")
                if settings.linear_shift:
                    LS_output = LinearShift(X, 
                                            y = (df['binned_angles'].to_numpy().T),
                                            stat_computation_func = linear_discriminant_analysis,
                                            size_of_central_chunk = np.round(np.shape(X)[0]/3))
                    LS_compiled.update({str(variable + str(j)): LS_output})
                    del LS_output
                title = np.append(title,str('randP' + str(j)))
    
    # make a plot of prediction accuracy across variables
    # TODO: if randP > 3.... plot distribution instead!
    PlotPredictionAccuracy(self, prediction_accuracy,title,settings)
    if self.object_present == True:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + ".pkl"
    else:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + "_noObj.pkl"
    with open(filename, 'wb') as fp:
        pickle.dump(prediction_accuracy, fp) 

    # make a plot of prediction accuracy across variables with linear shift stats
    if settings.linear_shift:
        PlotLSPredictionAccuracy(self,LS_compiled,title,settings)
        if self.object_present == True:
            filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_LS_prediction_accuracy" + ".pkl"
        else:
            filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_LS_prediction_accuracy" + "_noObj.pkl"
        with open(filename, 'wb') as fp:
            pickle.dump(LS_compiled, fp)  

    # TODO: random points analysis:
    if len(list(filter(lambda x: 'randP' in x, title))) > 10:
        PredictionAccuracyMapped(self,prediction_accuracy)
    
def BinDfbyAngle(self, variable, settings):
    """
    A function that processes dataframe for discriminant analysis
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """
    # edges for binning firing rate at different angles
    bin_angles, bin_angle_center = generate_bin_angles(settings.number_of_bins)


    # subselect relevant times
    filtered_video_df, angle_filt, title = filter_video_dataframe(self.data_df, variable, self.object_present)

    # bin angles
    filtered_video_df = filtered_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
    filtered_video_df = filtered_video_df.with_columns(filtered_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
    filtered_video_df = filtered_video_df.fill_null(strategy="zero")
    filtered_video_df = filtered_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 
    clu = filtered_video_df["spike_clusters"].unique().to_numpy()
    # if clu[0] == 0: clu = clu[1:]

    # group the  data
    df_first = filtered_video_df.groupby(["frames"]).first()
    df_first = df_first.sort('frames')
    df_all = filtered_video_df.groupby(["frames"]).all()
    df_all = df_all.sort('frames')
    df_all.replace("binned_angles",df_first['binned_angles'])
    df_all = df_all.sort('frames')


    # median filter!
    x = sp.medfilt(np.cos(df_all['binned_angles'].to_numpy()),41)
    y = sp.medfilt(np.sin(df_all['binned_angles'].to_numpy()),41)
    df_all.replace('binned_angles',pl.Series('binned_angles',np.digitize(np.arctan2(y,x),bin_angles[1:-1])))

    return df_all, title, clu

def binDfbyEpoch(matrix, matriy, n_bins, epoch_num):

    # make angle bins equally populated
    matrix, matriy = EqualAngleBins_matrix(matrix, matriy) # this step randomly subsamples!!

    # chunk data into training and test data for each angle bin!!
    epochs = np.empty_like(matriy)
    for i in np.arange(n_bins):
        x_filt = matrix[matriy == i,:]
        binned_frames = data_chunker(x_filt,epoch_num)
        epochs[matriy == i] = binned_frames
    
    epochs = epochs[np.argsort(matrix[:,0])]
    
    return matrix, matriy, epochs

def ProcessPredictors(df, clu, settings):
    # initialize predictor matrix
    X = np.zeros((int(df['frames'].max()),len(clu)))

    # frames by firing per cluster matrix
    fillMatrix(df,X,clu)
    if clu[0] == 0: X = X[:,1:]
    X = X[df['frames'].unique().to_numpy().astype(int)-1,:]

    # remove NaN columns (empty clusters)
    nancolumns = np.where(np.sum(X == 0,axis = 0) == np.shape(X)[0])[0]
    if len(nancolumns) > 0:
        X = np.delete(X, nancolumns, axis=1)

    # optional: transform to firing rate estimate in Hz
    if settings.use_firing_rate:
        for i in np.arange(np.shape(X)[1]):
            X[:,i] = FiringRateEstimate(X[:,i],sampling_rate = 40, window_size = 100)

    # normalize firing rates
    # X = X/np.amax(X,axis=0)
    # z-score firing rates
    X = (X - np.mean(X,axis=0))/np.std(X,axis=0)

    # optional: run PCA
    if settings.PCA_process:
        pca = PCA(n_components = 15)
        X = pca.fit_transform(X)

    # first column of X is frame num
    X = np.c_[df['frames'].unique().to_numpy(),X]

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
    X, Y, epochs = binDfbyEpoch(X, Y, n_bins, epoch_num)

    # LDA
    for i in np.arange(epoch_num):
        test_idx = epochs == (i+1)
        train_idx = epochs != (i+1)

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
        conf_matrix_all_train[:,:,i] = plotConfusionMatrix(y,clf.predict(X1),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

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
        conf_matrix_all_test[:,:,i] = plotConfusionMatrix(y,clf.predict(X2),'test data',plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

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

            if self.object_present == True:
                filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + ".png"
            else:
                filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + "_noObj.png"
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

        if self.object_present == True:
            filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + ".png"
        else:
            filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + "_noObj.png"
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
    if self.object_present == True:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + ".png"
    else:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + "_noObj.png"
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
    if self.object_present == True:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_LS_prediction_accuracy" + ".png"
    else:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_LS_prediction_accuracy" + "_noObj.png"
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
    if self.object_present == True:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy_map" + ".png"
    else:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy_map" + "_noObj.png"
    plt.savefig(filename)
    if self.show_plots: plt.show()
    plt.close()

# Utility functions ------------------------------------------------------------------------------------------------

def BuildSavingFolder(basepath, settings, cluster_type):

    if settings.discriminant_type == 'linear':
        pathh = str(basepath) + "/" + "LDA"
    elif settings.discriminant_type == 'quadratic':
        pathh = str(basepath) + "/" + "QDA"
    if len(settings.PCA_process) > 0:
        pathh = str(pathh) + "_PCA"
    if settings.use_firing_rate:
        pathh = str(pathh) + "_fr"

    pathh = str(pathh) + "/" + str(cluster_type)

    if not(os.path.exists(pathh)): 
        os.makedirs(pathh) 
    
    return pathh

def fillMatrix(df,matrix,clu_id):
    for i2 in df["frames"].unique():
        d = df.filter(df["frames"] == i2).to_dict(as_series=False)
        spikes = np.array(d.get('spike_count')[0])
        clusters = np.array(d.get('spike_clusters')[0])
        spikes = spikes[np.argsort(clusters)]
        clusters = np.sort(clusters)
        matrix[int(i2)-1,np.where(np.in1d(clu_id,clusters))[0]] = spikes

def FiringRateEstimate(x,sampling_rate,window_size):
    # sampling rate in fps
    # window size in ms
    nbins = 1000/window_size
    x2 = np.convolve(x,np.ones(int(sampling_rate/nbins),dtype = int),'same')*nbins
    return x2

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

def df_chunker(df,epoch_num):
    epoch_df = df.sort("frames")
    epoch_df = epoch_df.hstack([pl.Series('rows',np.arange(len(epoch_df)))])
    epoch_edge = np.round(np.linspace(epoch_df["rows"].unique().min()-1,epoch_df["rows"].unique().max(),epoch_num+1))
    epoch_df = epoch_df.with_columns(epoch_df["rows"].cut(bins = epoch_edge, labels = [str(x) for x in np.arange(epoch_num+2)])['category'].alias('binned_frames'))
    epoch_df = epoch_df.fill_null(strategy="zero")
    epoch_df = epoch_df.select([pl.col('binned_frames').apply(float),pl.exclude('binned_frames')]) 
    return epoch_df

def compute_prediction_accuracy(matrixx):
    pos = np.floor(np.shape(matrixx)[1]/2).astype(int)
    pred_acc = np.zeros(np.shape(matrixx)[0])
    for i in np.arange(np.shape(matrixx)[0]):
        x = np.roll(matrixx[i.astype(int),:],pos-i)
        pred_acc[i] = np.sum(x[pos-1:pos+2])
    return np.mean(pred_acc)


##----------OLD FUNCS   
def EqualAngleBins_df(df):
    df_samples = df.groupby(['binned_angles']).count().min()
    samples = df_samples['count'].to_numpy()[0]

    for c, i in enumerate(df['binned_angles'].unique()):
        d_filt = df.filter(df['binned_angles'] == i)
        d_filt = d_filt.sample(samples)
        if c == 0: df_new = d_filt
        if c > 0: df_new = df_new.vstack(d_filt)
    
    df_new = df_new.sort('frames')
    return df_new

def fillMatrix_small(df,matrix,clu_id):
    for i, i2 in enumerate(df["frames"].unique()):
        d = df.filter(df["frames"] == i2).to_dict(as_series=False)
        spikes = np.array(d.get('spike_count')[0])
        clusters = np.array(d.get('spike_clusters')[0])
        spikes = spikes[np.argsort(clusters)]
        clusters = np.sort(clusters)
        matrix[i,np.where(np.in1d(clu_id,clusters))[0]] = spikes