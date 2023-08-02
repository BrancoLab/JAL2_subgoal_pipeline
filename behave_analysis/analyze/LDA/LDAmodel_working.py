# OS libaries
from loguru import logger
import numpy as np
import scipy.signal as sp
import os
import polars as pl
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# import functions
from behave_analysis.visualize.visualize_efizz import filter_video_dataframe, generate_bin_angles 

def run_LDA_model(self, settings):
    """ A function that runs discriminant analysis based on user settings"""

    prediction_accuracy = []
    title = []
    self.savepath = BuildSavingFolder(self.dir, settings)

    # run LDA on different angles
    for variable in settings.run_LDA:
        if variable != 'randP':
            logger.info(f"Running LDA on {variable}")
            df, savename, clu = BinDfbyAngleAndEpoch(self,variable, settings)
            pa = linear_discriminant_analysis(self, df, savename, settings, clu)
            prediction_accuracy = np.append(prediction_accuracy,pa)
            title = np.append(title,variable)
        else:
            for j in np.arange(self.data_df.select(pl.col('^head_randP_.*$')).width):
                logger.info(f"Running LDA on {variable + str(j)} of {self.data_df.select(pl.col('^head_randP_.*$')).width}")
                df, savename, clu = BinDfbyAngleAndEpoch(self,str('head_randP_' + str(j)), settings)
                pa = linear_discriminant_analysis(self, df, savename, settings, clu)
                prediction_accuracy = np.append(prediction_accuracy,pa)
                title = np.append(title,str('head_randP_' + str(j)))
    
    # make a plot of prediction accuracy across variables
    PlotPredictionAccuracy(self, prediction_accuracy,title,settings)

def BinDfbyAngleAndEpoch(self, variable, settings):
    """
    A function that processes dataframe for discriminant analysis
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """

    # edges for binning firing rate at different angles
    bin_angles, bin_angle_center = generate_bin_angles(settings.number_of_bins)

    # subselect relevant times
    filtered_video_df, angle_filt, title = filter_video_dataframe(self.data_df, variable, settings.object_present)

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

    # make angle bins equally populated
    equalized_df = EqualAngleBins_df(df_all) # this step randomly subsamples!!

    # chunk data into training and test data for each angle bin!!
    for i in np.arange(settings.number_of_bins-1):
        bin_df = equalized_df.filter((equalized_df['binned_angles'] == (i)))
        epoch_df = data_chunker(bin_df,settings.epoch_num)
        if i == 0: all_epoch_df = epoch_df
        if i > 0: all_epoch_df = all_epoch_df.vstack(epoch_df)
    
    all_epoch_df = all_epoch_df.sort('frames')
    all_epoch_df = all_epoch_df.select(pl.exclude("rows"))
    
    output_df = all_epoch_df.select(['binned_frames','frames','binned_angles','spike_clusters','spike_count'])

    return output_df, title, clu

def linear_discriminant_analysis(self, df, title, settings, clu):
    """
    A function for doing LDA on data
    """
    
    # initialize variables
    conf_matrix_all_train = np.empty((settings.number_of_bins-1,settings.number_of_bins-1,settings.epoch_num))
    conf_matrix_all_test = np.empty((settings.number_of_bins-1,settings.number_of_bins-1,settings.epoch_num))
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

    # LDA
    for i in np.arange(settings.epoch_num):
        test_all = df.filter((df['binned_frames'] == (i+1)))
        train_all = df.filter((df['binned_frames'] != (i+1)))

        # figure set up
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        
        # make train matrix of frames x clusters
        X1 = X[(df['binned_frames'] != (i+1)).to_list(),:]

        # make test matrix of frames x clusters
        X2 = X[(df['binned_frames'] == (i+1)).to_list(),:]

        # train model
        y = train_all["binned_angles"].to_numpy()
        if settings.discriminant_type == 'linear':
            clf = LinearDiscriminantAnalysis()
        elif settings.discriminant_type == 'quadratic':
            clf = QuadraticDiscriminantAnalysis()
        clf.fit(X1, y)

        # plot confusion matrix of prediction on training data
        conf_matrix_all_train[:,:,i] = plotConfusionMatrix(y,clf.predict(X1),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

        # plot histogram of frames per angle bin
        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
        ax.hist(clf.predict(X1), np.arange(1,settings.number_of_bins+1))
        ax.hist(y, np.arange(1,settings.number_of_bins+1))
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
        y = test_all["binned_angles"].to_numpy()
        conf_matrix_all_test[:,:,i] = plotConfusionMatrix(y,clf.predict(X2),'test data',plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

        # plot histogram of frames per angle bin
        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
        ax.hist(clf.predict(X2), np.arange(1,settings.number_of_bins+1))
        ax.hist(y, np.arange(1,settings.number_of_bins+1))
        ax.set_title('test data')

        # look at data side-by-side
        ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
        ax.plot(clf.predict(X2))
        ax.plot(y)
        ax.legend(["prediction","real"])
        ax.set_title("test data")
        ax.set_ylabel('binned angles')
        ax.set_xlabel('time')

        if settings.object_present == True:
            filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + ".png"
        else:
            filename = self.savepath + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + "_noObj.png"
        plt.savefig(filename)
        if self.show_plots: plt.show()
        plt.close()
    
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

    if settings.object_present == True:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + ".png"
    else:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + "_noObj.png"
    plt.savefig(filename)
    if self.show_plots: plt.show()
    plt.close()

    prediction_accuracy = compute_prediction_accuracy(np.mean(conf_matrix_all_test, axis=2))

    return prediction_accuracy

def PlotPredictionAccuracy(self, prediction_accuracy, title,settings):
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=0.3)
    plt.bar(np.arange(len(prediction_accuracy)),prediction_accuracy,tick_label = title)
    plt.ylim(0,1)
    plt.xticks(rotation = 45)
    plt.ylabel('prediction accuracy')
    if settings.object_present == True:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + ".png"
    else:
        filename = str(self.savepath) + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + "_noObj.png"
    plt.savefig(filename)
    plt.close()

# Utility functions ------------------------------------------------------------------------------------------------

def BuildSavingFolder(basepath, settings):
    
    if settings.discriminant_type == 'linear':
        pathh = str(basepath) + "/" + "LDA"
    elif settings.discriminant_type == 'quadratic':
        pathh = str(basepath) + "/" + "QDA"
    if len(settings.PCA_process) > 0:
        pathh = str(pathh) + "_PCA"
    if settings.use_firing_rate:
        pathh = str(pathh) + "_fr"

    pathh = str(pathh) + "/" + str(settings.cluster_type)

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

def fillMatrix_small(df,matrix,clu_id):
    for i, i2 in enumerate(df["frames"].unique()):
        d = df.filter(df["frames"] == i2).to_dict(as_series=False)
        spikes = np.array(d.get('spike_count')[0])
        clusters = np.array(d.get('spike_clusters')[0])
        spikes = spikes[np.argsort(clusters)]
        clusters = np.sort(clusters)
        matrix[i,np.where(np.in1d(clu_id,clusters))[0]] = spikes

def plotConfusionMatrix(y,x,title,axy):
    conf = confusion_matrix(y, x)
    conf = conf.astype('float64')
    conf = conf/np.sum(conf,axis=1)

    axy.imshow(conf, cmap = "Blues", vmin = 0, vmax = 1)
    axy.set_ylabel('real')
    axy.set_xlabel('predicted')
    axy.set_title(title)
    return conf

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

def data_chunker(df,epoch_num):
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