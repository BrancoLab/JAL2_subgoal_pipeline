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
from sklearn.metrics import confusion_matrix

# import functions
from behave_analysis.visualize.visualize_efizz import filter_video_dataframe, generate_bin_angles 

def run_LDA_model(self, settings):

    prediction_accuracy = []
    title = []

    # run LDA on different angles
    for i,variable in enumerate(settings.run_LDA):
        if variable != 'randP':
            logger.info(f"Running LDA on {variable}")
            pa = linear_discriminant_analysis(self, variable,settings.object_present)
            prediction_accuracy = np.append(prediction_accuracy,pa)
            title = np.append(title,variable)
        else:
            for i in np.arange(self.data_df.select(pl.col('^head_randP_.*$')).width):
                logger.info(f"Running LDA on {variable + str(i)} of {self.data_df.select(pl.col('^head_randP_.*$')).width}")
                pa = linear_discriminant_analysis(self, str('head_randP_' + str(i)),settings.object_present)
                prediction_accuracy = np.append(prediction_accuracy,pa)
                title = np.append(title,str('head_randP_' + str(i)))
    
    # make a plot of prediction accuracy across variables
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(hspace=0.3)
    plt.bar(np.arange(len(prediction_accuracy)),prediction_accuracy,tick_label = title)
    plt.ylim(0,1)
    plt.xticks(rotation = 45)
    plt.ylabel('prediction accuracy')
    if settings.object_present == True:
        filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + ".png"
    else:
        filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_prediction_accuracy" + "_noObj.png"
    plt.savefig(filename)
    plt.close()

def linear_discriminant_analysis(self, variable, object_present = True):
    """
    A function for doing LDA on data
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """
    epoch_num = 6 # chunks of time for training and testing data
    number_of_bins = 19

    # edges for binning firing rate at different angles
    bin_angles, bin_angle_center = generate_bin_angles(number_of_bins)

    # subselect relevant times
    filtered_video_df, angle_filt, title = filter_video_dataframe(self.data_df, variable, object_present)

    # bin angles
    filtered_video_df = filtered_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
    filtered_video_df = filtered_video_df.with_columns(filtered_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in bin_angle_center])['category'].alias('binned_angles'))
    # filtered_video_df = filtered_video_df.with_columns(filtered_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in np.arange(len(bin_angle_center))])['category'].alias('binned_angles'))
    filtered_video_df = filtered_video_df.fill_null(strategy="zero")
    filtered_video_df = filtered_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 
    clu = filtered_video_df["spike_clusters"].unique().to_numpy()

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
    equalized_df = EqualAngleBins_df(df_all)

    # chunk data into training and test data for each angle bin!!
    for i in np.arange(number_of_bins-1):
        bin_df = equalized_df.filter((equalized_df['binned_angles'] == (i)))
        epoch_df = data_chunker(bin_df,epoch_num)
        if i == 0: all_epoch_df = epoch_df
        if i > 0: all_epoch_df = all_epoch_df.vstack(epoch_df)
    
    all_epoch_df = all_epoch_df.sort('frames')
    all_epoch_df = all_epoch_df.select(pl.exclude("rows"))

    # initialize variables
    conf_matrix_all_train = np.empty((number_of_bins-1,number_of_bins-1,epoch_num))
    conf_matrix_all_test = np.empty((number_of_bins-1,number_of_bins-1,epoch_num))

    # LDA
    for i in np.arange(epoch_num):
        test_all = all_epoch_df.filter((all_epoch_df['binned_frames'] == (i+1)))
        train_all = all_epoch_df.filter((all_epoch_df['binned_frames'] != (i+1)))

        # figure set up
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        
        # make train matrix of frames x clusters
        X = np.zeros((len(train_all["frames"].unique()),len(clu)))
        fillMatrix(train_all,X,clu)
        if clu[0] == 0: X = X[:,1:]

        # make test matrix of frames x clusters
        X2 = np.zeros((len(test_all["frames"].unique()),len(clu)))
        fillMatrix(test_all,X2,clu)
        if clu[0] == 0: X2 = X2[:,1:]

        # remove NaN columns
        nancolumns = np.concatenate((np.where(np.sum(X == 0,axis = 0) == np.shape(X)[0])[0],np.where(np.sum(X2 == 0,axis = 0) == np.shape(X2)[0])[0]))
        if len(nancolumns) > 0:
            X = np.delete(X, nancolumns, axis=1)
            X2 = np.delete(X2, nancolumns, axis=1)
        X = X/np.amax(X,axis=0)
        X2 = X2/np.amax(X2,axis=0)

        # train model
        y = train_all["binned_angles"].to_numpy()
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)

        # plot confusion matrix of prediction on training data
        conf_matrix_all_train[:,:,i] = plotConfusionMatrix(y,clf.predict(X),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

        # plot histogram of frames per angle bin
        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
        ax.hist(clf.predict(X), np.arange(1,number_of_bins+1))
        ax.hist(y, np.arange(1,number_of_bins+1))
        ax.set_title('training data')

        # look at data side-by-side
        ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
        ax.plot(clf.predict(X))
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
        ax.hist(clf.predict(X2), np.arange(1,number_of_bins+1))
        ax.hist(y, np.arange(1,number_of_bins+1))
        ax.set_title('test data')

        # look at data side-by-side
        ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
        ax.plot(clf.predict(X2))
        ax.plot(y)
        ax.legend(["prediction","real"])
        ax.set_title("test data")
        ax.set_ylabel('binned angles')
        ax.set_xlabel('time')

        if not(os.path.exists(str(self.dir) + "/" + "LDA")): 
            os.makedirs(str(self.dir) + "/" + "LDA")
        if object_present == True:
            filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + ".png"
        else:
            filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + "_noObj.png"
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

    if object_present == True:
        filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + ".png"
    else:
        filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + "_noObj.png"
    plt.savefig(filename)
    if self.show_plots: plt.show()
    plt.close()

    prediction_accuracy = compute_prediction_accuracy(np.mean(conf_matrix_all_test, axis=2))

    return prediction_accuracy

# Utility functions ------------------------------------------------------------------------------------------------

def fillMatrix(df,matrix,clu_id):
    for i, i2 in enumerate(df["frames"].unique()):
        d = df.filter(df["frames"] == i2).to_dict(as_series=False)
        matrix[i,np.where(np.in1d(clu_id, d.get('spike_clusters')))[0]] = d.get('spike_count') 

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

def pol2cart(theta):
    '''Parameters:
    - r: float, vector amplitude
    - theta: float, vector angle, radians
    Returns:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end'''
    r = 1
    z = r * np.exp(1j * theta)
    x, y = z.real, z.imag
    return x, y

def cart2pol(x, y):
    '''Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle'''
    z = x + y * 1j
    r,theta = np.abs(z), np.angle(z)
    return theta

if __name__ == '__main__':
    """
    The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
    writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.

    """


#------------------------SOME OLD VERSIONS

# def linear_discriminant_analysis(self, variable, object_present = True):
#     """
#     A function for doing LDA on data
#     variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
#     """
#     epoch_num = 6 # chunks of time for training and testing data
#     number_of_bins = 19

#     # edges for binning firing rate at different angles
#     bin_angles, bin_angle_center = generate_bin_angles(number_of_bins)

#     # subselect relevant times
#     filtered_video_df, angle_filt, title = filter_video_dataframe(self.data_df, variable, object_present)

#     # bin angles
#     filtered_video_df = filtered_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
#     filtered_video_df = filtered_video_df.with_columns(filtered_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in np.arange(len(bin_angle_center))])['category'].alias('binned_angles'))
#     filtered_video_df = filtered_video_df.fill_null(strategy="zero")
#     filtered_video_df = filtered_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 

#     # chunk data into training and test data
#     epoch_edge = np.round(np.linspace(filtered_video_df["frames"].unique().min()-1,filtered_video_df["frames"].unique().max(),epoch_num+1))
#     epoch_df = filtered_video_df.sort("frames")
#     epoch_df = epoch_df.with_columns(epoch_df["frames"].cut(bins = epoch_edge, labels = [str(x) for x in np.arange(epoch_num+2)])['category'].alias('binned_frames'))
#     epoch_df = epoch_df.fill_null(strategy="zero")
#     epoch_df = epoch_df.select([pl.col('binned_frames').apply(float),pl.exclude('binned_frames')]) 

#     conf_matrix_all_train = np.empty((number_of_bins-1,number_of_bins-1,epoch_num))
#     conf_matrix_all_test = np.empty((number_of_bins-1,number_of_bins-1,epoch_num))

#     # LDA
#     for i in np.arange(epoch_num):
#         test = epoch_df.filter((epoch_df['binned_frames'] == (i+1)))
#         train = epoch_df.filter((epoch_df['binned_frames'] != (i+1)))
    
#         # group the training data
#         train_g2 = train.groupby(["frames"]).first()
#         train_all = train.groupby(["frames"]).all()
#         train_all.replace("binned_angles",train_g2['binned_angles'])
#         train_all.replace("binned_frames",train_g2['binned_frames'])

#         # make angle bins equally populated
#         plt.figure(figsize=(20, 16))
#         plt.subplots_adjust(hspace=0.3)
#         d_new = EqualAngleBins_df(train_all)

#         # group the test data
#         test_g2 = test.groupby(["frames"]).first()
#         test_all = test.groupby(["frames"]).all()
#         test_all.replace("binned_angles",test_g2['binned_angles'])
#         test_all.replace("binned_frames",test_g2['binned_frames'])

#         # make angle bins equally populated
#         d_new_test = EqualAngleBins_df(test_all)
        
#         # make train matrix of frames x clusters
#         X = np.zeros((len(d_new["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
#         clu = epoch_df["spike_clusters"].unique().to_numpy()
#         fillMatrix(d_new,X,clu)
#         if clu[0] == 0: X = X[:,1:]

#         # make test matrix of frames x clusters
#         X2 = np.zeros((len(d_new_test["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
#         fillMatrix(d_new_test,X2,clu)
#         if clu[0] == 0: X2 = X2[:,1:]

#         # remove NaN columns
#         nancolumns = np.concatenate((np.where(np.sum(X == 0,axis = 0) == np.shape(X)[0])[0],np.where(np.sum(X2 == 0,axis = 0) == np.shape(X2)[0])[0]))
#         if len(nancolumns) > 0:
#             X = np.delete(X, nancolumns, axis=1)
#             X2 = np.delete(X2, nancolumns, axis=1)
#         X = X/np.amax(X,axis=0)
#         X2 = X2/np.amax(X2,axis=0)

#         # train model
#         y = d_new["binned_angles"].to_numpy()
#         clf = LinearDiscriminantAnalysis()
#         clf.fit(X, y)

#         # plot confusion matrix of prediction on training data
#         conf_matrix_all_train[:,:,i] = plotConfusionMatrix(y,clf.predict(X),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

#         # plot histogram of frames per angle bin
#         ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
#         ax.hist(clf.predict(X))
#         ax.hist(y)
#         ax.set_title('training data')

#         # look at data side-by-side
#         ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
#         ax.plot(clf.predict(X))
#         ax.plot(y)
#         ax.legend(["prediction","real"])
#         ax.set_title("training data")
#         ax.set_ylabel('binned angles')
#         ax.set_xlabel('time')

#         # plot confusion matrix of prediction on test data
#         y = d_new_test["binned_angles"].to_numpy()
#         conf_matrix_all_test[:,:,i] = plotConfusionMatrix(y,clf.predict(X2),'test data',plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

#         # plot histogram of frames per angle bin
#         ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
#         ax.hist(clf.predict(X2))
#         ax.hist(y)
#         ax.set_title('test data')

#         # look at data side-by-side
#         ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
#         ax.plot(clf.predict(X2))
#         ax.plot(y)
#         ax.legend(["prediction","real"])
#         ax.set_title("test data")
#         ax.set_ylabel('binned angles')
#         ax.set_xlabel('time')

#         if not(os.path.exists(str(self.dir) + "/" + "LDA")): 
#             os.makedirs(str(self.dir) + "/" + "LDA")
#         if object_present == True:
#             filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + ".png"
#         else:
#             filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_epoch" + str(i+1) + "_noObj.png"
#         plt.savefig(filename)
#         if self.show_plots: plt.show()
#         plt.close()
    
#     # plot average confusion matrix
#     plt.figure(figsize=(20, 16))
#     plt.subplots_adjust(hspace=0.3)
#     ax = plt.subplot(1,2,1)
#     ax.imshow(np.mean(conf_matrix_all_train, axis=2), cmap = "Blues", vmin = 0, vmax = 1)
#     ax.set_ylabel('real')
#     ax.set_xlabel('predicted')
#     ax.set_title('train')

#     ax = plt.subplot(1,2,2)
#     ax.imshow(np.mean(conf_matrix_all_test, axis=2), cmap = "Blues", vmin = 0, vmax = 1)
#     ax.set_ylabel('real')
#     ax.set_xlabel('predicted')
#     ax.set_title('test')

#     if object_present == True:
#         filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + ".png"
#     else:
#         filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_avg" + "_noObj.png"
#     plt.savefig(filename)
#     if self.show_plots: plt.show()
#     plt.close()



# def linear_discriminant_analysis_OLD(self, variable, object_present = True):
#     """
#     A function for doing LDA on data
#     variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
#     """
#     epoch_num = 6 # chunks of time for training and testing data

#     # edges for binning firing rate at different angles
#     bin_angles, bin_angle_center = generate_bin_angles(number_of_bins = 19)

#     # subselect relevant times
#     filtered_video_df, angle_filt, title = filter_video_dataframe(self.data_df, variable, object_present)

#     # bin angles
#     filtered_video_df = filtered_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
#     filtered_video_df = filtered_video_df.with_columns(filtered_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in np.arange(len(bin_angle_center))])['category'].alias('binned_angles'))
#     filtered_video_df = filtered_video_df.fill_null(strategy="zero")
#     filtered_video_df = filtered_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 

#     # chunk data into training and test data
#     epoch_edge = np.round(np.linspace(filtered_video_df["frames"].unique().min()-1,filtered_video_df["frames"].unique().max(),epoch_num+1))
#     epoch_df = filtered_video_df.sort("frames")
#     epoch_df = epoch_df.with_columns(epoch_df["frames"].cut(bins = epoch_edge, labels = [str(x) for x in np.arange(epoch_num+2)])['category'].alias('binned_frames'))
#     epoch_df = epoch_df.fill_null(strategy="zero")
#     epoch_df = epoch_df.select([pl.col('binned_frames').apply(float),pl.exclude('binned_frames')]) 

#     # LDA
#     train = epoch_df.filter((epoch_df['binned_frames'] == 1) | 
#                             (epoch_df['binned_frames'] == 3) | 
#                             (epoch_df['binned_frames'] == 5))
    
#     # group the training data
#     train_g2 = train.groupby(["frames"]).first()
#     train_all = train.groupby(["frames"]).all()
#     train_all.replace("binned_angles",train_g2['binned_angles'])
#     train_all.replace("binned_frames",train_g2['binned_frames'])

#     # make angle bins equally populated
#     plt.figure(figsize=(20, 16))
#     plt.subplots_adjust(hspace=0.3)
#     d_new = EqualAngleBins_df(train_all)
#     # d_new = train_all # use this line if you want to use all data, not equal bins

#     X = np.zeros((len(d_new["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
#     clu = epoch_df["spike_clusters"].unique().to_numpy()
    
#     # make matrix of frames x clusters
#     fillMatrix(d_new,X,clu)
#     if clu[0] == 0: X = X[:,1:]
#     # X = X[:,np.where(np.sum(X == 0,axis = 0) < np.shape(X)[0])[0]]
#     X = X/np.amax(X,axis=0)

#     # train model
#     y = d_new["binned_angles"].to_numpy()
    
#     clf = LinearDiscriminantAnalysis()
#     clf.fit(X, y)

#     # plot confusion matrix of prediction on training data
#     plotConfusionMatrix(y,clf.predict(X),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

#     # plot histogram of frames per angle bin
#     ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
#     ax.hist(clf.predict(X))
#     ax.hist(y)
#     ax.set_title('training data')

#     # look at data side-by-side
#     ax = plt.subplot2grid(shape=(4, 2), loc=(0, 0), colspan=2)
#     ax.plot(clf.predict(X))
#     ax.plot(y)
#     ax.legend(["prediction","real"])
#     ax.set_title("training data")
#     ax.set_ylabel('binned angles')
#     ax.set_xlabel('time')

#     # predict test data
#     test = epoch_df.filter((epoch_df['binned_frames'] == 2) | 
#                            (epoch_df['binned_frames'] == 4) | 
#                            (epoch_df['binned_frames'] == 6))
    
#     # group the test data
#     test_g2 = test.groupby(["frames"]).first()
#     test_all = test.groupby(["frames"]).all()
#     test_all.replace("binned_angles",test_g2['binned_angles'])
#     test_all.replace("binned_frames",test_g2['binned_frames'])

#     # make angle bins equally populated
#     d_new_test = EqualAngleBins_df(test_all)
#     # d_new_test = test_all

#     X = np.zeros((len(d_new_test["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
#     y = d_new_test["binned_angles"].to_numpy()
    
#     # make matrix of frames x clusters
#     fillMatrix(d_new_test,X,clu)
#     if clu[0] == 0: X = X[:,1:]
#     # X = X[:,np.where(np.sum(X == 0,axis = 0) < np.shape(X)[0])[0]]
#     X = X/np.amax(X,axis=0)

#     # plot confusion matrix of prediction on test data
#     plotConfusionMatrix(y,clf.predict(X),'test data',plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

#     # plot histogram of frames per angle bin
#     ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
#     ax.hist(clf.predict(X))
#     ax.hist(y)
#     ax.set_title('test data')

#     # look at data side-by-side
#     ax = plt.subplot2grid(shape=(4, 2), loc=(1, 0), colspan=2)
#     ax.plot(clf.predict(X))
#     ax.plot(y)
#     ax.legend(["prediction","real"])
#     ax.set_title("test data")
#     ax.set_ylabel('binned angles')
#     ax.set_xlabel('time')

#     if not(os.path.exists(str(self.dir) + "/" + "LDA")): 
#         os.makedirs(str(self.dir) + "/" + "LDA")
#     if object_present == True:
#         filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + ".png"
#     else:
#         filename = str(self.dir) + "/" + "LDA" + "/" + str(self.cluster_type) + "_LDA_" + str(title) + "_noObj.png"
#     plt.savefig(filename)
#     if self.show_plots: plt.show()
#     plt.close()