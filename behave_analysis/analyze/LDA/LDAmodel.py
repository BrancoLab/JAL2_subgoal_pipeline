# OS libaries
from loguru import logger
import numpy as np
import os
import polars as pl
import matplotlib 
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from loguru import logger
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# import functions
from behave_analysis.visualize.visualize_efizz import filter_video_dataframe, generate_bin_angles 

def linear_discriminant_analysis(self, variable, object_present = True):
    """
    A function for doing LDA on data
    variable: what we're trying to predict (e.g. head_shelter_angle), it needs to be one of the columns of video_df
    """
    epoch_num = 4 # chunks of time for training and testing data
    number_of_bins = 13

    # edges for binning firing rate at different angles
    bin_angles, bin_angle_center = generate_bin_angles(number_of_bins)

    # subselect relevant times
    filtered_video_df, angle_filt, title = filter_video_dataframe(self.data_df, variable, object_present)

    # bin angles
    filtered_video_df = filtered_video_df.sort(angle_filt) # polars can be annoying, when using cut it doesn't preserve order :/
    filtered_video_df = filtered_video_df.with_columns(filtered_video_df[angle_filt].cut(bins = bin_angles, labels = [str(x) for x in np.arange(len(bin_angle_center))])['category'].alias('binned_angles'))
    filtered_video_df = filtered_video_df.fill_null(strategy="zero")
    filtered_video_df = filtered_video_df.select([pl.col('binned_angles').apply(float),pl.exclude('binned_angles')]) 

    # chunk data into training and test data
    epoch_edge = np.round(np.linspace(filtered_video_df["frames"].unique().min()-1,filtered_video_df["frames"].unique().max(),epoch_num+1))
    epoch_df = filtered_video_df.sort("frames")
    epoch_df = epoch_df.with_columns(epoch_df["frames"].cut(bins = epoch_edge, labels = [str(x) for x in np.arange(epoch_num+2)])['category'].alias('binned_frames'))
    epoch_df = epoch_df.fill_null(strategy="zero")
    epoch_df = epoch_df.select([pl.col('binned_frames').apply(float),pl.exclude('binned_frames')]) 

    conf_matrix_all_train = np.empty((number_of_bins-1,number_of_bins-1,epoch_num))
    conf_matrix_all_test = np.empty((number_of_bins-1,number_of_bins-1,epoch_num))

    # LDA
    for i in np.arange(epoch_num):
        test = epoch_df.filter((epoch_df['binned_frames'] == (i+1)))
        train = epoch_df.filter((epoch_df['binned_frames'] != (i+1)))
    
        # group the training data
        train_g2 = train.groupby(["frames"]).first()
        train_all = train.groupby(["frames"]).all()
        train_all.replace("binned_angles",train_g2['binned_angles'])
        train_all.replace("binned_frames",train_g2['binned_frames'])

        # make angle bins equally populated
        plt.figure(figsize=(20, 16))
        plt.subplots_adjust(hspace=0.3)
        d_new = EqualAngleBins_df(train_all)

        # group the test data
        test_g2 = test.groupby(["frames"]).first()
        test_all = test.groupby(["frames"]).all()
        test_all.replace("binned_angles",test_g2['binned_angles'])
        test_all.replace("binned_frames",test_g2['binned_frames'])

        # make angle bins equally populated
        d_new_test = EqualAngleBins_df(test_all)
        
        # make train matrix of frames x clusters
        X = np.zeros((len(d_new["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
        clu = epoch_df["spike_clusters"].unique().to_numpy()
        fillMatrix(d_new,X,clu)
        if clu[0] == 0: X = X[:,1:]

        # make test matrix of frames x clusters
        X2 = np.zeros((len(d_new_test["frames"].unique()),len(epoch_df["spike_clusters"].unique())))
        fillMatrix(d_new_test,X2,clu)
        if clu[0] == 0: X2 = X2[:,1:]

        # remove NaN columns
        nancolumns = np.concatenate((np.where(np.sum(X == 0,axis = 0) == np.shape(X)[0])[0],np.where(np.sum(X2 == 0,axis = 0) == np.shape(X2)[0])[0]))
        if len(nancolumns) > 0:
            X = np.delete(X, nancolumns, axis=1)
            X2 = np.delete(X2, nancolumns, axis=1)
        X = X/np.amax(X,axis=0)
        X2 = X2/np.amax(X2,axis=0)

        # train model
        y = d_new["binned_angles"].to_numpy()
        clf = LinearDiscriminantAnalysis()
        clf.fit(X, y)

        # plot confusion matrix of prediction on training data
        conf_matrix_all_train[:,:,i] = plotConfusionMatrix(y,clf.predict(X),'training data',plt.subplot2grid(shape=(4, 2), loc=(2, 0)))

        # plot histogram of frames per angle bin
        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 0))
        ax.hist(clf.predict(X))
        ax.hist(y)
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
        y = d_new_test["binned_angles"].to_numpy()
        conf_matrix_all_test[:,:,i] = plotConfusionMatrix(y,clf.predict(X2),'test data',plt.subplot2grid(shape=(4, 2), loc=(2, 1)))

        # plot histogram of frames per angle bin
        ax = plt.subplot2grid(shape=(4, 2), loc=(3, 1))
        ax.hist(clf.predict(X2))
        ax.hist(y)
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
    ax.imshow(np.mean(conf_matrix_all_train, axis=2), cmap = "Blues")
    ax.set_ylabel('real')
    ax.set_xlabel('predicted')
    ax.set_title('train')

    ax = plt.subplot(1,2,2)
    ax.imshow(np.mean(conf_matrix_all_test, axis=2), cmap = "Blues")
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

# Utility functions ------------------------------------------------------------------------------------------------

def fillMatrix(df,matrix,clu_id):
    for i, i2 in enumerate(df["frames"].unique()):
        d = df.filter(df["frames"] == i2).to_dict(as_series=False)
        matrix[i,np.where(np.in1d(clu_id, d.get('spike_clusters')))[0]] = d.get('spike_count') 

def plotConfusionMatrix(y,x,title,axy):
    conf = confusion_matrix(y, x)
    conf = conf.astype('float64')
    conf = conf/np.sum(conf,axis=1)

    axy.imshow(conf, cmap = "Blues")
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

if __name__ == '__main__':
    """
    The below logic is used to run the module as a standalone module for the purpose of testing toy data. It is currently set up to run on synthetic data that matches the matlab code
    writen to produce the model from Campagner et al., 2022. Should the functions above be modified, this code will need to be modified to match.

    """


#------------------------SOME OLD VERSIONS
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