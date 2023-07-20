import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.stats import vonmises

class AngleDecoder:
    def __init__(self, data, target_col, model='lda'):
        self.data = data
        self.target_col = target_col
        self.angle_bins = None

        if model == 'lda':
            self.model = LinearDiscriminantAnalysis()
        elif model == 'qda':
            self.model = QuadraticDiscriminantAnalysis()
        elif model == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=1)

    def preprocess_data(self, num_bins=20):
        print('Preprocessing data...')
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
        self.data[self.target_col + '_bins'] = discretizer.fit_transform(self.data[[self.target_col]])
        self.data[self.target_col + '_bins'] = self.data[self.target_col + '_bins'].astype(int)
        self.angle_bins = discretizer.bin_edges_[0]

    def prepare_data(self):
        print('Preparing data...')
        scaler = StandardScaler()
        X = pd.pivot_table(self.data, values='spike_count', index='frames', columns='spike_clusters', fill_value=0)
        X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
        y = self.data.groupby('frames')[self.target_col + '_bins'].mean().round(0).astype(int)
        return X, y

    def evaluate_model(self, X, y):
        print('Evaluating model...')
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        pca = PCA(n_components=10)
        scores = []
        for train_index, test_index in tqdm(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            scores.append(score)
        print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    def run(self):
        self.preprocess_data()
        X, y = self.prepare_data()
        self.evaluate_model(X, y)

def main():
    np.random.seed(0)

    n_frames = 10000
    n_clusters = 20

    kappas = np.random.uniform(5, 15, n_clusters)

    hdir = np.random.uniform(0, 360, n_frames)
    preferred_dirs = np.linspace(0, 360, n_clusters)

    spike_counts = []
    for cluster in range(n_clusters):
        tuning = vonmises.pdf(np.radians(hdir - preferred_dirs[cluster]), kappas[cluster])
        spike_counts.append(np.random.poisson(tuning))
    spike_counts = np.array(spike_counts).T

    # Generate a random column
    random_col = np.random.uniform(0, 360, n_frames*n_clusters)

    data_df = pd.DataFrame({
        'frames': np.repeat(np.arange(n_frames), n_clusters),
        'spike_clusters': np.tile(np.arange(n_clusters), n_frames),
        'hdir': np.repeat(hdir, n_clusters),
        'spike_count': spike_counts.flatten(),
        'correlated_col': np.repeat(hdir, n_clusters) * 0.5,
        'random_col': random_col
    })

    decoder = AngleDecoder(data_df, target_col='correlated_col', model='rf')
    decoder.run()

if __name__ == '__main__':
    main()
