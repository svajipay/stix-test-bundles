import math
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from tabulate import tabulate

# For Facebook's Prophet
from prophet import Prophet

# For EmaDetector
from timemachines.skaters.simple.hypocraticensemble import quick_aggressive_ema_ensemble

# For HstDetector
from river import anomaly, compose, preprocessing
from river import time_series

import stumpy


class Detector:
    """Predict timeseries anomalies"""
    def __str__(self):
        return self.__class__.__name__

    def predict_one(self, timestamp, value):
        return 0

    def predict(self, df: pd.DataFrame, learning: bool = False):
        """
        Add a predicted `label` in `df`

        Parameters
        ----------
        df: A DataFrame with columns `timestamp` and `value`
        learning: if True, still in "learning" (burn-in) phase

        Returns
        -------
        A list of labels, one for each row in df
        """
        return [0] * len(df.index)


# https://facebook.github.io/prophet/docs/additional_topics.html#updating-fitted-models
def warm_start_params(m):
    """
    Retrieve parameters from a trained model in the format used to initialize a new Stan model.
    Note that the new Stan model must have these same settings:
        n_changepoints, seasonality features, mcmc sampling
    for the retrieved parameters to be valid for the new model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0][0]
        else:
            res[pname] = np.mean(m.params[pname])
    for pname in ['delta', 'beta']:
        if m.mcmc_samples == 0:
            res[pname] = m.params[pname][0]
        else:
            res[pname] = np.mean(m.params[pname], axis=0)
    return res


# Facebook Prophet is currently (2023-02-27) unusable.
# Bug in 1.1.2: https://github.com/facebook/prophet/issues/2354 (works in 1.1.1)
# Bug in 1.1.1: https://github.com/facebook/prophet/issues/2229 (fixed in 1.1.2)
class ProphetDetector(Detector):
    def __init__(self):
        self.model = None

    def predict_one(self, timestamp, value):
        #df = pd.DataFrame({'ds': [timestamp], 'y': [value]})
        return 0

    def predict(self, df, learning=False):
        print(df.head())
        _df = pd.DataFrame({'ds': df['timestamp'].tolist(), 'y': df['value'].tolist()})
        if self.model:
            forecast = self.model.predict(_df)
            _df['upper'] = forecast['yhat_upper']  #TODO: need a k?
            _df['lower'] = forecast['yhat_lower']  #TODO: need a k?
        else:
            learning = True
        _df['label'] = 0
        print(_df.head())
        if not learning:
            _df[(_df['y'] < _df['lower']) | (_df['y'] > _df['upper'])] = 1

        # Do a "warn start" re-fit
        update = Prophet().fit(_df)
        if not self.model:
            self.model = update
        else:
            self.model = Prophet().fit(_df, init=warm_start_params(self.model))

        return _df['label'].tolist()


class EmaDetector(Detector):
    def __init__(self, k=3, n=1, protect=False):
        self.k = k
        self.n_steps = n
        assert(n == 1)  #TODO: support n > 1?
        self.protect = protect
        self.xi = None
        self.x_std = None
        self.state = {}
        self.prev = 0  # Previously condition: abnormal low (-1), normal (0), high (1)

    def __str__(self):
        return f'{self.__class__.__name__}(k={self.k}, n={self.n_steps})'

    def _evaluate(self, lower, value, upper):
        label = 0
        if self.prev < 0:
            # Previous sample was abnormally low
            label = 1
            if value > upper:
                pass # Smooth out "ripples?" #self.prev = 1
            elif value > lower:
                #TODO: Introduce a lag? label = 0
                self.prev = 0
        elif self.prev == 0:
            # Previous sample normal
            if value < lower:
                label = 1
                self.prev = -1
            elif value > upper:
                label = 1
                self.prev = 1
            else:
                self.prev = 0
        else:
            # Previous sample was abnormally high
            label = 1
            if value < lower:
                pass # self.prev = 1
            elif value < upper:
                #TODO: Introduce a lag? label = 0
                self.prev = 0
        return label

    def predict_one(self, timestamp, value):
        # Use the previously seen sample to generate a forecast for this step
        label = 0
        if self.x_std is not None:
            delta = self.k * self.x_std[0]
            upper = self.xi[0] + delta
            lower = self.xi[0] - delta

            self.prev = 0  # TEMP: disable the "state machine" in _evaluate
            label = self._evaluate(lower, value, upper)
            # if label:
            #     print('anomaly:', timestamp, value, lower, upper)

        if self.protect and label == 1:
            # We think it's an anomaly, so use our forecast instead
            value = self.xi[0]

        # Now "fit" with the current sample
        self.xi, self.x_std, self.state = quick_aggressive_ema_ensemble(
            y=value,
            s=self.state,
            k=self.n_steps,
            t=timestamp)

        #self.prev = label
        return label

    def predict(self, df, learning=False):
        #TODO: timemachines will predict more than 1 step ahead, so do that instead of this loop?
        labels = []
        for i in df.itertuples():
            labels.append(self.predict_one(i.timestamp, i.value))
        return labels


class HstDetector(Detector):
    def __init__(self, threshold: float = 0.90):
        # self.hst = anomaly.HalfSpaceTrees(
        #     n_trees=5,
        #     height=3,
        #     window_size=3,
        #     seed=42
        # )
        self.threshold = threshold
        self.model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(seed=42)
        )

    def __str__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'

    def predict(self, df, learning=False):
        scores = []
        for i in df.itertuples():
            x = {'value': i.value}
            scores.append(self.model.score_one(x))
            self.model = self.model.learn_one(x)
        return [1 if score > self.threshold else 0 for score in scores]


class HoltWintersDetector(Detector):
    def __init__(self,
                 alpha = 1.0,
                 beta = 0.0,
                 gamma = 0.0,
                 seasonality: int = 12,  # ?
                 multiplicative: bool = True,
                 threshold: float = 3.0
    ):
        self.model = time_series.HoltWinters(
            alpha=alpha,
            #beta=beta,
            #gamma=gamma,
            #seasonality=seasonality,
            #multiplicative=multiplicative
        )
        self.threshold = threshold

    def __str__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'

    def predict(self, df, learning=False):
        preds = []
        for i in df.itertuples():
            if learning:
                pred = i.value
            else:
                pred = self.model.forecast(1)[0]
            if math.isclose(i.value, 0.0):
                # Avoid division by zero
                error = (i.value - pred) / 1e-09
            else:
                error = (i.value - pred) / i.value
            if abs(error) > self.threshold:
                preds.append(1)
            else:
                preds.append(0)

            # Give the "model" the real value
            self.model = self.model.learn_one(i.value)

        return preds


class SnarimaxDetector(Detector):
    def __init__(self,
                 period,
                 threshold: float = 3.0
    ):
        self.model = time_series.SNARIMAX(
            p=period,
            d=1,
            q=period,
            m=period,
            sd=1
        )
        self.threshold = threshold

    def __str__(self):
        return f'{self.__class__.__name__}(period={self.period}, threshold={self.threshold})'

    def predict(self, df, learning=False):
        preds = []
        for i in df.itertuples():
            if True:  ##########learning:
                pred = i.value
            else:
                assert False, 'This detector is way too slow'
                pred = self.model.forecast(1)[0]
            if math.isclose(i.value, 0.0):
                # Avoid division by zero
                error = (i.value - pred) / 1e-09
            else:
                error = (i.value - pred) / i.value
            if abs(error) > self.threshold:
                preds.append(1)
            else:
                preds.append(0)

            # Give the "model" the real value
            self.model = self.model.learn_one(i.value)

        return preds


## STUMP

class StumpyDetector(Detector):
    def __init__(self, window=None):
        self.stream = None
        self.window = window
        self.train = np.ndarray(self.window)
        self.max_dist = 0.0

    def _determine_window(self, df):
        #TODO
        #delta = df['timestamp'].max() - df['timestamp'].min()
        self.window = len(self.train)  #df.index)

    def predict_one(self, timestamp, value):
        #df = pd.DataFrame({'ds': [timestamp], 'y': [value]})
        return 0

    def predict(self, df, learning=False):
        if learning:
            #self.train.extend(df['value'].astype('float64').tolist())
            self.train = np.concatenate([self.train, df['value'].astype('float64').values])
            return [0 for i in range(len(df.index))]
        if not self.stream:
            if not self.window:
                self._determine_window(df)
            #print('train len =', len(self.train))
            #print('initial train:', self.train)
            arr = np.asarray(self.train)
            self.train = []
            self.stream = stumpy.stumpi(arr, self.window, egress=True)
        scores = []
        for i in df.itertuples():
            self.stream.update(i.value)
            #scores.append(self.stream.P_[-1])
        #print(self.stream.P_)
        #threshold = self.stream.P_.max()
        #return [1 if score > threshold else 0 for score in scores]
        discord_idx = np.argsort(self.stream.P_)[-1]
        max_dist = self.stream.P_[discord_idx]
        start_idx = self.window - len(df.index)
        scores = [0 for i in range(len(df.index))]
        #if not np.isinf(max_dist) and max_dist > self.max_dist and discord_idx > start_idx:
        if not np.isinf(max_dist) and discord_idx > start_idx:
            #print('Discord at', discord_idx, max_dist, len(self.stream.P_))
            #print('start_idx =', start_idx)
            idx = discord_idx - start_idx - 1
            if 0 < idx < len(scores):
                scores[idx] = 1
            self.max_dist = max_dist
        return scores

# Data sources
##############


class DataSrc:
    def __init__(self):
        self.df = None
        self.offset = 0

    def get(self, n):
        """
        Return a DataFrame with n rows, columns "timestamp", "value", "label"
        """
        start = self.offset
        end = min(self.offset + n, len(self.df.index))
        self.offset += n
        result = self.df.iloc[start:end]
        return result


class AIOpsKPI(DataSrc):
    def __init__(self, filename, kpi_id=None):
        super().__init__()
        self.filename = filename
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        if not kpi_id:
            kpi_id = df.iloc[0]['KPI ID']
        self.kpi_id = kpi_id
        self.df = df[df['KPI ID'] == self.kpi_id]

    def __str__(self):
        return f'{self.filename}; (kpi_id={self.kpi_id})'


class KaggleWafer(DataSrc):
    FILENAME = '/tmp/Data/timeseries/KaggleWafer/Train.csv'
    def __init__(self, feature_name=None):
        super().__init__()
        if not feature_name:
            feature_name = 'feature_1'
        self.feature_name = feature_name
        self.df = pd.read_csv(self.FILENAME, usecols=[feature_name, 'Class'])
        self.df.columns = ['value', 'label']
        self.df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(self.df.index), freq='T')

    def __str__(self):
        return f'{self.FILENAME}; (feature_name={self.feature_name})'


class OrigSynthetic(DataSrc):
    """Synthetic data I created"""
    FILENAME = '/tmp/github/kestrel-jupyter/notebooks/synthetic_1.parquet'
    def __init__(self):
        super().__init__()
        df = pd.read_parquet(self.FILENAME).reset_index()
        df.columns = ['timestamp', 'value']
        df['label'] = 0
        # Introduce some anomalies
        for i in range(18, 21):
            #FIXME: there's a better way to do this
            ts = pd.to_datetime(f'2023-01-29 03:{i}:00')
            df['value'] = np.where(df['timestamp'] == ts, df['value'] * 2, df['value'])
            df['label'] = np.where(df['timestamp'] == ts, 1, df['label'])
        self.df = df
        print(self.df.head())
        print(self.df['label'].value_counts())

    def __str__(self):
        return f'{self.FILENAME}'


class Synthetic(DataSrc):
    """Synthetic data I created"""
    FILENAME = '/tmp/synthetic_anomalies.parquet'
    def __init__(self):
        super().__init__()
        self.df = pd.read_parquet(self.FILENAME)
        print(self.df.head())
        print(self.df['label'].value_counts())

    def __str__(self):
        return f'{self.FILENAME}'
   

def print_table(row_labels, col_labels, data):
    rows = []
    for i, row in enumerate(data):
        rows.append([row_labels[i]] + row.tolist())
    print(tabulate(rows, headers=col_labels))


def score(model, datasrc, burnin, n, stop=None, soft=True):
    i = 0
    batch = datasrc.get(n)
    results = []
    labels = []
    start = time.time()
    while len(batch.index):
        learning = (i < burnin)
        # Fudge scoring a bit?
        if soft:
            tmps = batch['label'].tolist()
            if 1 in tmps:
                tmps = [1 for i in tmps]
            labels.extend(tmps)
            preds = model.predict(batch, learning)
            if 1 in preds:
                preds = [1 for i in preds]
            results.extend(preds)
        else:
            labels.extend(batch['label'].tolist())
            results.extend(model.predict(batch, learning))
        i += n
        if stop and stop <= i:
            break
        batch = datasrc.get(n)
    elapsed = round(time.time() - start, 2)

    preds = pd.Series(results)
    begin = burnin
    if stop:
        end = stop
    else:
        end = None
    #trues = datasrc.df['label'].iloc[begin:end].astype(int) #TODO: need method call
    trues = labels[begin:]
    preds = preds.iloc[begin:end].astype(int)
    #assert len(trues) == len(preds)
    acc = accuracy_score(trues, preds)
    f1s = f1_score(trues, preds)
    prfs = precision_recall_fscore_support(trues, preds)
    cm = confusion_matrix(trues, preds)

    print('Model:', str(model))
    print(f'Elapsed time: {elapsed}s')
    print('Total samples:', len(preds.index) + burnin)
    print(f'Burn-in: {burnin} samples')
    print('Steps:', n)
    print('Accuracy:', round(acc, 3))
    print('F1 Score:', round(f1s, 3))
    print()
    metrics = ['Precision', 'Recall', 'FScore', 'Support']
    classes = ['normal', 'anomaly']
    print_table(metrics, ['metric'] + classes, prfs)
    print()
    print_table(classes, classes, cm)
    return {
        'model': str(model),
        'dataset': ds,
        'elapsed': elapsed,
        'accuracy': acc,
        'f1': f1s,
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__file__,
        description='Timeseries anomaly detection in near-real-time')
    parser.add_argument('-k', '--k-coeff', default=3.0, type=float)
    parser.add_argument('-t', '--threshold', default=0.95, type=float)
    parser.add_argument('-l', '--learn-samples', default=1000, type=int)
    parser.add_argument('-s', '--steps', default=5, type=int)
    parser.add_argument('-p', '--protect', action='store_true', default=False)
    parser.add_argument('--max', default=None, type=int)
    parser.add_argument('--soft', action='store_true', default=False)
    args = parser.parse_args()
    #model = EmaDetector(k=args.k_coeff, protect=args.protect)
    #model = ProphetDetector()
    #model = HstDetector(args.threshold)

    # ds = AIOpsKPI('/tmp/Data/timeseries/phase2_train.csv',
    #               #kpi_id='a07ac296-de40-3a7c-8df3-91f642cc14d0')
    #               kpi_id='0efb375b-b902-3661-ab23-9a0bb799f4e3')
    #               #)
    # ds = KaggleWafer('feature_1')

    datasets = [
        #KaggleWafer('feature_1'),
        #KaggleWafer('feature_2'),
        #KaggleWafer('feature_3'),
        #AIOpsKPI('/tmp/timeseries/phase2_train.csv', kpi_id='a07ac296-de40-3a7c-8df3-91f642cc14d0'),

        Synthetic(),
    ]

    stats = []
    for ds in datasets:
        print('Data set:', ds)

        # Get sampling interval
        #print(ds.df['timestamp'].max() - ds.df['timestamp'].min())
        interval = ds.df.iloc[0:2]['timestamp'].diff().iloc[1]
        window = int(pd.Timedelta(1, 'day') / interval)
        learn_samples = window
        n = 2
        while learn_samples < 1000:
            learn_samples = int(pd.Timedelta(n, 'day') / interval)
            n += 1

        #print(interval, learn_samples)
        #continue

        print()

        models = [
            EmaDetector(k=args.k_coeff, protect=args.protect),
            HstDetector(args.threshold),
            HoltWintersDetector(seasonality=window),  # args.threshold),
            #SnarimaxDetector(period=window),
            StumpyDetector(window=window),
        ]
    
        for model in models:
            stats.append(score(model, ds, learn_samples, args.steps, args.max, args.soft))

            # Reset datasource
            ds.offset = 0

            print()
            print('-' * 78, '\n')

    print('\nSummary')
    print(tabulate(stats, headers='keys'))
    pd.DataFrame(stats).to_csv('report.csv')
