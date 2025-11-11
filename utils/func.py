import pandas as pd

from .segmentation import extract_segments


def extract_mortality_events(timeseries, label, num_segments, threshold):
    segments = extract_segments(timeseries, label, num_segments)
    df_events = pd.DataFrame(segments)
    df_events.rename(columns={
        'label': 'id',
        'segment': 'mortality',
        'length': 'duration',
    }, inplace=True)
    df_events.set_index('id', inplace=True)
    df_events['start_week'] = timeseries.index[df_events['start_index']]
    df_events['end_week'] = timeseries.index[df_events['end_index']]
    df_events['threshold'] = threshold
    df_events['excess_mortality'] = df_events['mortality'] - threshold
    df_events.drop([
        'indices',
        'start_index',
        'end_index',
    ], axis=1, inplace=True)
    return df_events
