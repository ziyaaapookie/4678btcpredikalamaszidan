import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                  'close_time', 'quote_asset_volume', 'number_of_trades',
                  'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df['date'] = pd.to_datetime(df['open_time'])
    df = df[['date', 'high', 'low', 'open', 'close', 'volume']]
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Prediction_360D'] = df['close'].shift(-360)
    df['high_shifted'] = df['high'].shift(360)
    df['low_shifted'] = df['low'].shift(360)
    df['open_shifted'] = df['open'].shift(360)
    df['volume_shifted'] = df['volume'].shift(360)

    df.dropna(inplace=True)
    return df