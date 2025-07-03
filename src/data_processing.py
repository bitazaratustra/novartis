import pandas as pd

def load_data(feats_path, target_path):
    df_feats = pd.read_csv(feats_path)
    df_target = pd.read_csv(target_path)
    return pd.concat([df_feats, df_target], axis=1)

def preprocess_data(df):
    nicotine_map = {
        "CL6": "Last Day",
        "CL5": "Last Week",
        "CL4": "Last Month",
        "CL3": "Last Year",
        "CL2": "Last Decade",
        "CL1": "Over a Decade Ago",
        "CL0": "Never Used"
    }

    df['nicotine_label'] = df['nicotine'].map(nicotine_map)
    df['target'] = df['nicotine'].apply(lambda x: 1 if x in ['CL1', 'CL2'] else 0)

    ordinal_map = {
        "CL0": 0, "CL1": 1, "CL2": 2,
        "CL3": 3, "CL4": 4, "CL5": 5, "CL6": 6
    }

    ordinal_cols = [
        'alcohol', 'amphet', 'amyl', 'benzos', 'caff', 'cannabis', 'choc',
        'coke', 'crack', 'ecstasy', 'heroin', 'ketamine', 'legalh', 'lsd',
        'meth', 'mushrooms', 'nicotine', 'semer', 'vsa'
    ]

    for col in ordinal_cols:
        if col in df.columns:
            df[col] = df[col].map(ordinal_map)

    return df

def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    df = load_data('data/raw/feats.csv', 'data/raw/target.csv')
    df = preprocess_data(df)
    save_processed_data(df, 'data/processed/final_dataset.csv')
