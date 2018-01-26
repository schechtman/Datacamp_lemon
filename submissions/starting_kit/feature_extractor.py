import pandas as pd

def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0


def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    d = d[columns]
    return d


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y=None):
        global column_dummies
        if y is not None:
            column_dummies = pd.concat(
            [X_df.get(['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'PurchDate', 'WarrantyCost', 'RefId', 'VehYear', 'VehicleAge']),
             pd.get_dummies(X_df.Size, prefix = 'Size', drop_first=True),
             pd.get_dummies(X_df.Auction, prefix='Auction', drop_first=True),
             pd.get_dummies(X_df.Color, prefix='Color', drop_first=True),
             pd.get_dummies(X_df.Transmission, prefix='Transmission', drop_first=True),
             pd.get_dummies(
                 X_df.Nationality, prefix='Nationality', drop_first=True),
             pd.get_dummies(X_df.Model, prefix='Model', drop_first=True),
             pd.get_dummies(X_df.SubModel, prefix='SubModel', drop_first=True),
             pd.get_dummies(X_df.Make, prefix="Make", drop_first=True),
             pd.get_dummies(X_df.WheelType, prefix="WheelType", drop_first=True),
             pd.get_dummies(X_df.TopThreeAmericanName, prefix="TopThreeAmericanName", drop_first=True),
             pd.get_dummies(X_df.VNZIP1, prefix="VNZIP1", drop_first=True),
             ],
            axis=1).columns
        return self

    def transform(self, X_df):
        X_df["MMRAcquisitionAuctionAveragePrice"].fillna(X_df["MMRAcquisitionAuctionAveragePrice"].median())
        X_df["MMRAcquisitionAuctionCleanPrice"].fillna(X_df["MMRAcquisitionAuctionCleanPrice"].median())
        X_df["MMRAcquisitionRetailAveragePrice"].fillna(X_df["MMRAcquisitionRetailAveragePrice"].median())
        X_df["MMRAcquisitonRetailCleanPrice"].fillna(X_df["MMRAcquisitonRetailCleanPrice"].median())
        X_df.fillna(-1)
        X_df_new = pd.concat(
            [X_df.get(['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'PurchDate', 'WarrantyCost', 'RefId', 'VehYear', 'VehicleAge']),
             pd.get_dummies(X_df.Size, prefix = 'Size', drop_first=True),
             pd.get_dummies(X_df.Auction, prefix='Auction', drop_first=True),
             pd.get_dummies(X_df.Color, prefix='Color', drop_first=True),
             pd.get_dummies(X_df.Transmission, prefix='Transmission', drop_first=True),
             pd.get_dummies(
                 X_df.Nationality, prefix='Nationality', drop_first=True),
             pd.get_dummies(X_df.Model, prefix='Model', drop_first=True),
             pd.get_dummies(X_df.SubModel, prefix='SubModel', drop_first=True),
             pd.get_dummies(X_df.Make, prefix="Make", drop_first=True),
             pd.get_dummies(X_df.WheelType, prefix="WheelType", drop_first=True),
             pd.get_dummies(X_df.TopThreeAmericanName, prefix="TopThreeAmericanName", drop_first=True),
             pd.get_dummies(X_df.VNZIP1, prefix="VNZIP1", drop_first=True),
             ],
            axis=1)
        X_df_new = X_df_new.fillna(-1)
        X_df_new = fix_columns(X_df_new, column_dummies).as_matrix()
        return X_df_new