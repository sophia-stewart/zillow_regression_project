# Functions for Wrangling the Zillow Data

import pandas as pd

def get_url(db):
    '''
    This function takes in a database name and returns a url (using the specified 
    database name as well as host, user, and password from env.py) for use in the 
    pandas.read_sql() function.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow():
    '''
    This function first searches for a .csv file containing the zillow data and then 
    reads that file into a dataframe. If zillow.csv is not found, it uses
    the get_url() helper function to access the SQL server and read the query result 
    into a dataframe. It then caches this data into a csv file (zillow.csv).
    This function takes no arguments and returns a dataframe of zillow data.
    '''
    import os
    if os.path.isfile('zillow.csv'):
        zillow = pd.read_csv('zillow.csv', index_col=0)
        return zillow
    else:
        sql = """
        SELECT bedroomcnt as beds,
        bathroomcnt as baths, 
        calculatedfinishedsquarefeet as sq_ft, 
        taxvaluedollarcnt as tax_value, 
        fips 
        FROM properties_2017 
        JOIN predictions_2017 USING(parcelid) 
        WHERE transactiondate <= '2017-12-31' 
        AND propertylandusetypeid = 261;
        """
        zillow = pd.read_sql(sql, get_url('zillow'))
        zillow.to_csv('zillow.csv')
        return zillow

def remove_outliers(df, cols, k):
    '''
    This function takes in a list of column names from a dataframe and a 
    k-value (int) which is used to specify the upper and lower bounds for
    removing outliers. It returns the dataframe with the outliers removed.
    '''
    # make for loop to remove outliers in each column
    for col in cols:
        # get quartiles
        q1, q3 = df[col].quantile([.25, .75])
        # compute iqr
        iqr = q3 - q1
        # get cutoff points for removing outliers
        upper = q3 + k * iqr
        lower = q1 - k * iqr
        # remove outliers
        df = df[(df[col]>lower)&(df[col]<upper)]
    return df

def split_data(df):
    '''
    This function takes in a dataframe and splits it into three dataframes.
    It returns these dataframes in this order: train, validate, test.
    Train makes up 56% of the total observations, validate 24%, and test 20%.
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train, test_size=0.3, random_state=123)
    return train, validate, test

def standard_scale_data(train, validate, test, scaled_cols):
    '''
    This function takes in train, validate, and test dataframes and a list of columns 
    to be scaled. It returns those dataframes with standard-scaled data columns added.
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train[scaled_cols])
    train[[f'{col}_scaled' for col in scaled_cols]] = scaler.transform(train[scaled_cols])
    validate[[f'{col}_scaled' for col in scaled_cols]] = scaler.transform(validate[scaled_cols])
    test[[f'{col}_scaled' for col in scaled_cols]] = scaler.transform(test[scaled_cols])
    return train, validate, test

def label_fips(zillow):
    zillow['fips'] = zillow.fips.astype(int)
    zillow['fips_loc'] = zillow.fips.replace({6037:'Los Angeles, CA',
                       6059:'Orange, CA',
                       6111:'Ventura, CA'})
    return zillow

def wrangle_zillow(out_cols, k, scaled_cols):
    '''
    This function takes in a list of columns from which to remove outliers, a k-value 
    that determines cutoff points for outlier removal, and a list of columns to be scaled.
    It returns three dataframes (train, validate, and test) derived from the zillow data 
    after outliers were removed and columns of scaled data were added.
    '''
    train, validate, test = split_data(remove_outliers(label_fips(acquire_zillow()), out_cols, k))
    train, validate, test = standard_scale_data(train, validate, test, scaled_cols)
    return train, validate, test