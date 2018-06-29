'''
Works in Python 3.6, Windows 10
Made: Jan Hynek

TODO:


'''
import pandas as pd
import numpy as np
import math
from numpy import mean
import warnings



def process_users(users, X):
    '''Make basic processing of users table and merge with table X
    users: raw users table, with 3 dimensions - age, gender, ID_user, pandas.DataFrame
    X: processed table, 
        with one customer on one row and assuming ID_user as an index, pandas.DataFrame
    '''
    users = users.fillna(0)
    users['female'] = users.gender == 'female'
    users['male'] = users.gender == 'male'
    users['no_age'] = users.age == 0
    users = users.drop(['gender'], axis=1)
    users = users.set_index('ID_user')
    X = X.merge(users, how='left', left_index=True, right_index=True)
    return X


def process_products(products, transactions, X):
    '''Find the most common category for given user and append to X.
    products: raw table with ID_product and ID_category, pandas.DataFrame
    transactions: raw transactions table with 5 columns, pandas.DataFrame
    X: processed table, with one customer on one row, assuming ID_user as an index, pandas.DataFrame
    '''
    trans_temp = transactions.copy()
    trans_temp['favorite_category'] = trans_temp \
        .merge(products, how='left', on='ID_product') \
        .ID_category \
        .fillna(0) \
        .astype(int) \
        .astype(str)
    products_users = pd.DataFrame(trans_temp
                                .groupby('ID_user')
                                .favorite_category
                                .agg(lambda x: x.value_counts().index[0]))
    products_users['missing_cat'] = products_users == '0'
    X = X.merge(products_users, how='left',
                left_index=True, right_index=True)
    return X


def feature_extraction(subset_before, n_quarters, users_temp, verbose):    
    if verbose:
        print('Creating features')

    # I take the number of periods I am interested in
    # I had good results with n_quarters=8
    d = subset_before[subset_before.quarter <= n_quarters]
    # number of items bought, totally (in the last year)
    d = d \
        .groupby(d.columns.tolist()) \
        .size() \
        .reset_index() \
        .rename(columns={0: 'item_count'})
    # Cost of transaction obtained from given user (in the last year)
    d = d \
        .assign(overall_price=lambda x: x['price'] * x['item_count'])
    # number of transactions of given user (in the last year)
    d['txn_total'] = d.groupby('ID_user') \
        .ID_txn \
        .transform(len)
    # revenue obtained from given user
    d['revenue_total'] = d.groupby('ID_user') \
        .overall_price \
        .transform(sum)
    # of things bought by given user
    d['things_total'] = d.groupby('ID_user')['item_count'] \
        .transform(sum)

    # Shortcut
    Grouped = d.groupby('ID_user')

    # Helping functions, used in the next part
    def in_work(x):
        m = mean(x)
        return int((9 < m) & (m < 17))

    def in_evening(x): return int(17 <= mean(x))

    # When was the transaction made?
    d['hour_bought'] = [i.hour for i in d.txn_time_x]
    # Was that during usual working hours? (ignoring weekends, though)
    d['in_work'] = Grouped \
        .hour_bought \
        .transform(in_work)
    # Was that in evening?
    d['in_evening'] = Grouped \
        .hour_bought \
        .transform(in_evening)
    # This could provide us with some profile of the customers. Maybe those who buy
    # things while at work are fired afterwards and have to churn ... :)

    # Proxy for socioeconomic status of the customer
    # Is he or she buying expensive things?
    d['avg_price_txn'] = d.revenue_total / d.txn_total
    d['avg_price_thing'] = d.revenue_total / d.things_total

    # Revenue from given customer per quarter
    revenue_per_q = d[['ID_user', 'overall_price', 'quarter']]
    revenue_per_q = revenue_per_q.groupby(['ID_user', 'quarter']).sum()
    revenue_per_q = revenue_per_q.unstack().fillna(0).reset_index()

    # How many transactions has the customer made in given quarter
    transactions_per_q = d[['ID_user', 'item_count', 'quarter']]
    transactions_per_q = transactions_per_q.groupby(
        ['ID_user', 'quarter']).size()
    transactions_per_q = transactions_per_q.unstack().fillna(0).reset_index()

    # Suppressing warnings, as the merge issue several
    # as the tables are multiindex ones, however merging them
    # creates desired structure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verbose:
            print('merging temporary datasets')
        users_temp = users_temp.merge(revenue_per_q, how='left', on='ID_user')
        users_temp = users_temp.merge(transactions_per_q, how='left',
                                      on='ID_user').drop('txn_time', axis=1)
        # But why some columns are not named afterwards,
        # remains mystery for me
        d = d \
            .merge(users_temp, how='left', on='ID_user') \
            .rename(
                columns={_key: ('item_count', _key)
                         for _key in range(n_quarters + 1)}
            ) \
            .drop(['txn_time_y', 'txn_time_x'], axis=1)
        # dropping unneeded columns
    # these columns either have no use (IDs)
    # or they are transaction dependent (hour_bought)
    # or would cause multicollinearity (revenue_total)
    if verbose:
        print('dropping duplicates and unecessary columns')
    X = d.drop(
        ['quarter',
         'item_count',
         'ID_txn',
         'hour_bought',
         'ID_product',
         'price',
         'overall_price',
         'revenue_total'], axis=1) \
        .sort_values('ID_user') \
        .drop_duplicates() \
        .set_index('ID_user')

    # rates of transactions and money spent
    X['txn_rate_month'] = X['txn_total'] / \
                         ((X['recency_true'] + 365) / 30)
    X['price_rate_month'] = (X['txn_total'] *
                             X['avg_price_txn']) / \
                             ((X['recency_true'] + 365) / 30)
    # One more variable - what is the trend?
    # Is the customer buying more or less with respect to most recent quarter?
    X[('trend_revenue', 0)] = (
        X[('overall_price', 1)] - X[('overall_price', 0)]) > 0
    X[('trend_revenue', 1)] = (
        X[('overall_price', 2)] - X[('overall_price', 0)]) > 0
    X[('trend_revenue', 2)] = (
        X[('overall_price', 3)] - X[('overall_price', 0)]) > 0
    return X


def create_churn(users_temp, transactions, verbose):
    ######################
    ### CHURN CREATION ###
    ######################
    # Now, creation of the churn
    #  I am looking at the original dataset
    # I look at the transactions, subset them for future year from the last transaction
    # and if that dataset is empty
    # I append one, otherwise I append zero
    year = pd.Timedelta(days=365)
    if verbose:
        print('Creating churn')
    churn = []
    for i, j in users_temp.itertuples(index=False):
        churn.append(int(transactions[
            (transactions.ID_user == i) &
            (transactions.txn_time < j + year) &
            (transactions.txn_time > j)]
            .empty))
    return churn


def get_subset(X, y, percentile=70, rich=True):
    X2 = X.copy()
    X2['churn'] = y
    # X2['churn']
    X2['total'] = X2.txn_total * X2.avg_price_txn
    perc = np.percentile(X2.total, percentile)
    if rich:
        X_subset = X2[(X2.total > perc)]
    else:
        X_subset = X2[(X2.total <= perc)]
    # X_subset
    y_subset = X_subset.churn
    X_subset = X_subset.drop(['churn', 'total'], axis=1)
    return X_subset, y_subset


def get_X_y_datasets(transactions,
        time_reference="Infer", 
        users=None,
        products=None,
        verbose=False,
        check_time=False,
        ndays_backward = 365, n_quarters=4):
    ''' Creates datasets usable in modelling with all features from given tables
    - time_reference (pandas._libs.tslib.Timestamp), default Infer: 
    \n timepoint, from which is whole dataset calculated
        if left as Infer, function takes timestamp which is one year smaller from maximal time stamp present in users
        dataset
    - transactions (pandas.core.frame.DataFrame): 
    \n raw dataset with transactions (5 columns)
        requires that column 'txn_time' is parsed as a date
    - users (optional - pandas.core.frame.DataFrame): 
    \n raw users table
    - products (optional - pandas.core.frame.DataFrame): 
    \n raw products table
    - verbose (bool), default False: 
    \n should function print reports?
    - check_time (bool), default False: 
    \n should function check, whether time_reference is at least one year?
    - ndays_backward (optional - integer), default 365: 
    \n how many days before timereference should be used to determine users present in the dataset?
    - n_quarters (optional - int), default 8:
    \n how many quarters should be used to create temporal variables?
    \n \n For ID_user, which are available in last year from time_reference,
    function finds date of the last ID_tnx, 
    and look at all transactions in the year before 
    and make quarterly variables, along with several constant ones.
    If no transaction is in the year after the last txn, 
    function appends churn = 1, otherwise 0'''
    # it is tedious to write year over an over
    year = pd.Timedelta(days=365)

    ######################
    ### ERROR CHECKING ###
    ######################

    if type(transactions.txn_time[0]) is not pd._libs.tslib.Timestamp:
        try:
            transactions['txn_time'] = pd.to_datetime(transactions['txn_time'])
            warnings.warn('\n "txn_time" of "transactions" matrix converted to datetime')
        except:
            raise TypeError(
                "\n'txn_time' column of 'transactions' matrix is not pandas._libs.tslib.Timestamp")

    # if transactions.shape[1] != 5:
    #     raise ValueError("Width of transaction matrix is not 5")
    
    if users is not None:
        if users.shape[1] != 3:
            raise ValueError("Width of user matrix is not 3")

    if products is not None:
        if products.shape[1] != 2:
            raise ValueError("Width of user matrix is not 2")
    
    if (time_reference is not "Infer") and \
        (type(time_reference) is not pd._libs.tslib.Timestamp):
            warnings.warn(
                'time_reference is not pandas._libs.tslib.Timestamp or "Infer". Inferring date.')
            time_reference = "Infer"
    if time_reference is "Infer":
        time_reference = (max(transactions.txn_time) - year - pd.Timedelta(days=1))

    if (time_reference >= (max(transactions.txn_time)) - year) and check_time:
        warnings.warn('\n Time reference is not at least one year from the most current date in dataset, \
                        \n function returns X only. \
                        \n You can override this behaviour by setting check_time=False, \
                        \n but target variable might be corrupted')
    
    max_date = max(transactions['txn_time'])
    transactions_copy = transactions.copy()
    transactions_copy['T'] = (transactions \
        .groupby('ID_user') \
        .txn_time \
        .transform(lambda x: max_date - min(x)).dt.days) / 7
    
    
    transactions_copy['recency_true'] = (transactions \
        .groupby('ID_user') \
        .txn_time \
        .transform(lambda x: max(x) - min(x)).dt.days) / 7


    transactions_copy
    # unique IDs before time reference - I am looking at all
    # user Ids with any transaction in the last year
    if verbose:
        print('dividing datasets')
    unq_IDs_b4_timeref = transactions_copy[
            (transactions_copy.txn_time < time_reference) & \
            (transactions_copy.txn_time > (time_reference - pd.Timedelta(days=ndays_backward)))
        ] \
        .ID_user \
        .unique()
    
    if verbose:
        print('number of active users in year before time reference: ' + \
            str(len(unq_IDs_b4_timeref)))
    if len(unq_IDs_b4_timeref) == 0:
        raise ValueError("no data before time reference")
    # For feature engeneering, I decided to take only only data 
    # which are already available at the given moment. 
    # This will ensure that this function is applicable for any dataset
    # I am assuming, that only these informations are relevant (strong assumption)

    subset_before = transactions_copy[
        transactions_copy.ID_user.isin(unq_IDs_b4_timeref) & \
        (transactions_copy.txn_time < time_reference)
        ]


    # Unit testing. I am checking for some errors
    def unit_test_sb(subset_before):
        if (max(subset_before.txn_time) > time_reference) or \
            len(subset_before.ID_user.unique()) != len(unq_IDs_b4_timeref):
            raise UserWarning('Unit test failed')

    unit_test_sb(subset_before)


    # When was the last transaction for given user?
    # This will be the ending point for every user - from this time, I am 
    # calculating his churn, and obtaining his features
    users_temp = subset_before \
        .groupby('ID_user') \
        .txn_time.agg(max) \
        .reset_index() \
        .sort_values(by='txn_time', ascending=False)
    
    # I classify the quarters of each transaction in the dataset
    # This is the way I am preserving some temporal dimensions of the data
    # But in this way I will change it to classification task
    if verbose:
        print('Classifying quarters')
    def classify_quarter(date, last_date):
        months = last_date.to_period('M') - date.to_period('M')
        return(math.floor(months/3))

    subset_before = subset_before.merge(users_temp, how='left', on='ID_user')
    subset_before['quarter'] = [
        classify_quarter(
            subset_before.iloc[i, ].txn_time_x,
            subset_before.iloc[i, ].txn_time_y)
        for i in range(subset_before.shape[0])]

    ############################
    ##### Creating churn  ######
    ############################

    churn = create_churn(users_temp, transactions_copy, verbose)

    users_temp['churn'] = churn
    users_temp['last_txn_days'] = [i.days for i in (time_reference - users_temp.txn_time)]

    ###############################
    #### Feature extraction  ######
    ###############################

    X = feature_extraction(subset_before, n_quarters, users_temp, verbose)

    ###############################
    ### Processing other tables ###
    ###############################
    # adding columns from optional tables:
    if users is not None:
        if verbose:
            print('processing users')
        X = process_users(users, X)
    if products is not None:
        if verbose:
            print('processing products')
        X = process_products(products, transactions_copy, X)



    # Creating y and X.
    y = X.churn
    X = X.drop(['churn'], axis=1)
    if verbose:
        print('churn rate: ' + str(mean(y)))
        print('DONE')
    
    if (time_reference > (max(transactions_copy.txn_time)) - year) and check_time:
        return X
    else:
        return X, y



