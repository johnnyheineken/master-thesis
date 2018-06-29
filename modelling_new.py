#%%
from data_manipulation import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from lifetimes import BetaGeoFitter
from sklearn.model_selection import train_test_split

def print_annotation(ann):
    annotation = f'____   {ann}   ____'
    print('_' * len(annotation))
    print(annotation)
    print('_' * len(annotation))

#%%

orig = pd.read_csv(
    'Data\\transactions.txt',
    sep='|',
    parse_dates=['txn_time']
)
users = pd.read_csv('Data\\users.txt', sep='|')
products = pd.read_csv('Data\\products.txt', sep='|')

n_quarters = 4

time_reference = max(orig.txn_time) - pd.Timedelta(days=365)


def get_groups_RFM(X_orig):
    X = X_orig.copy()
    X['cash_total'] = X['txn_total'] * X['avg_price_txn']
    X['frequency'] = pd.qcut(
        x=X['txn_total'], 
        q=2, 
        duplicates='drop', 
        labels=False)
    X['recency'] = np.abs(pd.qcut(
        x=X['recency_true'],
        q=2,
        duplicates='drop',
        labels=False) - 1)
    X['monetary'] = pd.qcut(
        x=X['cash_total'],
        q=2,
        duplicates='drop',
        labels=False)

    X['groups'] = pd.Categorical(X['recency'].astype(str) +
                X['frequency'].astype(str) + 
                X['monetary'].astype(str)).codes
    return X



def get_subset(X, y, group):
    X2 = X.copy()
    X2['churn'] = y
    X_subset = X2[X2['groups'] == group]
    y_subset = X_subset['churn']
    X_subset = X_subset.drop(['churn', 'frequency', 'recency', 'monetary'], axis=1)
    return X_subset, y_subset

def get_subset_model(X, y, group):
    Xs, ys = get_subset(X, y, group=group)
    rf = RandomForestClassifier(verbose=False,
                                # class_weight={0: 1, 1: 2},
                                bootstrap=True,
                                max_depth=20,
                                max_features='auto',
                                min_samples_leaf=5,
                                min_samples_split=4,
                                n_estimators=100
                                ) \
        .fit(Xs, ys)
    return rf



X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference -
                                    pd.Timedelta(days=365),
                                    verbose=True,
                                    n_quarters=n_quarters)


X_test, y_test = get_X_y_datasets(
    transactions=orig,
    time_reference=time_reference,
    verbose=True,
    n_quarters=n_quarters)
#%%
X_train = get_groups_RFM(X_train)
X_test = get_groups_RFM(X_test)
y_pred = []



y_pred = []

y_test2 = []
for group in range(8):
    print("______________________________________________")
    print("____________________")
    print(f"       {group}")
    print("____________________")
    Xsr_train, ysr_train = get_subset(
        X_train, y_train, group=group)
    Xsr_test, ysr_test = get_subset(
        X_test, y_test, group=group)

    

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=4)]
    learning_rate = [x for x in np.linspace(start=0.1, stop=1, num=4)]
    # Number of features to consider at every split
    # max_features = ['auto']
    # # Maximum number of levels in tree
    # max_depth = [int(x) for x in np.linspace(3, 110, num=4)]
    # max_depth.append(None)
    # # Minimum number of samples required to split a node
    # min_samples_split = [2, 5, 10]
    # # Minimum number of samples required at each leaf node
    # min_samples_leaf = [1, 2, 4]
    # # Method of selecting samples for training each tree
    # bootstrap = [True]
    # random_grid = {'n_estimators': n_estimators,
    #                'max_features': max_features,
    #                'max_depth': max_depth,
    #                'min_samples_split': min_samples_split,
    #                'min_samples_leaf': min_samples_leaf,
    #                'bootstrap': bootstrap}
    random_grid = {'n_estimators': n_estimators, 'learning_rate': learning_rate}
    clf = GridSearchCV(clf, random_grid, verbose=False,
                       n_jobs=3, scoring='f1').fit(Xsr_train, ysr_train)

    # print(clf.best_params_)
    yr_pred = clf.predict(Xsr_test)
    print(confusion_matrix(ysr_test, yr_pred))
    print(classification_report(ysr_test, yr_pred))
    y_test2 += ysr_test.tolist()
    y_pred += yr_pred.tolist()



#%%
print(confusion_matrix(y_test2, y_pred))
print(classification_report(y_test2, y_pred))



clf = AdaBoostClassifier()
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=4)]
learning_rate = [x for x in np.linspace(start=0.1, stop=1, num=4)]

random_grid = {'n_estimators': n_estimators,
                'learning_rate': learning_rate}
clf = GridSearchCV(clf, random_grid, verbose=False,
                    n_jobs=3, scoring='f1').fit(X_train, y_train)

# print(clf.best_params_)
yr_pred = clf.predict(X_test)
print(confusion_matrix(y_test, yr_pred))
print(classification_report(y_test, yr_pred))

X_test['churn'] = y_test2
X_test['pred_8m'] = y_pred
X_test['pred_1m'] = yr_pred

#%%
X_test.to_csv('matrix.csv')

#%%
from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(X_train['txn_total'], X_train['recency_true']/7,
        X_train['T']/7)
print(bgf)

%matplotlib inline
from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)

#%%
from lifetimes.plotting import plot_probability_alive_matrix

f=plot_probability_alive_matrix(bgf)

t=52
X_train['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t, X_train['txn_total'], X_train['recency_true']/7,
    X_train['T']/7)
#%%
from lifetimes.plotting import plot_period_transactions
f = plot_period_transactions(bgf)

#%%
X_train.sort_values('predicted_purchases')
#%%
# X_train.sort_values(by='predicted_purchases').head(5)
from lifetimes.plotting import plot_period_transactions
f = plot_period_transactions(bgf)
#%%
import matplotlib.pyplot as plt

f = plt.figure()
plot_frequency_recency_matrix(bgf)
f.savefig("foo.pdf", bbox_inches='tight')

#%%
X_train.predicted_purchases.describe(percentiles=np.linspace(0,1, num=25)).apply(lambda x: format(x, 'f'))
#%%
X_train['predicted_churn'] = X_train.predicted_purchases < 1
X_train['predicted_churn']




#%%
pd.set_option('display.max_columns', 500)
y_train
#%%
def test_everything(X_train, y_train, X_test, y_test):
    '''
    1) test whether Full AdaBoost model performs better than BG/NBD
    2) test whether AdaBoost model trained on same 
        variables performs better
    3) test Adaboost splitted in 8 RFM groups
         vs AdaBoost at once vs AdaBoost at RFM
    4) alternative test/train split
    '''
    #####################
    ##  FULL ADABOOST  ##
    #####################
    print_annotation('FULL ADABOOST')
    ada = AdaBoostClassifier()
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=4)]
    learning_rate = [x for x in np.linspace(start=0.1, stop=1, num=4)]

    random_grid = {'n_estimators': n_estimators,
                'learning_rate': learning_rate}
    clf = GridSearchCV(ada, random_grid, verbose=False,
                    n_jobs=3, scoring='f1').fit(X_train, y_train)

    # print(clf.best_params_)
    y_pred_full_ada = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred_full_ada))
    print(classification_report(y_test, y_pred_full_ada))


    ########################
    ##  PARTIAL ADABOOST  ##
    ########################
    print_annotation('PARTIAL ADABOOST')
    ada = AdaBoostClassifier()
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=4)]
    learning_rate = [x for x in np.linspace(start=0.1, stop=1, num=4)]

    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate}
    clf = GridSearchCV(ada, random_grid, verbose=False,
                       n_jobs=3, scoring='f1') \
                       .fit(X_train[['txn_total', 'recency_true', 'T']], y_train)
    y_pred_part_ada = clf.predict(X_test[['txn_total', 'recency_true', 'T']])
    print(confusion_matrix(y_test, y_pred_part_ada))
    print(classification_report(y_test, y_pred_part_ada))


    ##################
    ###   BG/NBD   ###
    ##################
    print_annotation('BG/NBD')
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(X_train['txn_total'], X_train['recency_true'] / 7,
        X_train['T'] / 7)
    
    t = 52
    y_pred_bgnbd = bgf \
        .conditional_expected_number_of_purchases_up_to_time(
                t, X_test['txn_total'], 
                X_test['recency_true'] / 7,
                X_test['T'] / 7
            )
    for threshold in np.linspace(0.7, 1.8, 4):
        threshold = round(threshold, 2)
        print('_' * 25)
        print(f"BG/NBD threshold: {threshold}")
        y_pred_bgnbd_tf = y_pred_bgnbd < threshold
        print('churn rate: ' + str(sum(y_pred_bgnbd_tf) / len(y_pred_bgnbd_tf)))
        print(confusion_matrix(y_test, y_pred_bgnbd_tf))
        print(classification_report(y_test, y_pred_bgnbd_tf))


    #############################
    ###   ALTERNATIVE SPLIT   ###
    #############################
    print('_' * 25)
    print('_,-*-,' * 4)
    print('_' * 25)
    print_annotation('FULL ADABOOST alt split')

    X_train_alt, X_test_alt, y_train_alt, y_test_alt = \
        train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    
    
    ada = AdaBoostClassifier()
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=4)]
    learning_rate = [x for x in np.linspace(start=0.1, stop=1, num=4)]

    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate}
    clf = GridSearchCV(ada, random_grid, verbose=False,
                       n_jobs=3, scoring='f1').fit(X_train_alt, y_train_alt)

    # print(clf.best_params_)
    y_pred_ada_alt = clf.predict(X_test_alt)
    print(confusion_matrix(y_test_alt, y_pred_ada_alt))
    print(classification_report(y_test_alt, y_pred_ada_alt))
    ######################################

    print_annotation('PARTIAL ADABOOST alt split')
    ada = AdaBoostClassifier()
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=4)]
    learning_rate = [x for x in np.linspace(start=0.1, stop=1, num=4)]

    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate}
    clf = GridSearchCV(ada, random_grid, verbose=False,
                       n_jobs=3, scoring='f1') \
        .fit(X_train_alt[['txn_total', 'recency_true', 'T']], y_train_alt)
    y_pred_part_ada_alt = clf.predict(
        X_test_alt[['txn_total', 'recency_true', 'T']])
    print(confusion_matrix(y_test_alt, y_pred_part_ada_alt))
    print(classification_report(y_test_alt, y_pred_part_ada_alt))


    ######################################
    print_annotation('BD/NBD alt split')
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(X_train_alt['txn_total'], X_train_alt['recency_true'] / 7,
            X_train_alt['T'] / 7)

    t = 52
    y_pred_bgnbd_ALT = bgf \
        .conditional_expected_number_of_purchases_up_to_time(
            t, X_test_alt['txn_total'],
            X_test_alt['recency_true'] / 7,
            X_test_alt['T'] / 7
        )
    for threshold in np.linspace(0.2, 2.5, 6):
        print('_' * 25)
        print(f"BG/NBD threshold: {threshold}")
        y_pred_bgnbd_tf_alt = y_pred_bgnbd_ALT < threshold
        print('churn rate: ' + str(sum(y_pred_bgnbd_tf_alt) / len(y_pred_bgnbd_tf_alt)))
        print(confusion_matrix(y_test_alt, y_pred_bgnbd_tf_alt))
        print(classification_report(y_test_alt, y_pred_bgnbd_tf_alt))

#%%
test_everything(X_train, y_train, X_test, y_test)

#%%
X_train.columns

