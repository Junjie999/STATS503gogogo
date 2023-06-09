# helper functions

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt
import pandas as pd

# 读数据
import zipfile
def read_data():
    zf = zipfile.ZipFile('../Data/Base.csv.zip') 
    df = pd.read_csv(zf.open('Base.csv'))

    df = df.drop(['device_fraud_count'], axis=1) 

    df = df[df.bank_months_count != -1]
    df = df[df.session_length_in_minutes != -1]
    df = df[df.device_distinct_emails_8w != -1] 

    df=df.reset_index(drop=True)

    print("读好数据啦！")
    return df

# 重新分配label，使得label0的数量是label1数量的4倍
def undersample(df):
    X = df.drop('fraud_bool', axis=1)
    y = df['fraud_bool']

    target_count = pd.Series(y).value_counts()

    target_count_1 = target_count[1]
    target_count_0 = int(target_count_1 * 4)
    sampling_strategy = {0: target_count_0, 1: target_count_1}

    under_sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_train_resampled, y_train_resampled = under_sampler.fit_resample(X, y)

    print("在Under Sampling之后有")
    print("没问题的",y_train_resampled.value_counts()[0],"个")
    print("有问题的",y_train_resampled.value_counts()[1],"个")

    df_resampled = pd.concat([X_train_resampled,y_train_resampled],axis=1)
    return df_resampled

# 给dataframe输出undersampled好的X_train, X_test, y_train, y_test
def under_split(df):
    X = df.drop('fraud_bool',axis=1)
    y = df['fraud_bool']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    df_resampled = undersample(pd.concat([X_train,y_train],axis=1))
    df_resampled = df_resampled.sample(frac=1)

    X_train = df_resampled.drop('fraud_bool',axis=1)
    y_train = df_resampled['fraud_bool']

    return X_train, X_test, y_train, y_test

from imblearn.over_sampling import SMOTE
# 给dataframe输出oversampled好的X_train, X_test, y_train, y_test
def over_split(df):
    X = df.drop('fraud_bool',axis=1)
    y = df['fraud_bool']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    df_resampled = pd.concat([X_train_resampled,y_train_resampled],axis=1)
    df_resampled = df_resampled.sample(frac=1)

    X_train = df_resampled.drop('fraud_bool',axis=1)
    y_train = df_resampled['fraud_bool']

    print("在Over Sampling之后有")
    print("没问题的",y_train.value_counts()[0],"个")
    print("有问题的",y_train.value_counts()[1],"个")

    return X_train, X_test, y_train, y_test

from sklearn.preprocessing import OneHotEncoder
def encode(df_resampled):
    cate_var = ['payment_type','employment_status','housing_status',
                'source','device_os','email_is_free','phone_home_valid',
                'phone_mobile_valid','has_other_cards','foreign_request',
                'keep_alive_session','month']

    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded_data = encoder.fit_transform(df_resampled[cate_var])
    columns = encoder.get_feature_names_out(cate_var)
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=columns)
    df_resampled_encoder = pd.concat([df_resampled.drop(cate_var,axis=1), one_hot_encoded_df], axis=1)

    print("使用了OneHotEncoding，得到：")
    print(df_resampled_encoder.shape)

    return df_resampled_encoder

from sklearn.metrics import roc_curve, auc 
# Compute the True Positive Rate and the False Positive Rate
def auc_related(y_test,y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    roc_auc =  auc(fpr, tpr)
    plt.plot(fpr,tpr,label= 'AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1],'--')  # plot a diagonal line from the lower left to the upper right
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc = 'lower right')
    plt.show()
    return roc_auc