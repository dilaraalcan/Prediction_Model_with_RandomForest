###################################################
# PROJECT: SALARY PREDICTİON WITH MACHINE LEARNING
###################################################

# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör



###################################################
# GÖREV: Veri ön işleme ve özellik mühendisliği tekniklerini kullanarak maaş tahmin modeli geliştiriniz.
###################################################


# gerekli kütüphaneler ve importlar

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns

import warnings
import pandas as pd
import missingno as msno
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Tum Base Modeller
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from pandas.core.common import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)



df = pd.read_csv("/Users/dlaraalcan/Desktop/DSMLBC/hitters(2).csv")
df.head()

def load():
    data = pd.read_csv("/Users/dlaraalcan/Desktop/DSMLBC/hitters(2).csv")
    return data


hitters = load()
hitters.head()
hitters.columns #değişkeler

hitters.shape

# bagımlı değişken: salary
# bagımlı değişken analizi

df["Salary"].describe()
sns.distplot(df.Salary)
plt.show()

sns.boxplot(df["Salary"])
plt.show()

# Kategorik, numerik ve kategorik olup kardinal olan değişkenleri verilen thresholda göre bulma.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(hitters)

cat_cols #kategorik değişkenler

# Kategorik değişkenlerin analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# Rare Encoding: target'a baglı kategorik değişkenlerin oranları
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(hitters,"Salary",cat_cols) # oranlar birbirine yakın oldugundan birleştirme işlemine gerek duyulmaz
# rare degişken olusturmaya gerek yoktur.

# aykırı gözlem analizi
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(hitters, col))
# aykırı gözlem bulunmuyor


#eksik gözlem analizi
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(hitters) # salary değişkeninde 59 eksik deger bulunuyor. salary: bagımlı değişken


# korelasyon analizi
hitters. corr()

# bagımlı değişken ile diger değişkenlerin korelasyonu:
def target_correlation_matrix(dataframe, corr_th=0.50, target="Salary"):
    """
    Bağımlı değişken ile verilen threshold değerinin üzerindeki korelasyona sahip değişkenleri getirir.
    :param dataframe:
    :param corr_th: eşik değeri
    :param target:  bağımlı değişken ismi
    :return:
    """
    corr = dataframe.corr()
    corr_th = corr_th
    try:
        filter = np.abs(corr[target]) > corr_th
        corr_features = corr.columns[filter].tolist()
        sns.clustermap(dataframe[corr_features].corr(), annot=True, fmt=".2f")
        plt.show()
        return corr_features
    except:
        print("Yüksek threshold değeri, corr_th değerinizi düşürün!")


target_correlation_matrix(hitters, corr_th=0.50, target="Salary")

# veri ön işleme
hitters["Hit_class"] = pd.qcut(hitters['Hits'], 4, labels=['D','C','B','A'])
hitters['NEW_HitRatio'] = hitters['Hits'] / hitters['AtBat']
hitters['NEW_RunRatio'] = hitters['HmRun'] / hitters['Runs']
hitters['NEW_CHitRatio'] = hitters['CHits'] / hitters['CAtBat']
hitters['NEW_CRunRatio'] = hitters['CHmRun'] / hitters['CRuns']

hitters['NEW_Avg_AtBat'] = hitters['CAtBat'] / hitters['Years']
hitters['NEW_Avg_Hits'] = hitters['CHits'] / hitters['Years']
hitters['NEW_Avg_HmRun'] = hitters['CHmRun'] / hitters['Years']
hitters['NEW_Avg_Runs'] = hitters['CRuns'] / hitters['Years']
hitters['NEW_Avg_RBI'] = hitters['CRBI'] / hitters['Years']
hitters['NEW_Avg_Walks'] = hitters['CWalks'] / hitters['Years']

hitters.loc[(hitters['Years'] <= 5), 'Exp'] = 'Fresh'
hitters.loc[(hitters['Years'] > 5) & (hitters['Years'] <= 10), 'Exp'] = 'Starter'
hitters.loc[(hitters['Years'] > 10) & (hitters['Years'] <= 15), 'Exp'] = 'Average'
hitters.loc[(hitters['Years'] > 15) & (hitters['Years'] <= 20), 'Exp'] = 'Experienced'
hitters.loc[(hitters['Years'] > 20), 'Exp'] = 'Veteran'

cat_cols, num_cols, cat_but_car = grab_col_names(hitters)
hitters.head()


# one hot encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


hitters = one_hot_encoder(hitters, cat_cols, drop_first=True)
hitters.head()

# bagımlı değişkendeki eksik değerleri ayıralım.
hitters_null = hitters[hitters["Salary"].isnull()]
# eksik degerleri silelim.
hitters.dropna(inplace=True)
hitters['Salary'].isnull().sum() # eksik deger kalmadı


# MODEL KURALIM
y = hitters['Salary'] # bagımlı değişken
X = hitters.drop("Salary", axis=1) # bagımlı değişken harici diger değişkenler


# Hold- out: verisetini train ve test olarak ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# base models
def all_models(X, y, test_size=0.2, random_state=12345, classification = True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 12345)
    all_models = []

    if classification :
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df

all_models = all_models(X, y, test_size=0.2, random_state=46, classification=False)
# train hataları cok düşük olmasına ragmen test hataları asırı yüksek
# bu demektir ki aşırı öğrenme gercekleşmiş.


# RANDOM FORESTS MODEL TUNING
rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

best_params = {'max_depth': 5,
               'max_features': 6,
               'min_samples_split': 12,
               'n_estimators': 80}

# RANDOM FORESTS TUNED MODEL
rf_tuned = RandomForestRegressor(max_depth=5, max_features=6, n_estimators=80,
                                 min_samples_split=12, random_state=42).fit(X_train, y_train)

# TUNED MODEL TRAIN HATASI
y_pred = rf_tuned.predict(X_train)

print("RF Tuned Model Train RMSE:", np.sqrt(mean_squared_error(y_train, y_pred))) # 193

# TEST HATASI
y_pred = rf_tuned.predict(X_test)
print("RF Tuned Model Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred))) #181



# SALARYDEKİ NA DEĞERLERİN TAHMİNİ İLE BİRLİKTE MODEL KURMA

# NA Olan Salary Değerlerini kurduğumuz model ile test edip, yeni bir model kuralım.

hitters_null.head()
hitters_null.drop("Salary", axis=1, inplace=True)

salary_pred = rf_tuned.predict(hitters_null)
#tahminlenen salary degerleri
salary_pred[0:5]

hitters_null['Salary'] = salary_pred
# tahmin edilen degerleri ekliyoruz
hitters_final = pd.concat([hitters, hitters_null])

# yeniden model kuruyoruz
y = hitters_final["Salary"]
X = hitters_final.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


rf_final_model = RandomForestRegressor(max_depth=5, max_features=6, n_estimators=80, min_samples_split=12,
                                       random_state=42).fit(X_train, y_train)

y_pred = rf_final_model.predict(X_test)
y_pred2 = rf_final_model.predict(X_train)

print("ALL DATA TRAIN RMSE:", np.sqrt(mean_squared_error(y_train, y_pred2))) # 174
print("ALL DATA TEST RMSE:", np.sqrt(mean_squared_error(y_test, y_pred))) # 177
