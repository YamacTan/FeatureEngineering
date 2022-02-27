#######################
# Yamac TAN - Data Science Bootcamp - Week 6 - Project 2
#######################

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 15)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# %%

# Veri seti degiskenleri:

# CustomerId Müşteri İd’si
# Gender Cinsiyet
# SeniorCitizen Müşterinin yaşlı olup olmadığı (1, 0)
# Partner Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure Müşterinin şirkette kaldığı ay sayısı
# PhoneService Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges Müşteriden tahsil edilen toplam tutar
# Churn Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


# %%
# Kod okunabilirliğini arttırabilmek amacıyla, kod kapsamında kullanılan tüm fonksiyonlar bu hücrede tanımlanmıştır.

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


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# %%
###############################################
# Görev 1 - Adım 1
###############################################

df_ = pd.read_csv("Odevler/HAFTA_06 FEATURE ENGINEERING/PROJE_II/Telco-Customer-Churn.csv")
df = df_.copy()
df.head(10)
df.isnull().values.any()
df.shape
df.describe().T
# %%
###############################################
# Görev 1 - Adım 2
###############################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Kategorik değişken: Outcome

# %%
###############################################
# Görev 1 - Adım 3
###############################################

for col in num_cols:
    num_summary(df, col, plot=True)

for col in cat_cols:
    cat_summary(df, col)

# %%
###############################################
# Görev 1 - Adım 4
###############################################

df.groupby("Churn").mean()

# %%
###############################################
# Görev 1 - Adım 5
###############################################

for col in num_cols:
    print(col, check_outlier(df, col))
# Numerik degiskenlerde aykırı degerler yoktur.

# %%
###############################################
# Görev 1 - Adım 6
###############################################

missing_values_table(df)
# Dataframede herhangi bir eksik değer bulunmamaktadır.

# %%
###############################################
# Görev 1 - Adım 7
###############################################

#num_cols: tenure and MonthlyCharges
df["tenure"].corr(df["MonthlyCharges"])

# %%
###############################################
# Görev 2 - Adım 1
###############################################

for col in num_cols:
    print(col, check_outlier(df, col))

missing_values_table(df)

# Datasette eksik ya da aykırı gözlem bulunmadığından bir işlem yapmaya gerek yoktur.

# %%
###############################################
# Görev 2 - Adım 2
###############################################

df.loc[(df['tenure'] > 33), "customer_class"] = "loyal"
df.loc[(df['tenure'] <= 33), "customer_class"] = "standard"

# %%
###############################################
# Görev 2 - Adım 3
###############################################
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
for col in binary_cols:
    label_encoder(df, col)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

# %%
###############################################
# Görev 2 - Adım 4
###############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# %%
###############################################
# Görev 2 - Adım 5
###############################################
y = df[["Churn"]]
X = df.drop(["Churn", "TotalCharges", "customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

rf_model = RandomForestClassifier(random_state=1).fit(X_train, np.ravel(y_train))

y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

