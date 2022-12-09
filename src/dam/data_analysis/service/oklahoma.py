import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import  train_test_split
import pandas as pd


OKLAHOMA_MENUS = ["종료", #0
                  "Rename_ko",#1
                  "Data_spec",#2
                  "Target",#3
                  "Nominal",#5
                  "Ordinal",#6
                  "Inteval",#4
                  "Ratio",#7
                  "Patition",#8
                  "학습",#9
                  "예측"#10
                  ]

oklahoma_meta = {'AGEP':'나이',#in
                 'BDSP':'침실수',
                 'ELEP':'월전기료',
                 'GASP':'월가스비',
                 'HINCP':'가계소득',
                 'NRC':'자녀수',
                 'RMSP':'방수',
                 'VALP':'주택가격',
                 'VALP_B1':'주택가격1',
                 'BLD':'건물타입', #no
                 'MAR':'결혼상태',
                 'HHT':'가구타입',
                 'SCHL':'학업성취수준',
                 'ACCESS':'인터넷접근',
                 'ACR':'부지크기',
                 'COW':'근로형태',
                 'FPARC':'자녀나이',
                 'HHL':'가정내언어',
                 'MV':'입주시기',
                 'R65':'65세이상거주유무',
                 'SCH':'학교재학여부',
                 'SEX':'성별',
                 'LANX':'영어외사용유무',
                 }

oklahoma_menu = {
    "1" : lambda t: t.rename_ko(),
    "2" : lambda t: t.data_spec(),
    "3" : lambda t: t.target_variables(),
    "4" : lambda t: t.nominal_variables(),
    "5" : lambda t: t.ordinal_variables(),
    "6" : lambda t: t.inteval_variables(),
    "7" : lambda t: t.ratio_variables(),
    "8" : lambda t: t.partition(),
    "9" : lambda t: print(" ** No Function ** "),
    "10" : lambda t: print(" ** No Function ** "),
}

class OklahomaService:
    def __init__(self):
        self.c = pd.read_csv('../../../../static/data/dam/crime/oklahoma_comb32.csv')
        self.ci = pd.read_csv('../../../../static/data/dam/crime/oklahoma_comb31-IQR30.csv')
        self.dc = pd.read_csv('../../../../static/data/dam/crime/oklahoma_comb31-IQR30.csv')
        self.my_c = None
    '''
    1.데이터 한글 변환
    '''
    def rename_ko(self):
        self.my_c = self.c.rename(columns= oklahoma_meta)
        self.dc = self.c.rename(columns=oklahoma_meta)
        print(" --- 2.Features ---")
        print(self.my_c.columns)
    '''
    2.스펙보기
    '''
    def data_spec(self):
        c = self.my_c
        print(" --- 1.Shape ---")
        print(c.shape)
        print(" --- 2.Features ---")
        print(c.columns)
        print(" --- 3.Info ---")
        print(c.info())
        print(" --- 4.Case Top1 ---")
        print(c.head(1))
        print(" --- 5.Case Bottom1 ---")
        print(c.tail(3))
        print(" --- 6.Describe ---")
        print(c.describe())
        print(" --- 7.Describe All ---")
        print(c.describe(include='all'))
    '''
    3.타깃변수
    '''
    def target_variables(self):
        print(self.my_c['주택가격'].dtype)
        print(self.my_c['주택가격'].isnull().sum())
        print(self.my_c['주택가격'].value_counts(dropna=False))
        print(self.my_c['주택가격'].skew())
        print(self.my_c['주택가격'].kurtosis)
    '''
    4.명목
    '''
    def nominal_variables(self):
        cols1 = ['나이', '침실수', '월전기료', '월가스비', '가계소득', '자녀수',
                 '방수', '주택가격', '주택가격1']
        c = self.my_c
        c1 = c.drop(cols1, axis = 1) # 구간변수 제외 = 나머지 범주형 변수
        print(c1.shape) #23개 198p
        print(c1['결혼상태'].value_counts(dropna=False))
        print(c1['가구타입'].value_counts(dropna=False))
        print(c1['학업성취수준'].value_counts(dropna=False))
    '''
    5.순서
    '''
    def ordinal_variables(self):
        pass
    '''
    6.구간
    '''
    def inteval_variables(self):
        cols1 = ['나이', '침실수', '월전기료', '월가스비', '가계소득', '자녀수',
                 '방수', '주택가격', '주택가격1']
        print(self.my_c[cols1].describe())
        print(self.my_c[cols1].skew())
        self.my_c.drop('CONP', axis=1, inplace=True)
        c1 = self.my_c['월전기료'] <= 500
        c2 = self.my_c['월가스비'] <= 311
        c3 = self.my_c['가계소득'] <= 320000
        self.my_c1 = self.my_c[c1 & c2 & c3]
        print(self.my_c1.shape)
        # 데이터 추가처리(결측값 제거)
        print(self.ci.shape)
        print(self.ci.dtypes)
        print(self.ci.isna().any()[lambda x: x])  # 결측값 찾기
        print(self.ci.isna().mean().sort_values(ascending=False))  # 결측값 비율 내림차순
        cols = ['COW', 'FPARC', 'LANX', 'SCH', 'SCHL']
        self.ci[cols] = self.ci[cols].fillna(0).astype(np.int64)  # 5개 변수 결측값 0으로 대체
        print(self.ci.shape)
        print(self.ci.columns)
        self.ci.to_csv('./data/oklahoma_2017DC1-all.csv', index=False)
    '''
    7.비율
    '''
    def ratio_variables(self):
        pass
    '''
    8.데이터 분할
    '''
    def patition(self, accuracy=None):
        print(self.dc.shape)
        data = self.dc.drop(['주택가격1'], axis=1)
        target = self.dc['주택가격1']
        print(data.shape)
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.5, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=0)
        model = rf.fit(X_train, y_train)
        pred = model.predict(X_test)
        print("Accuracy on training set:{:.5f}".format(model.score(X_train, y_train)))
        print("Accuracy on test set:{:.5f}".format(accuracy.score(X_train, y_test)))
        







    '''
    9.학습
    '''
    def learning(self):
        pass
    '''
    10.예측
    '''
    def prediction(self):
        pass
