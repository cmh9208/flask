import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
Data columns (total 12 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Unnamed: 0    234 non-null    int64  
 1   manufacturer : 회사  234 non-null    object 
 2   model : 모델        234 non-null    object 
 3   displ : 배기량         234 non-null    float64
 4   year : 연식         234 non-null    int64  
 5   cyl : 실린더          234 non-null    int64  
 6   trans : 차축        234 non-null    object 
 7   drv : 오토          234 non-null    object 
 8   cty : 시내연비          234 non-null    int64  
 9   hwy : 시외연비          234 non-null    int64  
 10  fl : 연료            234 non-null    object 
 11  class : 차종         234 non-null    object 
dtypes: float64(1), int64(5), object(6)
'''
my_meta = {
    "manufacturer": "회사",
    "model": "모델",
    "displ": "배기량",
    "year": "연식",
    "cyl": "실린더",
    "trans": "차축",
    "drv": "오토",
    "cty": "시내연비",
    "hwy": "시외연비",
    "fl": "연료",
    "class": "차종"
}
def my_menu(ls):
    print(f'type is {type(ls)}')
    for i, j in enumerate(ls):
        print(f"{i}. {j}")
    return input('메뉴선택: ')

MENUS = ["종료",
         "mpg 앞부분 확인",
         "mpg 뒷부분 확인",
         "행,열 출력",
         "데이터 속성 확인",
         "요약 통계량 출력",
         "문자 변수 요약 통계량 함께 출력",
         # mpg 129페이지
         "manufacturer 를 company 로 변경",
         "test 변수 생성",
         # cty 와 hwy 변수를 머지(merge)하여 total 
         # 변수 생성하고 20이상이면 pass 미만이면 fail 저장
         "test 빈도표 만들기",
         "test 빈도 막대 그래프 그리기",
         # mpg 144페이지 문제
         "displ(배기량)이 4이하와 5이상 자동차의 hwy(고속도로 연비) 비교",
         "아우디와 토요타 중 도시연비(cty) 평균이 높은 회사 검색",
         "쉐보레, 포드, 혼다 데이터 출력과 hwy 전체 평균",
         # mpg 150페이지 문제
         # 메타데이터가 category, cty 데이터는 해당 raw 데이터인 객체생성
         # 후 다음 문제 풀이
         "suv / 컴팩 자동차 중 어떤 자동차의 도시연비 평균이 더 높은가?",
         # mpg 153페이지 문제
         "아우디차에서 고속도로 연비 1~5위 출력하시오",
         # mpg 158페이지 문제
         "평균연비가 가장 높은 자동차 1~3위 출력하시오",
         # mpg 166페이지 문제
         "차종 별 시내연비 평균을 구하시오",
         "차종별 시내연비 평균이 높은 순위대로 정렬하시오",
         "시외연비 평균이 가장놓은 회사 1~3위를 출력하시오",
         "회사별 compact 차종이 많은 순서대로 출력 하시오",
         # mpg 185페이지 문제
         "오토, 시외연비 결측치 개수를 출력 하시오",
         "시외연비 결측치 제거, 구동 방식별 시외연비 평균 비교"
         ]
class MpgService:
    def __init__(self):
        self.mpg = pd.read_csv('data/mpg.csv')
        self.my_mpg = self.mpg.rename(columns=my_meta)
        self.count_test = None
        self.new_category_mpg = None
        self.nan_mpg = self.my_mpg.copy()
        self.nan_mpg.loc[[64, 123, 130, 152, 211], "시외연비"] = np.nan

    def head(self):
        print(self.mpg.head(5))

    def tail(self):
        print(self.mpg.tail(5))

    def shape(self):
        print(self.mpg.shape)

    def info(self):
        print(self.mpg.info())

    def describe(self):
        print(self.mpg.describe())

    def describe_include(self):
        print(self.mpg.describe(include ='all'))

    def manufacturer_to_company(self): # 7
        self.mpg = self.mpg.rename(columns = {'manufacturer' : 'company'})
        print(self.mpg)

    def mytest_variable(self): # 8 변수 생성하고 20이상이면 pass 미만이면 fail 저장
        t = self.my_mpg
        t['총연비'] = (t['시내연비'] + t['시외연비'])/2
        t['연비테스트'] = np.where(t['총연비']>=20, 'pass', 'fail')
        print(t.columns)
        print(t.head())

    def mytest_frequency(self): # 9 빈도표
        self.mytest_variable()
        t = self.my_mpg
        self.count_test = t['연비테스트'].value_counts()
        print(self.count_test)

    def draw_freq_bar_graph(self): # 그래프
        self.mytest_frequency()
        self.count_test.plot.bar(rot=0)
        plt.savefig('./save/draw_freq_bar_graph.png')

    def compare_displ_and_hwy(self): # displ(배기량)이 4이하와 5이상 자동차의 hwy(고속도로 연비) 비교
        t = self.my_mpg
        a = t.query('배기량<= 4')
        b = t.query('배기량>= 5')
        print('배기량 4이하 평균: ', a['시외연비'].mean())
        print('배기량 5이하 평균: ', b['시외연비'].mean())

    def search_higher_cty(self): # 아우디와 토요타 중 도시연비(cty) 평균이 높은 회사 검색
        t = self.my_mpg
        audi = t.query("회사 == 'audi'")
        toyota = t.query("회사 == 'toyota'")
        print('아우디 평균: ', audi['시내연비'].mean())
        print('도요타 평균: ', toyota['시내연비'].mean())

    def find_hwy_average(self): # 쉐보레, 포드, 혼다 데이터 출력과 hwy 전체 평균
        t = self.my_mpg
        che_fo_hon = t.query("회사 in ['chevrolet', 'ford', 'honda']")
        print('쉐보레, 포드, 혼다 시외연비 평균 : ', che_fo_hon['시외연비'].mean())

    def which_higher_between_suv_compact(self): # 14.suv / 컴팩 자동차 중 어떤 자동차의 도시연비 평균이 더 높은가?
        category_cty = self.my_mpg[['차종', '시내연비']]
        print('suv 시내연비', category_cty.query("차종 == 'suv'")['시내연비'].mean())
        print('compact 시내연비',category_cty.query("차종 == 'compact'")['시내연비'].mean())

    def search_hwy_in_audi_top5(self): # 15.아우디차에서 고속도로 연비 1~5위 출력하시오
        audi = self.my_mpg.query("회사 == 'audi'")
        audi_hwy_ranking = audi.sort_values('시외연비', ascending = False) # 시외연비를 내림차순으로
        print(audi_hwy_ranking.head(5))

    def search_average_mileage_top3(self): # 16.평균연비가 가장 높은 자동차 1~3위 출력하시오
        t = self.my_mpg
        add_avg_mpg = t.assign(평균연비 = lambda x: x['시내연비'] + x['시외연비'] / 2)
        add_avg_mpg_ranking = add_avg_mpg.sort_values('평균연비', ascending = False)
        print(add_avg_mpg_ranking.head(3))

    # 166p
    def category_cty_avg(self): # 17.차종 별 시내연비 평균을 구하시오
        self.new_category_mpg = self.my_mpg.groupby('차종').agg(시내연비평균 = ('시내연비', 'mean') ) # 차종별 분리, 파생변수 생성
        print(self.new_category_mpg)

    def category_cty_avg_ranking(self): # 18.차종별 시내연비 평균이 높은 순위대로 정렬하시오
        self.category_cty_avg()
        cty_ranking = self.new_category_mpg.sort_values('시내연비평균', ascending = False)
        print(f"{'*'*30} \n 시내연비 평균 랭킹")
        print(cty_ranking)

    def manufacturer_hwy_ranking(self): # 19.시외연비 평균이 가장놓은 회사 1~3위를 출력하시오
        manufacturer_hwy_mpg = self.my_mpg.groupby('회사').agg(시외연비평균 = ('시외연비', 'mean') )
        manufacturer_hwy_ranking = manufacturer_hwy_mpg.sort_values('시외연비평균', ascending = False)
        print(manufacturer_hwy_ranking.head(3))

    def manufacturer_compact_counts(self): # 20.회사별 compact 차종이 많은 순서대로 출력 하시오
        compact_counts_mpg = self.my_mpg.query("차종 == 'compact'").value_counts('회사')
        print(compact_counts_mpg)
    # 185p
    def drv_hwy_nan_counts(self): # 21
        print(self.nan_mpg[['오토', '시외연비']].isna().sum())

    def hwy_nan_Removal_drv_avg(self): # 22
        print(self.nan_mpg.dropna(subset = ['시외연비']).groupby('오토').agg(시외연비평균 = ('시외연비','mean')))

if __name__ == '__main__':
    t = MpgService()
    while True:
        menu = my_menu(MENUS)
        if menu == '0':
            print("종료")
            break
        elif menu == '1':
            print("mpg 앞부분 확인")
            t.head()
        elif menu == '2':
            print("mpg 뒷부분 확인")
            t.tail()
        elif menu == '3':
            print("행,열 출력")
            t.shape()
        elif menu == '4':
            print("데이터 속성 확인")
            t.info()
        elif menu == '5':
            print("요약 통계량 출력")
            t.describe()
        elif menu == '6':
            print("문자 변수 요약 통계량 함께 출력")
            t.describe_include()
        elif menu == '7':
            print("manufacturer 를 company 로 변경")
            t.manufacturer_to_company()
        elif menu == '8':
            print("test 변수 생성")
            # cty 와 hwy 변수를 머지(merge)하여 total
            # 변수 생성하고 20이상이면 pass 미만이면 fail 저장
            t.mytest_variable()
        elif menu == '9':
            print("test 빈도표 만들기")
            t.mytest_frequency()
        elif menu == '10':
            print("test 빈도 막대 그래프 그리기")
            t.draw_freq_bar_graph()
        elif menu == '11':
            # mpg 144페이지 문제
            print("displ(배기량)이 4이하와 5이상 자동차의 hwy(고속도로 연비) 비교")
            t.compare_displ_and_hwy()
        elif menu == '12':
            print("아우디와 토요타 중 도시연비(cty) 평균이 높은 회사 검색")
            t.search_higher_cty()
        elif menu == '13':
            print("쉐보레, 포드, 혼다 데이터 출력과 hwy 전체 평균")
            t.find_hwy_average()
        elif menu == '14':
            # mpg 150페이지 문제
            # 메타데이터가 category, cty 데이터는 해당 raw 데이터인 객체생성
            # 후 다음 문제 풀이
            print("suv / 컴팩 자동차 중 어떤 자동차의 도시연비 평균이 더 높은가?")
            t.which_higher_between_suv_compact()
        elif menu == '15':
            print("아우디차에서 고속도로 연비 1~5위 출력하시오")
            t.search_hwy_in_audi_top5()
        elif menu == '16':
            print("평균연비가 가장 높은 자동차 1~3위 출력하시오")
            t.search_average_mileage_top3()

        elif menu == '17':
            print("차종별 시내연비 평균을 구하시오")
            t.category_cty_avg()
        elif menu == '18':
            print("차종별 시내연비 평균이 높은 순위대로 정렬하시오")
            t.category_cty_avg_ranking()
        elif menu == '19':
            print("시외연비 평균이 가장놓은 회사 1~3위를 출력하시오")
            t.manufacturer_hwy_ranking()
        elif menu == '20':
            print("회사별 compact 차종이 많은 순서대로 출력 하시오")
            t.manufacturer_compact_counts()

        elif menu == '21':
            print("오토, 시외연비 결측치 개수를 출력 하시오")
            t.drv_hwy_nan_counts()
        elif menu == '22':
            print("시외연비 결측치 제거, 구동 방식별 시외연비 평균 비교")
            t.hwy_nan_Removal_drv_avg()
        else:
            print("잘못된 번호입니다")

