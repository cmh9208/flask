import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MENUS = ["종료",
         "메타데이터 출력",
         "poptotal/popasian 변수를 total/asian로 이름변경",
         "전체 인구 대비 아시아 인구 백분율 변수 추가",
         "아시아 인구 백분율 전체 평균을 large/small 로 분류",
         "large/small 빈도표와 빈도막대그래프 작성"]

def my_menu(ls):
      for i, j in enumerate(ls):
          print(f'{i}번:{j}')
      return input('번호선택: ')

def change_poptotal_popasian(midwest): # 변수명 변경
    new_midwest = midwest.rename(columns={'poptotal': 'total'}).\
        rename(columns={'popasian': 'asian'})  # 두개 따로하면 안됨
    return new_midwest

class Midwest:
    def __init__(self):
        self.midwest = pd.read_csv('data/midwest.csv')
        self.m = change_poptotal_popasian(self.midwest)

    def meta_data_print(self): # 1
        m = self.midwest
        print(m.head(3), m.tail(3), m.shape, m.info, m.describe)

    def change_poptotal_popasian(self): # 2
        print(self.m.columns)
    '''
    전체값에서 일부값은 몇 퍼센트? 계산법 공식
    일부값 ÷ 전체값 X 100
    '''
    def asia_Percentage(self): # 3 전체 인구 대비 아시아 인구 백분율 변수 추가
        self.m['Percentage'] = self.m['asian'] / self.m['total'] * 100
        print(self.m['Percentage'])
        self.m['Percentage'].plot.hist()
        plt.savefig('./save/Percentage Graphs')

    def large_small(self): # 4 아시아 인구 백분율 전체 평균을 large/small 로 분류
        self.m['Percentage'] = self.m['asian'] / self.m['total'] * 100
        print(self.m['Percentage'].mean())
        self.m['Percentage'] = np.where(self.m['Percentage'] > 0.48724618343573406, 'large', 'small')
        print(self.m['Percentage'])

    def Frequency_bar_graph(self): # 5 large/small 빈도표와 빈도막대그래프 작성
        self.large_small()
        t = self.m
        large_small_count = t['Percentage'].value_counts()
        large_small_count.plot.bar(rot=0)
        plt.savefig('./save/bar_graph.png')

if __name__ == "__main__":
    t = Midwest()
    while True:
        menu = my_menu(MENUS)
        if menu == '0':
            break
        elif menu == '1':
            t.meta_data_print()
        elif menu == '2':
            t.change_poptotal_popasian()
        elif menu == '3':
            t.asia_Percentage()
        elif menu == '4':
            t.large_small()
        elif menu == '5':
            t.Frequency_bar_graph()
        else:
            print("잘못된 메뉴번호 입니다")





