import numpy as np
import pandas as pd

def new_fruits_df():
    ls1 = ['제품', '가격', '판매량']
    ls2 = ['사과', '딸기', '수박']
    ls3 = [1800, 1500, 3000]
    ls4 = [24, 38, 13]
    ls5 = [ls2, ls3, ls4]
    df = pd.DataFrame({j : ls5[i] for i, j in enumerate(ls1)})
    print(df)
    '''
    for i, j in enumerate(ls1):
        dc[j] = ls1[i]
        dc[j] = ls5[i]
    df = pd.DataFrame.from_dict(dc)
    # orient='index' 컬럼에 인덱스를(디폴트)
    '''
    print('가격평균: ' + str(df['가격'].mean()))
    print('판매량 평균: ', sum(df['판매량'])/3)
    print(df.drop('판매량', axis=1))
    print(df)

MENUS = ["종료", "과일2D", "숫자2D"]
from string import ascii_lowercase

alphabet_list = list(ascii_lowercase)

def new_number_2d():
    df = pd.DataFrame(np.array([list(range(1, 11)),
                                list(range(11, 21)),
                                list(range(21, 31))]), columns = [alphabet_list[:10]])
    print(df)

if __name__=="__main__":
    while True:
        print(MENUS)
        menu = input("메뉴선택: ")
        if menu == "0":
            print("0번 종료")
            break
        elif menu == "1":
            print("1번 과일2D")
            new_number_2d()
        elif menu == "2":
            print("2번 숫자2D")
        else:
            print("해당메뉴 없음")














'''
from util.common import Common

class Fruits(object):
    def __init__(self, number, name, pay):
        self.number = number
        self.name = name
        self.pay = pay

    def __str__(self):
        astar = "*"*40
        schema = "번호 이름 가격"
        return(f"{schema}\n{self.number}번 과일: {self.name} 가격: {self.pay}원\n{astar}")

    @staticmethod
    def del_fruits(ls, name):
        del ls[[i for i, j in enumerate(ls) if j.name == name][0]]

    @staticmethod
    def print_fruitslist(ls):
        [print(i) for i in ls]

    @staticmethod
    def new_fruits():
        number = int(input("과일 번호: "))
        name = input("과일 이름: ")
        pay = int(input("과일 가격: "))
        return Fruits(number, name, pay)

    @staticmethod
    def main():
        ls = []
        while True:
            menu = Common.print()
            if menu == 0:
                print("### 과일등록 ###")
                fruits = Fruits.new_fruits()
                ls.append(fruits)
            elif menu == 1:
                print("### 과일 목록 출력 ###")
                Fruits.print_fruitslist(ls)
            elif menu == 2:
                print("### 과일 삭제 ###")
                name = input("삭제할 과일: ")
                Fruits.del_fruits(ls, name)
            elif menu == 3:
                print("### 앱 종료 ###")
                break
            else:
                print("잘못된 번호 입니다.")

Fruits.main()
'''

