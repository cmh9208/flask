from random_list import Random_List # 파일로부터 클래스를 불러온다.

class Bubble(object):
    def __init__(self) -> None:
        pass

    def print_bubble(self):
        rl = Random_List() # 생성자를 인스턴스로 만들어줌
        ls = rl.get_random(10, 100, 10)
        print(ls)
        for i in range(len(ls) - 1):
            for j in range(len(ls) - 1):
                if ls[j] > ls[j + 1]:
                    ls[j], ls[j + 1] = ls[j + 1], ls[j]
        print("*" * 30)
        print(ls)

if __name__=="__main__":
    bubble = Bubble()
    bubble.print_bubble()

