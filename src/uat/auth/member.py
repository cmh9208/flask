'''
아이디, 비밀번호, 이름을 받아서
회원가입, 목록, 탈퇴하는 프로그램을 배발하시오.
'''
from src.cmm.com.service import Common

class Member(object):
    def __init__(self, id, pw, name):
        self.id = id
        self.pw = pw
        self.name = name

    def __str__(self):
        schema = "아이디, 비밀번호, 이름"
        astar = "*"*40
        return (f"{schema}\n{self.id},{self.pw},{self.name}\n{astar}")

    @staticmethod
    def delete_member(ls, name):
        del ls[[i for i,j in enumerate(ls) if j.name == name][0]]

    @staticmethod
    def print_member(ls):
        [print(i) for i in ls]

    @staticmethod
    def get_member():
        id = input("아이디: ")
        pw = input("비밀번호: ")
        name = input("이름: ")
        return Member(id, pw, name)

    @staticmethod
    def main():
        ls = []
        while True:
            menu = Common.print()
            if menu == 0:
                print(" ### 가입 ### ")
                member = Member.get_member()
                ls.append(member)
            elif menu == 1:
                print(" ### 목록 ### ")
                Member.print_member(ls)
            elif menu == 2:
                print(" ### 탈퇴 ### ")
                name = input("삭제할 이름: ")
                Member.delete_member(ls, name)
            elif menu == 3:
                print(" ### 종료 ### ")
                break
            else: print("잘못된 메뉴버호 입니다.")

Member.main()