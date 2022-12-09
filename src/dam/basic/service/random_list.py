import random

class Random_List(object):
    def __init__(self) -> None:
        pass

    def get_random(self, start, end, count):
        return random.sample(range(start,end), k = count)

    def set_bubble(self):
        print(self.get_random(10, 100, 10))
'''      
    @staticmethod
    def main():
        rl = Rl()
        rl.set_bubble()

Rl.main()
'''
if __name__=="__main__":
    rl = Random_List()
    rl.set_bubble()