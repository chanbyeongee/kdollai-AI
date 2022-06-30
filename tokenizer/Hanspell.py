from hanspell import spell_checker

class Hanspell:
    def __init__(self):
        self.checker = spell_checker

    def check(self,text):
        return self.checker.check(text).checked

if __name__ == "__main__":
    my_text = "안뇽? 나는 권도완이얌, 그리고 탐정일지도."
    test = Hanspell()
    checked_pars = test.check(my_text)
    print(checked_pars)
