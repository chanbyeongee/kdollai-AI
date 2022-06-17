from pykospacing import Spacing

class MakeSpace:
    def __init__(self):
        self.spacing_mod = Spacing()

    def convert(self,text):
        return self.spacing_mod(text)

if __name__ == "__main__":
    test = MakeSpace()
    msg = "안뇽? 나는 권도완이얌, 그리고 탐정일지도."
    converted_msg = test.convert(msg)
    print(converted_msg)