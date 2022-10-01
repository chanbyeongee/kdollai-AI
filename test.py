


from setup import setup_environ

setup_environ()

from submodules.the_classifier import *


new_model = load_THE_model()

sample = "오늘 동현이가 심하게 때렸어"

prediction = the_predict(new_model, sample)
print(prediction)



# if __name__ == "__main__":
#     from submodules.ner_classifier import *




#     new_model = load_NER_model()

#     sample = "오늘 엄마가 심하게 때렸어"

#     prediction = ner_predict(new_model, sample)
#     print(prediction)