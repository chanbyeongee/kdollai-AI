from setup import setup_environ
setup_environ()





if __name__ == "__main__":
    from submodules.ner_classifier import *




    new_model = load_NER_model()

    sample = "오늘 엄마가 심하게 때렸어"

    prediction = ner_predict(new_model, sample)
    print(prediction)