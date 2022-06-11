import KHUDoll_AIModels as KHU
import json
import warnings

warnings.filterwarnings('ignore')

GeneralCorpus_model, GCTok = KHU.load_general_corpus_model()
NER_model = KHU.load_NER_model()

file_index = 0

while(True):
    sentence = input("수신한 텍스트: ")
    if sentence == "end":
        break;
    name = "김채연"
    data = [sentence] + [name]

    file_data = KHU.To_DataStructure(data, GeneralCorpus_model, NER_model, GCTok)
    print(json.dumps(file_data, ensure_ascii=False, indent="\t"))
    file_name = "\data"+str(file_index)+".json"
    with open('C:\MyProjects\Other Projects\AI ChatBot Project\Jsons'+file_name, 'w', encoding='utf-8') as f:
        json.dump(file_data, f, ensure_ascii=False, indent='\t')
    file_index += 1