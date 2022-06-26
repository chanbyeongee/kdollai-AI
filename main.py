import json
import warnings
from models.transformers.bertmodels.AImodels import AIModel

warnings.filterwarnings('ignore')

main_model = AIModel()

file_index = 0

while(True):
    sentence = input("수신한 텍스트: ")
    if sentence == "end":
        break;
    name = "김채연"
    data = [sentence] + [name]

    file_data = main_model.To_DataStructure(data)
    print(json.dumps(file_data, ensure_ascii=False, indent="\t"))
    file_name = "\data"+str(file_index)+".json"
    with open('C:\MyProjects\Other Projects\AI ChatBot Project\Jsons'+file_name, 'w', encoding='utf-8') as f:
        json.dump(file_data, f, ensure_ascii=False, indent='\t')
    file_index += 1