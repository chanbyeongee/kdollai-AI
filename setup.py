import os, json
from zipfile import ZipFile # 모델 파일이 너무 많아서 zip 파일을 다운받고 압축 해제합니다.

def setup_environ():
    this_dir, this_filename = os.path.split(__file__)

    os.environ['CHATBOT_ROOT'] = this_dir

    print("Environment Variable Set Successfully. root: %s" % (os.environ['CHATBOT_ROOT']))

def download_weights():

    print("Check each weights version and update if they have to.")

    # 실제 구글 드라이브 주소 : https://drive.google.com/drive/u/0/folders/1M0t0ngQO-TdjeRYoS69C4ZiAqzbN2fIV
    config_url = 'https://drive.google.com/u/0/uc?id=1IrqwsC3TmrisU4KgEo_lyJOVJJaFp_0D&export=download'

    import gdown
    this_dir = os.environ['CHATBOT_ROOT']
    Emo_version = NER_version = TOPIC_version = GD_version = "1.0.0"

    if os.path.isfile(this_dir+"/resources/config.json"):
        with open(this_dir+"/resources/config.json",'r') as f :
            loaded = json.load(f)
            Emo_version = loaded["EMO-weights-version"]
            NER_version = loaded["NER-weights-version"]
            TOPIC_version = loaded["TOPIC-weights-version"]
            GD_version = loaded["GD-weights-version"]

    output = this_dir.replace("\\", "/") + "/resources/config.json"
    gdown.download(config_url, output, quiet=False)

    with open(this_dir + "/resources/config.json", 'r') as f:
        loaded = json.load(f)
        Emo_flag = not loaded["EMO-weights-version"] == Emo_version
        NER_flag = not loaded["NER-weights-version"] == NER_version
        TOPIC_flag = not loaded["TOPIC-weights-version"] == TOPIC_version
        GD_flag = not loaded["GD-weights-version"] == GD_version

    weight_path = this_dir.replace("\\", "/") + "/resources/weights"

###############################################################################################

    if not os.path.exists(weight_path + "/Emo_weights"):
        os.makedirs(weight_path + "/Emo_weights")

    if not os.path.isfile(weight_path + "/Emo_weights/Emo_weights.index") or Emo_flag:
        print("Downloading Emo pretrained index...")
        output = weight_path + "/Emo_weights/Emo_weights.index"
        gdown.download(loaded["EMO-index-url"], output, quiet=False)

    if not os.path.isfile(weight_path + "/Emo_weights/Emo_weights.data-00000-of-00001") or Emo_flag:
        print("Downloading Emo pretrained weights...")
        output = weight_path + "/Emo_weights/Emo_weights.data-00000-of-00001"
        gdown.download(loaded["EMO-data-url"], output, quiet=False)


###############################################################################################

    if not os.path.exists(weight_path + "/NER_weights"):
        os.makedirs(weight_path + "/NER_weights")

    if not os.path.isfile(weight_path + "/NER_weights/NER_weights.index") or NER_flag:
        print("Downloading NER pretrained index...")
        output = weight_path + "/NER_weights/NER_weights.index"
        gdown.download(loaded["NER-index-url"], output, quiet=False)

    if not os.path.isfile(weight_path + "/NER_weights/NER_weights.data-00000-of-00001") or NER_flag:
        print("Downloading NER pretrained weights...")
        output = weight_path + "/NER_weights/NER_weights.data-00000-of-00001"
        gdown.download(loaded["NER-data-url"], output, quiet=False)

###############################################################################################

    if not os.path.exists(weight_path + "/GeneralDialog_weights"):
        os.makedirs(weight_path + "/GeneralDialog_weights")

    if not os.path.isfile(weight_path + "/GeneralDialog_weights/General_weights.h5") or GD_flag:
        print("Downloading Transformer pretrained index...")
        output = weight_path + "/GeneralDialog_weights/General_weights.h5"
        gdown.download(loaded["GD-h5-url"], output, quiet=False)

###############################################################################################

    if not os.path.exists(weight_path + "/Topic_weights"):
        os.makedirs(weight_path + "/Topic_weights")

    if not os.path.isfile(weight_path + "/Topic_weights/Main_topic.zip") or TOPIC_flag:
        print("Downloading Transformer pretrained index...")
        output = weight_path + "/Topic_weights/Main_topic.zip"
        gdown.download(loaded["Main-model-url"], output, quiet=False)
        with ZipFile(weight_path + "/Topic_weights/Main_topic.zip", "r") as z:
            z.extractall(weight_path + "/Topic_weights")

    if not os.path.isfile(weight_path + "/Topic_weights/Sub_topic.zip") or TOPIC_flag:
        print("Downloading Transformer pretrained index...")
        output = weight_path + "/Topic_weights/Sub_topic.zip"
        gdown.download(loaded["Sub-model-url"], output, quiet=False)
        with ZipFile(weight_path + "/Topic_weights/Sub_topic.zip", "r") as z:
            z.extractall(weight_path + "/Topic_weights")

    print("Setup has just overed!")