try :
    from emo_classifier import *
    from ner_classifier import *
    from gd_generator import *
    from topic_classifier import *
except Exception :
    from .emo_classifier import *
    from .ner_classifier import *
    from .gd_generator import *
    from .topic_classifier import *

