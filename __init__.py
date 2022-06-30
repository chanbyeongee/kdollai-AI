from setup import setup_environ, download_weights
from .transformer_models.aimodel import AIModel
__all__=['tensormodels','tokenizer', 'AIModel']

setup_environ()
download_weights()



