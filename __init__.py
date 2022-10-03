try :
    from setup import setup_environ, download_weights
    setup_environ()
    download_weights()
except Exception :
    from .setup import setup_environ, download_weights

    setup_environ()
    download_weights()

# from aimodel import AIModel
# __all__=['AIModel']



