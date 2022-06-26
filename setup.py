import os

this_dir, this_filename = os.path.split(__file__)

os.environ['CHATBOT_ROOT'] = this_dir
print("Environment Variable Set Successfully. root: %s"%(os.environ['CHATBOT_ROOT']))

print("Downloading pretrained weights...")