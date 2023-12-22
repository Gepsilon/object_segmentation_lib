from os import path

PathValidator = lambda p: path.exists(p)
SkipValidation = lambda _: True
