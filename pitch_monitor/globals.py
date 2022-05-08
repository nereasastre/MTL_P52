import logging
import os
from pathlib import Path

import environ

BASE_DIR = Path(__file__).resolve().parent.parent

environment = environ.Env()
if os.path.isfile(".env"):
    logging.info("Reading Env file: .env")
    environ.Env.read_env(os.path.join(BASE_DIR, ".env"))
else:
    logging.info("Using os.environment")


def env(variable):
    return environment(variable)
