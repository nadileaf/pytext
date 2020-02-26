import os

PORT = int(os.getenv('PORT', '9200'))

MODEL_FILE = "data/news_model.c2"
MODEL_DB_TYPE = "minidb"