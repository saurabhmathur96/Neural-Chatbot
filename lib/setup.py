import nltk
import os
from os import path


if __name__ == '__main__':
    nltk.download('punkt')
    os.makedirs('data/raw') if not path.exists('data/raw') else None
    os.makedirs('data/processed/') if not path.exists('data/processed/') else None
    os.makedirs('data/processed/opus11') if not path.exists('data/processed/opus11') else None