import os
import sys
import zipfile

sys.path.append('src/utils')
from data_utils import download


if __name__ == '__main__':
    url = 'http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip'
    save_path = 'data/raw/cornell_movie_dialog_corpus.zip'
    # download(url, save_path)

    extract_path = 'data/raw'
    to_extract = [('cornell movie-dialogs corpus/movie_lines.txt', 'data/raw/movie_lines.txt'),
    ('cornell movie-dialogs corpus/movie_conversations.txt', 'data/raw/movie_conversation.txt')]
    with zipfile.ZipFile(save_path, 'r') as archive:
        for source, target in to_extract:
            contents = archive.read(source)
            with open(target, 'wb') as handle:
                handle.write(contents)

    # os.remove(save_path)