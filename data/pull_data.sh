wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip -O data.zip
unzip data.zip .
rm data.zip 
cp "cornell movie-dialogs corpus"/* .
rm -r "cornell movie-dialogs corpus"