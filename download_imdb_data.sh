#Downloaded on 9/15/17!
sudo apt-get install s3cmd
s3cmd --configure
s3cmd --requester-pays --continue get s3://imdb-datasets/documents/v1/current/title.basics.tsv.gz
s3cmd --requester-pays --continue get s3://imdb-datasets/documents/v1/title.crew.tsv.gz
s3cmd --requester-pays --continue get s3://imdb-datasets/documents/v1/current/title.episode.tsv.gz
s3cmd --requester-pays --continue get s3://imdb-datasets/documents/v1/current/title.principals.tsv.gz
s3cmd --requester-pays --continue get s3://imdb-datasets/documents/v1/current/title.ratings.tsv.gz
s3cmd --requester-pays --continue get s3://imdb-datasets/documents/v1/current/name.basics.tsv.gz
