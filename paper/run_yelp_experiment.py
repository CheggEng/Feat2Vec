import sys,os
import yaml

constants = yaml.load(open("paper/constants.yaml", 'rU'))
#old_stdout = sys.stdout
#log_file = open(os.path.join(constants['datadir'],'experimentlog.txt'), "w")
#sys.stdout = log_file

execfile("paper/create_yelp_data.py")
execfile("paper/clean_yelp_data.py")
execfile("paper/feat2vec_yelp.py")
execfile("paper/word2vec_yelp.py")
execfile("paper/doc2vec_yelp.py")
execfile("paper/eval_f2v_yelp.py")
#sys.stdout = old_stdout
#log_file.close()
