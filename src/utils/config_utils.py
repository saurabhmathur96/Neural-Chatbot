from ConfigParser import ConfigParser
from collections import namedtuple
from pprint import pprint


parser = ConfigParser()
parser.read('configuration.cfg')

TrainConfig = namedtuple('TrainConfig', 'batch_size n_iter n_epoch')
ModelConfig = namedtuple('ModelConfig', 'sequence_length vocabulary_size hidden_size weights_path')
DataConfig = namedtuple('DataConfig', 'blacklist_path pairs_path save_path extract_dir vocabulary_path filtered_path unk_ratio')
Settings = namedtuple('Settings', 'train model data')

train = TrainConfig(int(parser.get('Training', 'batch_size')),
                    int(parser.get('Training', 'n_iter')),
                    int(parser.get('Training', 'n_epoch')))

model = ModelConfig(int(parser.get('Model', 'sequence_length')),
                    int(parser.get('Model', 'vocabulary_size')),
                    int(parser.get('Model', 'hidden_size')),
                    parser.get('Model', 'weights_path'))

data = DataConfig(parser.get('Data', 'blacklist_path'),
                  parser.get('Data', 'pairs_path'),
                  parser.get('Data', 'opus11_save_path'),
                  parser.get('Data', 'opus11_extract_dir'),
                  parser.get('Data', 'vocabulary_path'),
                  parser.get('Data', 'filtered_path'),
                  float(parser.get('Data', 'unk_ratio')))

settings = Settings(train, model, data)


if __name__ == '__main__':
    print ('Settings: ')
    pprint(dict(settings._asdict()))
