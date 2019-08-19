from convlab.modules.nlu.multiwoz.onenet.nlu import OneNetLU
from convlab.modules.nlu.multiwoz.milu.nlu import MILU

oneNetLU = OneNetLU(model_file='https://convlab.blob.core.windows.net/models/onenet.tar.gz')
miLU = MILU(model_file='https://convlab.blob.core.windows.net/models/milu.tar.gz')

print(oneNetLU.parse('I want Indian food at 10:15 on wednesday .'))
print(miLU.parse('There is a train leaving at 10:15 on wednesday .'))
