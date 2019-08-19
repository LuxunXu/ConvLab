from convlab.modules.nlu.multiwoz.onenet.nlu import OneNetLU

input1 = input("Enter your name: ")
#There is a movie at 10:15 on wednesday .
print(input1)

oneNetLU = OneNetLU(model_file='https://convlab.blob.core.windows.net/models/onenet.tar.gz')

print(oneNetLU.parse(input1))
