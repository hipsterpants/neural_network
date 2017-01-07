# Reads in data files
import numpy as np

class FileReaderIris:


    file_name = input("Input the name of the data file to be scanned:")
    data_file = open(file_name, 'r')
    
    # maybe this is it? Naming shenanigans
    data = list()
    correct_output = list()
    for line in data_file:
        line = line.rstrip('\n')
        split_line = line.split(",")
        index = len(split_line)
        correct_output.append(split_line[index-1])
        split_line.pop()
        float_data = list()
        for info in split_line:
            float_data.append(float(info))
        data.append(float_data)
        
    data_file.close()

        
    #actually need to turn this into methods or something? but function should
        #work
    data = np.array(data)
    #hardcoded this :(
    output = np.zeros((150, 3))
    
    i = 0
    
    for flower_name in correct_output:
        if flower_name == 'Iris-setosa':
            output[i][0] = 1
        elif flower_name == 'Iris-versicolor':
            output[i][1] = 1
        elif flower_name == 'Iris-virginica':
            output[i][2] = 1
        i = i + 1
        
print(data)
print(correct_output)
print(output)
    