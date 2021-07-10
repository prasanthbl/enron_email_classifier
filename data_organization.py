import os
import re
import shutil
import ntpath

# This function gets all the file names and their path that is present in the given directory/sub directory
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

# This function segregates the email data based on category mentioned in *.cats file and put the files
# in the right folders which will be used for training /testing the classifier model.
def load_files(srcdirectory, dstdirectory):
    result = []
    # extension of the file name which have category details
    fileExt1 = r".cats"
    # total number of class that is present (including 7,8 which are not used in this assignment)
    n_class = [1,2,3,4,5,6,7,8]

    # Get the list of all files in directory tree at given path
    listOfFiles = getListOfFiles(srcdirectory)

    # Print the files
    for elem in listOfFiles:
        print(elem)

    for fname in listOfFiles:

        if fname.endswith(fileExt1):
            print(fname)
            ftext = open(fname, 'r', encoding='utf-8')
            print(ftext)
            print(ftext.read())
            count_class = 1
            for i in n_class:
                pattern = re.compile("1,"+str(i)+",")

                for line in open(fname):
                    for match in re.finditer(pattern, line):
                        # If-statement after search() tests if it succeeded and only one category
                        # of email data is considered
                        if match and count_class == 1:
                            print ('found', match.group())  ## 'found word:cat'
                            temp_var = os.path.splitext(fname)
                            file_name = ntpath.basename(temp_var[0])
                            shutil.copyfile(temp_var[0] + ".txt", dstdirectory + "/" + str(i) + "/" + file_name + ".txt")
                            count_class = count_class + 1
                        else:
                            print ('did not find')

    return result


positive_examples = load_files('enron_with_categories','enron_classifier_organized_data')
