# Author: Alexander Åblad
# 2022-01-07
# DD2434
from sklearn.feature_extraction.text import TfidfVectorizer
import os

class TextLoader:

    def __init__(self):
        self.stopwords = self.get_stop_words()

    def get_stop_words(self):
        filename = "malletsmart.txt"
        f = open(filename, 'r')
        stopwords = f.read().split()
        f.close()
        return stopwords

    def load_base_data(self, path, data=None):
        if not data:
            data = []

        for file in os.listdir(path):
            f = open(path + file, 'r', errors='ignore') # 'Ignore' to simplify process
            content = f.read()
            data.append(content)
            f.close()
        return data

    def get_vectorized_data(self, data):
        tvect = TfidfVectorizer(lowercase=True, stop_words=self.stopwords, analyzer='word', max_features=5000)
        V = tvect.fit_transform(data)
        return V.toarray()

    # load() when all documents are under the same folder
    def load(self, path):
        data = self.load_base_data(path)
        doc_array = self.get_vectorized_data(data)
        return doc_array

    # load_subfolder_data() when data is divided into subfolders
    def load_subfolder_data(self, path):
        data = None
        for subfolder in os.listdir(path):
            sub_path = path + subfolder + '/'
            data = self.load_base_data(sub_path, data)

        doc_array = self.get_vectorized_data(data)

        return doc_array


if __name__ == '__main__':
    # Replace 'data' in tvect.fit_transform(data) with test to run test data
    test = ['Hej jag heter 14 och han',
            'du heter inte nana@eduda',
            'hej där vem vet egentligen']

    # Test paths - one-folder path, subfolder path
    path = ''
    path2 = ''

    loader = TextLoader()
    array = loader.load(path)
    print(array)
    print(array.shape)
    print(sum(array[0]))

    print("\n\n")

    array2 = loader.load_subfolder_data(path2)
    print(array2)
    print(array2.shape)