import pickle
import codecs
import ujson
import os


def mkdir_folder(folder):
    os.makedirs(folder, exist_ok=True)


def write_json(filename, dataset):
    with codecs.open(filename, mode="w", encoding="utf-8") as f:
        ujson.dump(dataset, f)


def read_json(input_file):
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        data = ujson.load(f)
        return data


def read_file(path, encoding="utf-8"):
    with open(path, 'r', errors='ignore', encoding=encoding) as file:  # 文档中编码有些问题，所有用errors过滤错误
        content = file.read()
        return content


def write_data(sentences, fileName):
    """
    函数说明：把处理好的写入到文件中，备用
    参数说明：
    """
    out = open(fileName, 'w')
    for sentence in sentences:
        out.write(sentence+"\n")
    print("done!")


def read_bunch(path):
    with open(path, 'rb') as file:
        bunch = pickle.load(file)
        # pickle.load(file)
        # 函数的功能：将file中的对象序列化读出。
    return bunch


def write_bunch(path, bunchFile):
    """[bunch: 大量; 大批;]

    Args:
        path ([type]): [description]
        bunchFile ([type]): [description]
    """
    with open(path, 'wb') as file:
        pickle.dump(bunchFile, file)
