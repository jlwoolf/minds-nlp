from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import stem_text
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_numeric
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short

import os


def remove_metadata(s):
    return s[s.find("\n\n")+2:]


def make_lowercase(s):
    return s.lower()


filter_options = {
    "remove metadata": remove_metadata,
    "make lowercase": make_lowercase,
    "strip punctuation": strip_punctuation,
    "strip tags": strip_tags,
    "strip numeric": strip_numeric,
    "strip multiple whitespaces": strip_multiple_whitespaces,
    "strip short": strip_short,
    "stem text": stem_text,
    "remove stopwords": remove_stopwords,
}


def createConfig():
    file = open("preprocess_config.txt", "w")
    selected_filters = []
    user_input = ""
    while user_input != 'done':
        index = 0
        temp_dict = {}
        for filter_name in [w for w in filter_options.keys() if w not in selected_filters]:
            print("["+str(index)+"] " + filter_name)
            temp_dict[index] = filter_name
            index += 1

        user_input = input(
            "Selected filter to add. Enter 'done' when finished: ")
        if user_input == 'done':
            break
        if int(user_input) >= 0 or int(user_input) < len(temp_dict):
            selected_filters.append(temp_dict[int(user_input)])
            file.write(temp_dict[int(user_input)] + "\n")


def loadConfig():
    config_path = 'preprocess_config.txt'
    while not os.path.isfile(config_path):
        user_input = input(
            "[1] Config file not found. Create one (1) or load from path (2)? ")

        if(user_input == "2"):
            config_path = input("Paste file path here: ")
        else:
            createConfig()

    selected_filters = []
    file = open(config_path)
    for line in file:
        selected_filters.append(filter_options[line.replace("\n", "")])

    return selected_filters


def preprocess_documents(docs):
    selected_filters = loadConfig()
    return [preprocess_string(d, filters=selected_filters) for d in docs]


def preprocess_document(doc):
    selected_filters = loadConfig()
    return preprocess_string(doc, filters=selected_filters)


if __name__ == "__main__":
    loadConfig()
