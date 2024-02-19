import json
import PyPDF2
import docx
import heapq

from os import walk
from sentence_transformers import util

DATA_PATH = ''
TOP_SCORES_LIMIT = 0

def load_config():
    global DATA_PATH
    global TOP_SCORES_LIMIT
    with open('./config.json') as file:
        data = json.load(file)
        DATA_PATH = data["dataPath"]
        TOP_SCORES_LIMIT = data["topScoresLimit"]

def get_sentences_from_text(text: str) -> list[str]:
    text = text.replace('\n', ' ')
    sentences = text.split('. ')
    filtered_sentences = []
    for sentence in sentences:
        if len(sentence) != 0:
            filtered_sentences.append(sentence)
    return filtered_sentences

def get_sentences_from_txt(dirpath: str, filename: str) -> str:
    with open(f'{dirpath}/{filename}') as file:
        raw_text = ''
        for line in file.readlines():
            raw_text += line
        return raw_text

def get_sentences_from_file(dirpath: str, filename: str, file_ext: str) -> str:
    with open(f'{dirpath}/{filename}', 'rb') as file:
        raw_text = ''
        if file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                raw_text += page_text
        else:
            doc = docx.Document(file)
            for paragraph in doc.paragraphs:
                raw_text += paragraph.text
        return raw_text

def load_data_from_file(dirpath: str, filename: str) -> list[str]:
    file_ext = filename.split('.')[-1]
    raw_text = []
    if file_ext == 'txt':
        raw_text = get_sentences_from_txt(dirpath, filename)
    else:
        raw_text = get_sentences_from_file(dirpath, filename, file_ext)
    return get_sentences_from_text(raw_text)

def load_data() -> tuple[list[str], dict]:
    load_config()
    data = []
    metadata = {}
    for (_, dirnames, _) in walk(DATA_PATH):
        for dirname in dirnames:
            for (dirpath, _, filenames) in walk(DATA_PATH + dirname):
                for filename in filenames:
                    sentences = load_data_from_file(dirpath, filename)
                    for sentence in sentences:
                        data.append(sentence)
                        metadata[len(data) - 1] = {
                            'filename': filename,
                        }
    return data, metadata

def search(model, embedings, input_sentence):
    input_embedings = model.encode([input_sentence], convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedings, input_embedings)
    top_scores = []
    for i in range(len(embedings)):
        heapq.heappush(top_scores, (cosine_scores[i][0], i))
        if len(top_scores) > TOP_SCORES_LIMIT:
            heapq.heappop(top_scores)
    top_scores.reverse()
    return top_scores
