from utils import load_data, search
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def loop(sentences, embedings, metadata):
    try:
        while True:
            input_sentence = input("Enter a sentence you want to search: ")
            results = search(model, embedings, input_sentence)
            print('Best results for:', input_sentence)
            for i, (_score, idx) in enumerate(results):
                print(f"{i + 1}.")
                print("Text:", sentences[idx])
                print("Appears in:", metadata[idx]['filename'])
                print()
            print("")
    except KeyboardInterrupt:
        print("\nBye :)")

def main():
    print("Loading data...")
    sentences, metadata = load_data()
    print("Processing data...")
    embedings = model.encode(sentences, convert_to_tensor=True)
    print("Ready to answer queries")
    loop(sentences, embedings, metadata)

if __name__ == '__main__':
    main()
