"""
Original source of data and splits: https://cs.stanford.edu/people/ranjaykrishna/im2p/index.html

input1: paragraph dataset, .json file
input2: merged file with splits (train, val, test), .json file

output1: paragraphs in MSCOCO Karpathy style
"""

import json
import argparse
from tqdm import tqdm
from spacy.lang.en import English
from autocorrect import Speller
from pathlib import Path

def process_paragraphs(params):
    # Initialise spell checker and sentence segmenter
    spell = Speller(lang='en')
    nlp = English()
    nlp.add_pipe('sentencizer')

    # Define input and output paths
    input_paragraphs_path = Path(params['input_par_json'])
    input_splits_path = Path(params['input_splits_json'])
    output_path = Path(params['output_par_json'])

    # Check if the output directory exists, create it if it does not
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input data files
    with input_paragraphs_path.open('r') as f:
        paragraphs = json.load(f)
    with input_splits_path.open('r') as f:
        splits = json.load(f)

    curr_sentence_id = 0
    all_images = {'images': []}

    # Process each paragraph
    for par_num, paragraph_data in tqdm(enumerate(paragraphs, 1)):
        # Extract file path and name from URL
        url_parts = paragraph_data['url'].split('/')
        filepath, filename = url_parts[-2], url_parts[-1]
        imgid = par_num
        split = splits[filename.split('.')[0]]

        # Tokenise paragraph into sentences
        par_spacy = nlp(paragraph_data['paragraph'])
        sentences = list(par_spacy.sents)

        # Initialise dictionary for storing paragraph data
        paragraph_dict = {
            'filepath': filepath,
            'filename': filename,
            'imgid': imgid,
            'split': split,
            'sentences': [],
            'stanford_par_id': int(filename.split('.jpg')[0]),
            'sentids': []
        }

        # Process each sentence in the paragraph
        for raw_sentence in sentences:
            tokens = [spell(token.orth_.lower().rstrip()) for token in raw_sentence if token.orth_.strip()]

            if tokens:
                sentence_dict = {
                    'tokens': tokens,
                    'raw': str(raw_sentence),
                    'imgid': imgid,
                    'sentid': curr_sentence_id
                }
                paragraph_dict['sentences'].append(sentence_dict)
                paragraph_dict['sentids'].append(curr_sentence_id)
                curr_sentence_id += 1

        all_images['images'].append(paragraph_dict)

    # Save processed data to output file
    with output_path.open('w') as f:
        json.dump(all_images, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_par_json', default='../data/paragraphs_v1.json', help='Location of the file with original paragraphs')
    parser.add_argument('--input_splits_json', default='../data/splits.json', help='Location of the merged paragraph splits')
    parser.add_argument('--output_par_json', default='../data/data-out/dataset_paragraphs_v1.json', help='Location of the output file with preprocessed paragraphs')
    args = parser.parse_args()
    params = vars(args)
    process_paragraphs(params)
