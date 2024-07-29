# when-an-image-tells-a-story

The repository for the paper "When an Image Tells a Story: The Role of Visual and Semantic Information for Generating Paragraph Descriptions".


## Preparing data

1. `preproc-dataset.py`: the script tokenises Stanford image paragraphs into sentences and formats the dataset in COCO Karpathy style for the model.
2. `create_input_files.py`: the script generates files which include word map, image features, captions and their lengths for all three data splits. Note that image features where extracted in advance, and this script simply distributes them between data splits properly.