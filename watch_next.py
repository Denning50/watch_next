# Import libraries
import spacy
import numpy as np

# Create the function.
def get_next_movie(description):
    # load the pre-trained model
    nlp = spacy.load("en_core_web_md")

    # read the movie descriptions from the file
    with open('C:/Users/1708/Dropbox/JD22110005464/Software Engineer Bootcamp/T38/movies.txt', 'r', encoding='utf-8-sig') as f:
        movies = f.readlines()

    # preprocess the descriptions and create document vectors
    docs = [nlp(m.lower().strip()) for m in movies]

    # get the document vector of the input description
    query = nlp(description.lower())

    # compute the cosine similarity between the input description and all movie descriptions
    similarities = [np.dot(query.vector, d.vector) / (np.linalg.norm(query.vector) * np.linalg.norm(d.vector)) for d in docs]

    # get the index of the most similar movie to the input description
    idx = max(enumerate(similarities), key=lambda x: x[1])[0]

    # return the title of the most similar movie
    return movies[idx].strip()

# Define the desription
description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator"
# Print the result
print(get_next_movie(description))

# https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists