# Evaluation-of-pretrained-language-models-on-music-understanding
The official repository of 'Evaluation of pretrained language models on music understanding', accepted at NLP4MusA and ISMIR workshop.

In this work we tried to quantify the musical knowledge of Transformer Language, as well as discover specific deficiencies that they posses. The biggest problem discovered is the inabillity to model negation (E.g. the models cannot infer that the pair 'guitar', 'bass' is more semantically similar with respect to 'guitar', 'no guitar').

# Introduction and general information
Language models have been used in multimodal music applications but haven't been thoroughly examined about the musical knowledge they encompass. As a first step towards evaluation, we proposed to use the AudioSet taxonomy, which is large enough yet we are able to manually evaluate it. 

In more detail, we used the Instruments and Genre subtree and generated all the plausible triplets (you can find a visualization in https://www.jordipons.me/apps/audioset/). A triplet is defined as a set of (\<anchor\>, \<positive\>, \<negative\>), where the anchor term has to be more semantically similar with respect to the positive rather than the negative term. In order to find which pair is more semantically similar, we used the tree-based distance between the terms.

If you want to generate the full list of triplets, you can run ```build_triplets.py```, which includes the AudioSet music and Instrument tree.

After generating the triplets, we manually inspected all of them and found that there are a lot of triplets with 1) vague relative similarity: it's not very easy to discern if a bowed instrument is more similar to a guitar than a Zither, as both are stringed instruments but plucked 2) almost the same relative similarity: a guitar is a plucked instrument as much an electric bass.

The finalized version of the triplets to be used are **audioset_triplets_genre.csv** and **audioset_triplets_instruments.csv**.

Using the filtered triplets, we utilized the sentence-transformers package and its 6 generic pretrained models, which are:
1. all-mpnet-base-v2 (https://huggingface.co/microsoft/mpnet-base)
2. all-distilroberta-v1 (https://huggingface.co/distilbert/distilroberta-base)
3. all-MiniLM-L12-v2 (https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
4. all MiniLM-L6-v2 (https://huggingface.co/nreimers/MiniLM-L6-H384-uncased)
5. paraphrase-albert-small-v2 (https://huggingface.co/nreimers/albert-small-v2)
6. paraphrase-MiniLM-L3-v2 (https://huggingface.co/nreimers/MiniLM-L6-H384-uncased)

and used the following code snippet to generate the embeddings with different textual inputs and the pair-wise cosine similarity (for more details see: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html).

```
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("hkunlp/instructor-large")
query = "where is the food stored in a yam plant"
query_instruction = (
    "Represent the Wikipedia question for retrieving supporting documents: "
)
corpus = [
    'Yams are perennial herbaceous vines native to Africa, Asia, and the Americas and cultivated for the consumption of their starchy tubers in many temperate and tropical regions. The tubers themselves, also called "yams", come in a variety of forms owing to numerous cultivars and related species.',
    "The disparate impact theory is especially controversial under the Fair Housing Act because the Act regulates many activities relating to housing, insurance, and mortgage loansâ€”and some scholars have argued that the theory's use under the Fair Housing Act, combined with extensions of the Community Reinvestment Act, contributed to rise of sub-prime lending and the crash of the U.S. housing market and ensuing global economic recession",
    "Disparate impact in United States labor law refers to practices in employment, housing, and other areas that adversely affect one group of people of a protected characteristic more than another, even though rules applied by employers or landlords are formally neutral. Although the protected classes vary by statute, most federal civil rights laws protect based on race, color, religion, national origin, and sex as protected traits, and some laws include disability status and other traits as well.",
]
corpus_instruction = "Represent the Wikipedia document for retrieval: "

query_embedding = model.encode(query, prompt=query_instruction)
corpus_embeddings = model.encode(corpus, prompt=corpus_instruction)
similarities = cos_sim(query_embedding, corpus_embeddings)
print(similarities)
# => tensor([[0.8835, 0.7037, 0.6970]])
```

We didn't include any code as we think that the application using the aforementioned sentence-transformer snippet is quite straight-forward. Regardless, feel free to contact us (see Questions section for the email to contact).


## Prompt sensitivity
The prompts used for the experiment of prompts sensitivity are:
1. The sound of \<label\>
2. Music made with \<label\>
3. A \<label\> track
4. This is a recording of \<label\>
5. A song with \<label\>
6. A track with \<label\> recorded
7. A music project with \<label\>
8. Music made from \<label\>
9. Music of \<label\>
10. A music recording of \<label\>
11. This song is made from \<label\>
12. The song has \<label\>
13. Music song with \<label\>
14. Music song with \<label\> recorded
15. Musical sounds from \<label\>
16. This song sounds like \<label\>
17. This music sounds like \<label\>
18. Song with \<label\> recorded
19. A \<label\> music track
20. Sound of \<label\>

## Inability to model negation
For the experiment to evaluate if the Language models can model negation, we used the following prompts for each valid triplet with negation:
1. No \<label\>
2. No the sound of \<label\>
3. Doesn't sound like \<label\>
4. Not music from \<label\>

The finalized valid triplets with negation are **audioset_triplets_genres_with_negative.csv** and **audioset_triplets_instruemnts_with_negative.csv**. 

In order to use the prompts, use a string based replace function as in

```text.replace('no ', '<prompt in mind>')```

## Sensitivity towards the presence of specific words
We used AudioSet's definitions. To include both words and definitions, we used a prompt '\<label\> is a <definition>', where \<label\> is the word/term/phrase under consideration and <definition> is the definition provided in Audioset. For the definition only experiment we used only the definition provided by AudioSet.

For your convenience, we have included **audioset_terms_definitions_music.csv** with the AudioSet definition for every musical term.

# Citation
```

```

# Questions
For any questions send an email to i.vasilakis@qmul.ac.uk