import csv
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(prog = 'Evaluation of pretrained language models on music understanding',
                                 description = 'Official implementation of paper accepted at NLP4MusA')

parser.add_argument('-t', '--type', help = 'The type of subtree for which the triplets will be generated.',
                    default = 'instruments')

args = parser.parse_args()

instruments_subtree = {
    "Music": {
        "Musical instrument": {
            "Plucked string instrument": {
                "Guitar": [
                    "Electric guitar",
                    "Bass guitar",
                    "Acoustic guitar",
                    "Steel guitar, slide guitar",
                    "Tapping (guitar technique)",
                    "Strum"],
                "Banjo": [],
                "Sitar": [],
                "Mandolin": [],
                "Zither": [],
                "Ukulele": []
            },
            "Keyboard (musical)": {
                "Piano": {
                    "Electric piano": [
                        "Clavinet",
                        "Rhodes piano"
                    ]
                },
                "Organ": [
                    "Electric organ",
                    "Hammond organ"
                ],
                "Synthesizer": [
                    "Sampler",
                    "Mellotron"
                ],
                "Harpsichord": []
            },
            "Percussion": {
                "Drum kit": [
                    "Drum machine"
                ],
                "Drum": {
                    "Snare drum": [
                        "Rimshot",
                        "Drum roll"
                    ],
                    "Bass drum": [],
                    "Timpani": [],
                    "Tabla": []
                },
                "Cymbal": [
                    "Hi-hat",
                    "Crash cymbal"
                ],
                "Cowbell": [],
                "Wood block": [],
                "Tambourine": [],
                "Rattle (instrument)": [
                    "Maraca"
                ],
                "Gong": [],
                "Tubular bells": [],
                "Mallet percussion": [
                    "Marimba, xylophone",
                    "Glockenspiel",
                    "Vibraphone",
                    "Steelpan"
                ]
            },
            "Orchestra": [],
            "Brass Instrument": [
                "French horn",
                "Trumpet",
                "Trombone",
                "Cornet",
                "Bugle"
            ],
            "Bowed string instrument": {
                "String section": [],
                "Violin, fiddle": [
                    "Pizzicato"
                ],
                "Cello": [],
                "Double bass": []
            },
            "Wind instrument, woodwind instrument": {
                "Flute": [],
                "Saxophone": [
                    "Alto saxophone",
                    "Soprano saxophone"
                ],
                "Clarinet": [],
                "Oboe": [],
                "Bassoon": []
            },
            "Harp": [],
            "Choir": [],
            "Bell": {
                "Church bell": [],
                "Cowbell": [],
                "Jingle bell": [],
                "Bicycle bell": [],
                "Tuning fork": [],
                "Chime": [
                    "Wind chime"
                ],
                "Change ringing (campanology)": []
            },
            "Harmonica": [],
            "Accordion": [],
            "Bagpipes": [],
            "Didgeridoo": [],
            "Shofar": [],
            "Theremin": [],
            "Singing bowl": [],
            "Musical ensemble": [],
            "Bass (instrument role)": [],
            "Scratching (performance technique)": []
        }
    }
}

genre_subtree = {
    "Music": {
        "Music genre": {
            "Pop music": [],
            "Hip hop music": [
                "Grime music",
                "Trap music",
                "Beatboxing"
            ],
            "Rock music": [
                "Heavy metal",
                "Punk rock",
                "Grunge",
                "Progressive rock",
                "Rock and roll",
                "Psychedelic rock"
            ],
            "Rhythm and blues": [],
            "Soul music": [],
            "Reggae": [
                "Dub"
            ],
            "Country": [
                "Swing music",
                "Bluegrass"
            ],
            "Funk": [],
            "Folk music": [],
            "Middle Eastern music": [],
            "Jazz": [],
            "Disco": [],
            "Classical music": [
                "Opera"
            ],
            "Electronic music": {
                "House music": [],
                "Techno": [],
                "Dubstep": [],
                "Electro": [],
                "Drum and bass": [
                    "Oldschool jungle"
                ],
                "Electronica": [],
                "Electronic dance music": [],
                "Ambient music": [
                    "Drone music"
                ],
                "Trance music": [],
                "Noise music": [],
                "UK garage": []
            },
            "Music of Latin America": [
                "Cumbia",
                "Salsa music",
                "Soca music",
                "Kuduro",
                "Funk carioca",
                "Flamenco"
            ],
            "Blues": [],
            "Music for children": [],
            "New-age music": [],
            "Vocal music": [
                "A capella",
                "Chant",
                "Beatboxing"
            ],
            "Music of Africa": [
                "Afrobeat",
                "Kwaito"
            ],
            "Christian music": [
                "Gospel music"
            ],
            "Music of Asia": [
                "Carnatic music",
                "Music of Bollywood"
            ],
            "Ska": [],
            "Traditional music": [],
            "Independent music": []
        },
        "Musical concepts": {
            "Song": [],
            "Melody": [],
            "Musical note": [],
            "Beat": [
                "Drum beat"
            ],
            "Chord": [],
            "Harmony": [],
            "Bassline": [],
            "Loop": [],
            "Drone": []
        },
        "Music role": [
            "Background music",
            "Theme music",
            "Jingle (music)",
            "Soundtrack music",
            "Lullaby",
            "Video game music",
            "Christmas music",
            "Dance music",
            "Wedding music",
            "Birthday music"
        ],
        "Music mood": [
            "Happy music",
            "Funny music",
            "Sad music",
            "Tender music",
            "Exciting music",
            "Angry music",
            "Scary music"
        ]
    }
}


def flatten_ontology(ontology):
    nodes = []
    for key, value in ontology.items():
        if isinstance(value, dict):
            nodes.append(key)
            nodes.extend(flatten_ontology(value))
        elif isinstance(value, list):
            nodes.append(key)
            nodes.extend([item for item in value])
    return nodes

from collections import deque

class TreeNode:
    def __init__(self, name):
        self.name = name
        self.children = []

def build_tree(node, subtree):
    for key, value in subtree.items():
        child = TreeNode(key)
        if isinstance(value, dict):
            build_tree(child, value)
        elif isinstance(value, list):
            for item in value:
                child.children.append(TreeNode(item))
        node.children.append(child)

def shortest_distance(root, node1, node2):
    if root is None or node1 is None or node2 is None:
        return -1

    def find_path(node, target):
        if node is None:
            return None
        if node.name == target:
            return [node.name]
        for child in node.children:
            path = find_path(child, target)
            if path:
                return [node.name] + path
        return None

    path1 = find_path(root, node1)
    path2 = find_path(root, node2)

    if not path1 or not path2:
        return -1

    i = 0
    while i < len(path1) and i < len(path2) and path1[i] == path2[i]:
        i += 1

    if i == 2:
        return -1

    return len(path1) + len(path2) - 2 * i

def find_distance_to_root(root, node_name):
    def find_path_to_root(node, target, path):
        if node is None:
            return False
        path.append(node.name)
        if node.name == target:
            return True
        for child in node.children:
            if find_path_to_root(child, target, path):
                return True
        path.pop()
        return False

    path_to_root = []
    find_path_to_root(root, node_name, path_to_root)
    return len(path_to_root) - 1  # Subtract 1 to exclude the root itself from the distance count


def get_leaf_nodes(ontology):
    def _get_leaf_nodes_helper(subtree):
        leaf_nodes = []
        for key, value in subtree.items():
            if isinstance(value, dict):
                leaf_nodes.extend(_get_leaf_nodes_helper(value))
            elif isinstance(value, list):
                leaf_nodes.extend([item for item in value])
        if not isinstance(subtree, list) and not isinstance(subtree, dict):
            leaf_nodes.append('')
        return leaf_nodes
    
    return _get_leaf_nodes_helper(ontology)

tree = TreeNode('instrument_ontology')

if args.type == 'instruments':
    ontology = instruments_subtree
    filename_to_save = 'audioset_triplets_instruments.csv'
elif args.type == 'genres':
    ontology = genre_subtree
    filename_to_save = 'audioset_triplets_genres.csv'
else:
    print('Not a valid subtree type. Exitting...')
    exit()

build_tree(tree, ontology)

triplets = []

ontology_list = flatten_ontology(ontology)[1:]

leaf_nodes = get_leaf_nodes(ontology)
 
with open(filename_to_save, 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['anchor', 'positive', 'negative'])
    for term_1_counter, term_1 in enumerate(tqdm(ontology_list)):
        for term_2 in ontology_list[term_1_counter+1:]:
            if not(term_1 == term_2):
                pos_term_distance = shortest_distance(tree, term_1, term_2)

                if pos_term_distance > 0:
                    for term_3 in ontology_list:
                        if not(term_3 == term_1) and not(term_3 == term_2):
                            if shortest_distance(tree, term_1, term_3) > pos_term_distance:
                                triplets.append([term_1, term_2, term_3])
                                writer.writerow([term_1, term_2, term_3])


print('d')