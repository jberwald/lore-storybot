from gliner import GLiNER

def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged


if __name__ == "__main__":

    model = GLiNER.from_pretrained("numind/NuNerZero")

    # NuZero requires labels to be lower-cased!
    labels = ["organization", "initiative", "project"]
    labels = [l.lower() for l in labels]

    text = ("At the annual technology summit, the keynote address was delivered by a senior "
            "member of the Association for Computing Machinery Special Interest Group on Algorithms and Computation "
            "Theory, which recently launched an expansive initiative titled 'Quantum Computing and Algorithmic "
            "Innovations: Shaping the Future of Technology'. This initiative explores the implications of quantum "
            "mechanics on next-generation computing and algorithm design and is part of a broader effort that includes "
            "the 'Global Computational Science Advancement Project'. The latter focuses on enhancing computational "
            "methodologies across scientific disciplines, aiming to set new benchmarks in computational efficiency and accuracy.")

    entities = model.predict_entities(text, labels)

    print("** entities: ")

    entities = merge_entities(entities)

    # print(entities)
    et = {}
    for entity in entities:
        et[entity['label']] = entity['text']

    print(et)
    # for entity in entities:
    #     print(entity["text"], "=>", entity["label"])