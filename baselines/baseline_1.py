import argparse
import json
import numpy as np
from utils.text_processing import load_glove_model, text_to_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--data_path', default="data/terms/ESG_taxonomy_train.json")
    parser.add_argument('--tagset', default="data/tagset/ESG_finsim.json")
    parser.add_argument('--k', default=7, type=int)

    args = parser.parse_args()
    k = args.k

    model = load_glove_model("models/ESG_reports_custom_w2v_100d.txt")
    data = json.load(open(args.data_path, "r"))
    tagset = json.load(open(args.tagset, "r"))

    for t_entity in tagset:
        t_entity["vector"] = text_to_vector(t_entity["concept"], model)

    for entity in data:
        entity["vector"] = text_to_vector(entity["term"], model)

    for entity in data:
        ontology_x_distance = [
            (o_entity, np.linalg.norm(entity["vector"] - o_entity["vector"]))
            for o_entity in tagset
        ]
        ontology_x_distance.sort(key=lambda x: x[1])
        ontology_x_distance = ontology_x_distance[:k]
        entity["predicted_concepts"] = [
            o_entity["concept"] for o_entity, _ in ontology_x_distance
        ]
        del entity["vector"]

    json.dump(data, open("data/outputs/baseline_1_ESG_taxonomy_predictions_w2v100.json", "w"), indent=4)
