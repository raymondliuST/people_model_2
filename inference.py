from pm_experiment import pmExperiment
import json
import yaml
import torch
from bertviz import model_view

def inference(input, path):

    with open('config/pm.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Loading tokenizer
    with open("ml_dataset/tokenizer.json", 'r') as file:
        vocab_dict = json.load(file)

    mask_token = len(vocab_dict)
    vocab_dict[len(vocab_dict)] = "mask"

    model = pmExperiment.load_from_checkpoint(path,config=config,vocab_size=len(vocab_dict))


    index_vocab_dict = {v: k for k, v in vocab_dict.items()}

    def convert_to_indices(word_list_2d, index_vocab_dict):
        return [[int(index_vocab_dict.get(word, -1)) for word in row] for row in word_list_2d]
    def convert_to_str(idx_list_2d, vocab_dict):
        return [[vocab_dict.get(str(idx.item()), "") for idx in row] for row in idx_list_2d]

    input_idx = convert_to_indices(input, index_vocab_dict)

    model_input = {
        "input" :  torch.tensor(input_idx, dtype=torch.long),
        "label" : None,
        "mask" : None
    }

    with torch.no_grad():
        output = model.forward(model_input)

    predictions_idx = output.argmax(2)
    predictions = convert_to_str(predictions_idx, vocab_dict)

    return predictions
    import pdb
    pdb.set_trace()


if __name__ == "__main__":

    input = [["Firefox","Tablet", "Linux", "United States"],
             ["Firefox","Tablet", "mask", "United States"],
             ["Firefox","Tablet", "mask",  "mask"]]
    
    for i in range(5):
        print(inference(input, "checkpoints/pm5/sfvt79n8/checkpoints/epoch=999-step=136999.ckpt"))