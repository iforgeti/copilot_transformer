from torch import nn
import torch
import torchtext
from transformers import AutoTokenizer 
from utils.models_architect import Decoder
# models architect




def Load_model(device,save_path = "model/best_Decoder.pt",params_path = "model/param.pt"):
    params = torch.load(params_path)
    # "OUTPUT_DIM":OUTPUT_DIM,"HID_DIM":HID_DIM ,"DEC_HEADS":DEC_HEADS ,"DEC_PF_DIM":DEC_PF_DIM,"DEC_DROPOUT" :DEC_DROPOUT
    model = Decoder( params["OUTPUT_DIM"], params["HID_DIM"],params["DEC_LAYERS"], params["DEC_HEADS"],
                 params["DEC_PF_DIM"], params["DEC_DROPOUT"], device, params["TRG_PAD_IDX"])

    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()

    return model

def suggestor(prompt,model,tokenizer,device,max_lenght=130):
    tokenized =tokenizer(prompt)["input_ids"]
    trg =torch.LongTensor([tokenizer.bos_token_id]+tokenized).reshape(1,-1)
    trg=trg.to(device)
    with torch.no_grad():
        for i in range(max_lenght):
            output= model.greedy_decode(trg)
            trg = torch.cat((trg, output[-1].reshape(1,-1)), dim=1)

            if output[-1] == tokenizer.eos_token_id:
                break
    result = tokenizer.decode(trg.squeeze(0),skip_special_tokens=True)
    trg.detach()

    return result




if __name__ == "__main__":


    prompt = "import numpy"
    
    tokenizer = AutoTokenizer.from_pretrained("model/code-search-net-tokenizer-mod")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Load_model(device)

    print(suggestor(prompt,model,device,tokenizer))




