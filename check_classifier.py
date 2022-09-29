import torch
from models.sa_classifier import QuantizedClassifier
from modules.slot_attention import SlotAttentionBase


if __name__ == "__main__":

    slotattention = QuantizedClassifier()
    state_dict = torch.load("/home/alexandr_ko/quantised_sa_od/clevr10_sp")
    slotattention.load_state_dict(state_dict=state_dict, strict=False)
    print("Done")

    # slotattention = SlotAttentionBase(10, 64)
    # state_dict = torch.load("/home/alexandr_ko/quantised_sa_od/clevr10_sp")
    # state_dict = {key[len('slot_attention.'):]: state_dict[key]
    #               for key in state_dict if key.startswith('slot_attention')}
    # slotattention.load_state_dict(state_dict=state_dict)
    # answ = slotattention(torch.randn(512, 1024, 64))

    # print("DOne")
