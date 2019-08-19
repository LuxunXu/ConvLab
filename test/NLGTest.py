from convlab.modules.nlg.multiwoz.multiwoz_template_nlg.multiwoz_template_nlg import MultiwozTemplateNLG
from convlab.modules.nlg.multiwoz.sc_lstm.nlg_sc_lstm import SCLSTM

# dialog act
dialog_acts = {'Hotel-Recommend': [['Name', 'Hilton'], ['Price', '$200']]}
# whether from user or system
is_user = False

multiwoz_template_nlg = MultiwozTemplateNLG(is_user, mode='auto_manual')
print(dialog_acts)
print(multiwoz_template_nlg.generate(dialog_acts))
print(multiwoz_template_nlg.generate(dialog_acts))
print(multiwoz_template_nlg.generate(dialog_acts))


sclstm = SCLSTM(use_cuda=False, model_file='https://convlab.blob.core.windows.net/models/nlg-sclstm-multiwoz.zip')
meta = {'Hotel-Recommend': [['Name', 'Hilton'], ['Price', '$200']]}
print(sclstm.generate_delex(meta))
print(sclstm.generate_slots(meta))
print(sclstm.generate(meta))

