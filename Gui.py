import tkinter as tk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

root = tk.Tk()
canvas = tk.Canvas(root)
canvas.pack()

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

entry = tk.Text(root, height=10)
entry.pack(side="top")

entry1 = tk.Text(root, height=1)
entry1.pack(side="top")

def generate():
    instruction = 'Instruction: given a related knowledge, you need to summarize it.'
    knowledge = entry.get("1.0", "end")
    dialog = ''
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=64, min_length=16, top_p=0.95, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label.config(text=response)
    return instruction, knowledge, dialog

def button_command():
    generate()

button = tk.Button(root, text="Summarize Text", width=50, command=button_command)
button.pack(side="top")

label = tk.Label(root, text="")
label.pack(side="top")

root.mainloop()
