import tkinter as tk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

root = tk.Tk()
root.configure(bg="#333333")
root.geometry("600x600")

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-large-seq2seq")

entry = tk.Text(root, height=30, bg="#222222", fg="#ffffff")
entry.pack(side="top", fill="both", expand=True, padx=10, pady=10)
entry.pack_propagate(0)

def generate():
    instruction = 'Instruction: given a related knowledge, you need to summarize it.'
    knowledge = entry.get("1.0", "end")
    dialog = ''
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=64, min_length=16, top_p=1, do_sample=True, num_beams=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    label.config(text=response, fg="#ffffff")
    return instruction, knowledge, dialog

def button_command():
    generate()

button = tk.Button(root, text="Summarize Text", width=50, bg="#222222", fg="#ffffff", command=button_command)
button.pack(side="top", fill="both", expand=True, padx=10, pady=10)
button.pack_propagate(0)

label = tk.Label(root, text="", bg="#333333", fg="#ffffff", borderwidth=2, relief="groove")
label.pack(side="top", fill="both", expand=True, padx=10, pady=10)
label.pack_propagate(0)

canvas = tk.Canvas(root, bg="#333333", borderwidth=0, highlightthickness=0)
canvas.pack(side="top", fill="both", expand=True)
canvas.pack_propagate(0)

root.mainloop()
