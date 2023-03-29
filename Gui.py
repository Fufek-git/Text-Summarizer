import tkinter as tk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tkinter import *
from tkinter.ttk import *

root = tk.Tk()
root.configure(bg="#333333")
root.geometry("600x600")

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")

entry = tk.Text(root, height=30, bg="#222222", fg="#ffffff")
entry.pack(side="top", fill="both", expand=True, padx=10, pady=10)
entry.pack_propagate(0)

def generate():
    input_text = entry.get("1.0", "end")
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=64, min_length=16, top_p=1, do_sample=True, num_beams=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace("<n>", "\n")
    label.config(text=response)


def button_command():
    generate()

button = tk.Button(root, text="Summarize Text", width=50, command=button_command)
button.pack(side="top", fill="both", expand=True, padx=10, pady=10)
button.pack_propagate(0)

label = tk.Label(root, text="", borderwidth=2, relief="groove")
label.pack(side="top", fill="both", expand=True, padx=10, pady=10)
label.pack_propagate(0)

canvas = tk.Canvas(root, borderwidth=0, highlightthickness=0)
canvas.pack(side="top", fill="both", expand=True)
canvas.pack_propagate(0)

root.mainloop()
