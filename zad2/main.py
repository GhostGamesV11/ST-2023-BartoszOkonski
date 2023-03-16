import gradio as gr
import numpy as np

def read_line (file_name,line_number):
    with open(file_name, "r") as file:
        line = file.readlines()[line_number - 1]
    return line

def read_text_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def get_file_info(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    number_lines = len(lines)
    text_lines = read_text_file(file_name)
    info = f"File contains {number_lines} lines."
    classes = {}
    for line in text_lines:
        class_name = line.split()[0]
        if class_name not in classes:
            classes[class_name] = 0
        classes[class_name] += 1
    return classes, info, number_lines


def display_file(file_name, num_lines):
    classes, info , number_lines= get_file_info(file_name)
    response = "Liczba klas decyzyjnych: {}\n".format(len(classes))
    num_lines = int(num_lines)
    if num_lines <= 0:
        return "Invalid input: number smaller or equel to zero  ."
    elif number_lines > number_lines:
        return "Invalid input: number of lines is greater than file size."
    line = "Chciana linia: {}\n".format(read_line(file_name,num_lines))
    for class_name, class_size in classes.items():
        response += "Wielkość klasy {}: {}\n".format(class_name, class_size)
    return f"{info}\n{response}\n{line}"
#

#
file_name_input = gr.inputs.Textbox(label="Enter file name:")
num_lines_input = gr.inputs.Number(label="Number of lines to display:")
output_text = gr.outputs.Textbox(label="File preview:")

iface = gr.Interface(fn=display_file, inputs=[file_name_input, num_lines_input], outputs=output_text, title="File Preview Bot",
                     description="Enter a file name and the number of lines to display to see a preview of the file.")
iface.launch()

