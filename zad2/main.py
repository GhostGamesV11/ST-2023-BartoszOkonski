import gradio as gr



def get_file_info(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    num_lines = len(lines)
    text_lines = read_text_file(file_name)
    info = f"File contains {num_lines} lines."
    classes = {}
    for line in text_lines:
        class_name = line.split()[0]
        if class_name not in classes:
            classes[class_name] = 0
        classes[class_name] += 1
    return classes, info

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]


def display_file(file_name, num_lines):
    classes, info = get_file_info(file_name)
    response = "Liczba klas decyzyjnych: {}\n".format(len(classes))
    # if num_lines > len(first_few_lines.split('\n')):
    #     return "Invalid input: number of lines is greater than file size."
    for class_name, class_size in classes.items():
        response += "Wielkość klasy {}: {}\n".format(class_name, class_size)
    return f"{info}\n{response}"
#
#
#
file_name_input = gr.inputs.Textbox(label="Enter file name:")
num_lines_input = gr.inputs.Number(label="Number of lines to display:")
output_text = gr.outputs.Textbox(label="File preview:")

iface = gr.Interface(fn=display_file, inputs=[file_name_input, num_lines_input], outputs=output_text, title="File Preview Bot",
                     description="Enter a file name and the number of lines to display to see a preview of the file.")
iface.launch()





