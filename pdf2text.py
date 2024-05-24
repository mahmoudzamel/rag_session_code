import os
import textract

PDF_FILES_FOLDER = "./pdfs"
TEXT_FILES_FOLDER = "./textfiles"

if __name__ == '__main__':
    dir_files = os.listdir(PDF_FILES_FOLDER)
    for file in dir_files:
        split_tup = os.path.splitext(file)
        text_file = split_tup[0] + ".txt"
        doc = textract.process(os.path.join(PDF_FILES_FOLDER, file))
        with open(os.path.join(TEXT_FILES_FOLDER, text_file), 'w') as f:
            f.write(doc.decode('utf-8'))
