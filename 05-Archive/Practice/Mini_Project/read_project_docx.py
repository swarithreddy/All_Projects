import docx
path = r"c:\Users\swarith reddy\OneDrive\Desktop\gtidemo\All_Projects\Mini_Project\documents\mini project.docx"
doc = docx.Document(path)
for para in doc.paragraphs:
    print(para.text)