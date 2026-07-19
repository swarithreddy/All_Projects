import docx
path = r"c:\Users\swarith reddy\OneDrive\Desktop\gtidemo\All_Projects\data structures  in c\DATA STRUCTURE PROJECT.docx"
doc = docx.Document(path)
for para in doc.paragraphs:
    print(para.text)