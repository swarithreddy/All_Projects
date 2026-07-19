import docx
path = r"c:\Users\swarith reddy\OneDrive\Desktop\gtidemo\All_Projects\DBMS Project\DBMS_Project.docx"
doc = docx.Document(path)
for para in doc.paragraphs:
    print(para.text)