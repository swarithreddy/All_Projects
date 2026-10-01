# PDF Merger

A small Tkinter desktop application for selecting PDF files, combining them with `pypdf`, and saving the result to a chosen path.

## Requirements

- Python 3
- Tkinter (included with many Python installations; on some Linux distributions it is a separate package)
- `pypdf`

Install the Python dependency:

```bash
python -m pip install pypdf
```

## Run

From this project directory:

```bash
python main.py
```

Select PDFs in the file dialog, then choose **Merge PDFs** and a destination filename. The selected PDF paths are passed to `PdfWriter` in the order shown in the list.

## Limitations

- The app does not reorder or edit pages individually.
- Input files must be readable PDFs. Keep backups of important documents before merging.
- No automated tests are configured.