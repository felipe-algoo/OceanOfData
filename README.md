### `Overview` 

- Upload a CSV, PDF, DOCX or TXT file, or paste a public news URL. 
- The application extracts content, builds TF-IDF features, reduces dimensionality with PCA or t-SNE, clusters the points, and draws the result as a navigable scatter plot. 
- Clusters appear as colored islands, nearest-neighbor relations as faint bridges, and outliers as isolated triangles.

<img width="1410" height="807" alt="frontData" src="https://github.com/user-attachments/assets/9ed244b4-b787-41ba-b7c0-cd3a3ea8a890" />


### `Requirements`

- Python 3.10 or later
- Windows or Linux
- Packages: numpy, pandas, scikit-learn, PyQt5, pyqtgraph, requests, beautifulsoup4, pdfplumber, python-docx

            pip install numpy pandas scikit-learn PyQt5 pyqtgraph requests beautifulsoup4 pdfplumber python-docx

- On first run the application may download small NLTK resources if they are missing. 
- If that fails it falls back to a basic tokeniser and a hardcoded English/Portuguese stop word list.

### `Usage`

      python ocean_of_data.py

- Click Upload File and select a CSV, PDF, DOCX or TXT (maximum 10 MB), or paste a public http/https URL and click Load URL.
- Choose the clustering method (Auto via silhouette score, K-Means, or DBSCAN) and the maximum number of clusters.
- Optionally enable t-SNE instead of PCA.
- Click Explore Ocean.

### `Limitations`

- The tool has some limitations, which will be addressed as the tool is improved:

- The visualisation is strictly 2D.
- Very large documents are truncated (roughly 500 text chunks or 50 PDF pages).
- t-SNE on more than a few hundred points can be slow.
- The application does not embed models with sentence transformers; it relies on classical TF-IDF.
