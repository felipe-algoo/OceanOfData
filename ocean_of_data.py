#!/usr/bin/env python3
import sys
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import re
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QLineEdit, QComboBox,
    QTextEdit, QMessageBox, QProgressBar, QGroupBox, QSpinBox,
    QCheckBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QMutexLocker
import pyqtgraph as pg
import requests
from bs4 import BeautifulSoup
import pdfplumber
from docx import Document

NLTK_AVAILABLE = False
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except Exception:
    def sent_tokenize(text):
        return re.split(r'(?<=[.!?])\s+', text)
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text.lower())
    stopwords = None

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_TEXT_LENGTH = 500000
MAX_DOCUMENTS = 500
MAX_FEATURES = 5000
ALLOWED_EXTENSIONS = {'.csv', '.pdf', '.docx', '.txt'}
ALLOWED_MIME = {
    'text/csv', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'text/plain', 'application/octet-stream'
}
URL_TIMEOUT = 15
USER_AGENT = 'OceanOfData/1.0 (Research Tool; +https://localhost)'

LOG_DIR = Path(tempfile.gettempdir()) / 'ocean_of_data_logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'ocean.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('OceanOfData')

class SecureValidator:
    @staticmethod
    def sanitize_filename(name: str) -> str:
        name = os.path.basename(name)
        name = re.sub(r'[^\w\s\-\.]', '', name)
        return name[:200] if name else 'unnamed'

    @staticmethod
    def validate_extension(path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in ALLOWED_EXTENSIONS

    @staticmethod
    def validate_file_size(path: str) -> bool:
        try:
            return 0 < os.path.getsize(path) <= MAX_FILE_SIZE
        except OSError:
            return False

    @staticmethod
    def validate_url(url: str) -> bool:
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme not in ('http', 'https'):
                return False
            if not parsed.netloc or len(parsed.netloc) < 3:
                return False
            if any(x in parsed.netloc.lower() for x in ['localhost', '127.0.0.1', '0.0.0.0', '[::1]']):
                return False
            if re.search(r'[<>\"\'`]', url):
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def is_safe_text(text: str) -> bool:
        if not text or len(text) > MAX_TEXT_LENGTH:
            return False
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', text):
            return False
        return True

class TextExtractor:
    @staticmethod
    def from_csv(path: str) -> Tuple[List[str], Optional[np.ndarray]]:
        try:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', nrows=MAX_DOCUMENTS)
            if df.empty:
                return [], None
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            texts = []
            if text_cols:
                for _, row in df.iterrows():
                    parts = [str(row[c]) for c in text_cols if pd.notna(row[c])]
                    texts.append(' '.join(parts)[:2000])
            else:
                texts = [f"Row {i}" for i in range(len(df))]
            matrix = None
            if numeric_cols:
                matrix = df[numeric_cols].fillna(0).values.astype(np.float64)
                if matrix.shape[0] > MAX_DOCUMENTS:
                    matrix = matrix[:MAX_DOCUMENTS]
            return texts[:MAX_DOCUMENTS], matrix
        except Exception as e:
            logger.error(f"CSV extract error: {type(e).__name__}")
            return [], None

    @staticmethod
    def from_pdf(path: str) -> List[str]:
        texts = []
        try:
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i >= 50:
                        break
                    t = page.extract_text() or ''
                    if t.strip():
                        texts.append(t.strip()[:5000])
            return texts[:MAX_DOCUMENTS]
        except Exception as e:
            logger.error(f"PDF extract error: {type(e).__name__}")
            return []

    @staticmethod
    def from_docx(path: str) -> List[str]:
        texts = []
        try:
            doc = Document(path)
            current = []
            for para in doc.paragraphs:
                t = para.text.strip()
                if t:
                    current.append(t)
                    if len(' '.join(current)) > 2000:
                        texts.append(' '.join(current)[:5000])
                        current = []
            if current:
                texts.append(' '.join(current)[:5000])
            return texts[:MAX_DOCUMENTS]
        except Exception as e:
            logger.error(f"DOCX extract error: {type(e).__name__}")
            return []

    @staticmethod
    def from_url(url: str) -> List[str]:
        if not SecureValidator.validate_url(url):
            return []
        try:
            headers = {'User-Agent': USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=URL_TIMEOUT, allow_redirects=True, stream=True)
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                return []
            content = b''
            for chunk in resp.iter_content(8192):
                content += chunk
                if len(content) > MAX_FILE_SIZE:
                    break
            text = content.decode('utf-8', errors='replace')
            soup = BeautifulSoup(text, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            paragraphs = []
            for p in soup.find_all(['p', 'article', 'section', 'h1', 'h2', 'h3']):
                t = p.get_text(separator=' ', strip=True)
                if len(t) > 40:
                    paragraphs.append(t[:3000])
            if not paragraphs:
                body = soup.get_text(separator=' ', strip=True)
                paragraphs = [body[i:i+2000] for i in range(0, min(len(body), 20000), 2000)]
            return paragraphs[:MAX_DOCUMENTS]
        except Exception as e:
            logger.error(f"URL extract error: {type(e).__name__}")
            return []

class ContextAnalyzer:
    def __init__(self):
        basic_en = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they'
        }
        basic_pt = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'e', 'ou', 'mas', 'em', 'no',
            'na', 'nos', 'nas', 'de', 'do', 'da', 'dos', 'das', 'para', 'com', 'por', 'que',
            'se', 'ao', 'aos', 'à', 'às', 'é', 'são', 'foi', 'ser', 'estar', 'ter', 'haver',
            'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'isso', 'ele', 'ela', 'eles', 'elas'
        }
        self.stop_all = basic_en | basic_pt
        if NLTK_AVAILABLE and stopwords is not None:
            try:
                self.stop_all |= set(stopwords.words('english'))
            except Exception:
                pass
            try:
                self.stop_all |= set(stopwords.words('portuguese'))
            except Exception:
                pass

    def clean_text(self, text: str) -> str:
        if not SecureValidator.is_safe_text(text):
            return ''
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u00C0-\u017F.,!?;:\-]', '', text)
        return text.strip()[:5000]

    def tokenize_safe(self, text: str) -> List[str]:
        try:
            tokens = word_tokenize(text.lower())
            return [t for t in tokens if t.isalpha() and len(t) > 2 and t not in self.stop_all]
        except Exception:
            return []

    def extract_sentences(self, text: str) -> List[str]:
        try:
            sents = sent_tokenize(text)
            return [s.strip() for s in sents if 20 < len(s.strip()) < 500][:100]
        except Exception:
            return [text[:500]] if text else []

    def compute_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        cleaned = [self.clean_text(t) for t in texts]
        cleaned = [c for c in cleaned if c]
        if not cleaned:
            return np.array([])
        try:
            vectorizer = TfidfVectorizer(
                max_features=min(MAX_FEATURES, 2000),
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
            matrix = vectorizer.fit_transform(cleaned)
            return matrix.toarray().astype(np.float64)
        except Exception as e:
            logger.error(f"TFIDF error: {type(e).__name__}")
            return np.array([])

    def reduce_false_positives(self, embeddings: np.ndarray, labels: np.ndarray, texts: List[str]) -> np.ndarray:
        if embeddings.size == 0 or len(labels) == 0:
            return labels
        try:
            unique = np.unique(labels)
            if len(unique) < 2:
                return labels
            refined = labels.copy()
            for lab in unique:
                if lab == -1:
                    continue
                mask = labels == lab
                if mask.sum() < 3:
                    continue
                cluster_emb = embeddings[mask]
                centroid = cluster_emb.mean(axis=0)
                dists = np.linalg.norm(cluster_emb - centroid, axis=1)
                threshold = np.percentile(dists, 85)
                outliers = dists > threshold
                local_idx = np.where(mask)[0]
                for i, is_out in enumerate(outliers):
                    if is_out:
                        refined[local_idx[i]] = -1
            return refined
        except Exception as e:
            logger.error(f"Refine error: {type(e).__name__}")
            return labels

class DimensionalityReducer:
    @staticmethod
    def apply_pca(data: np.ndarray, n_components: int = 2) -> np.ndarray:
        if data.shape[0] < 2 or data.shape[1] < 1:
            return np.zeros((data.shape[0], 2))
        n_comp = min(n_components, data.shape[0] - 1, data.shape[1])
        if n_comp < 1:
            return np.zeros((data.shape[0], 2))
        try:
            pca = PCA(n_components=n_comp, random_state=42)
            result = pca.fit_transform(data)
            if result.shape[1] < 2:
                pad = np.zeros((result.shape[0], 2 - result.shape[1]))
                result = np.hstack([result, pad])
            return result
        except Exception as e:
            logger.error(f"PCA error: {type(e).__name__}")
            return np.zeros((data.shape[0], 2))

    @staticmethod
    def apply_tsne(data: np.ndarray, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        n = data.shape[0]
        if n < 5:
            return DimensionalityReducer.apply_pca(data, n_components)
        try:
            n_comp = min(n_components, 2)
            max_perp = max(1.0, (n - 1) / 3.0)
            perp = min(float(perplexity), max_perp)
            if perp < 1.0:
                perp = 1.0
            tsne = TSNE(
                n_components=n_comp,
                perplexity=perp,
                random_state=42,
                init='pca',
                learning_rate='auto',
                max_iter=750,
                n_jobs=1
            )
            return tsne.fit_transform(data)
        except Exception as e:
            logger.error(f"tSNE error: {type(e).__name__}")
            return DimensionalityReducer.apply_pca(data, n_components)

class ClusterEngine:
    @staticmethod
    def kmeans(data: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        if data.shape[0] < 2:
            return np.zeros(data.shape[0], dtype=int)
        n_clust = min(n_clusters, data.shape[0])
        try:
            km = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
            return km.fit_predict(data)
        except Exception as e:
            logger.error(f"KMeans error: {type(e).__name__}")
            return np.zeros(data.shape[0], dtype=int)

    @staticmethod
    def dbscan(data: np.ndarray, eps: float = 0.5, min_samples: int = 3) -> np.ndarray:
        if data.shape[0] < 2:
            return np.zeros(data.shape[0], dtype=int)
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
            return db.fit_predict(data)
        except Exception as e:
            logger.error(f"DBSCAN error: {type(e).__name__}")
            return np.zeros(data.shape[0], dtype=int)

    @staticmethod
    def auto_cluster(data: np.ndarray, max_k: int = 10) -> Tuple[np.ndarray, int]:
        if data.shape[0] < 3:
            return np.zeros(data.shape[0], dtype=int), 1
        best_labels = np.zeros(data.shape[0], dtype=int)
        best_score = -1.0
        best_k = 2
        max_k = min(max_k, data.shape[0] // 2)
        for k in range(2, max_k + 1):
            try:
                labels = ClusterEngine.kmeans(data, k)
                if len(np.unique(labels)) < 2:
                    continue
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_k = k
            except Exception:
                continue
        return best_labels, best_k

class NeighborSearch:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.nn = None
        if data.shape[0] >= 2:
            try:
                self.nn = NearestNeighbors(n_neighbors=min(6, data.shape[0]), algorithm='auto', metric='euclidean')
                self.nn.fit(data)
            except Exception as e:
                logger.error(f"NN init error: {type(e).__name__}")

    def find_neighbors(self, idx: int, k: int = 5) -> List[int]:
        if self.nn is None or idx < 0 or idx >= self.data.shape[0]:
            return []
        try:
            dists, indices = self.nn.kneighbors([self.data[idx]], n_neighbors=min(k + 1, self.data.shape[0]))
            return [int(i) for i in indices[0] if i != idx][:k]
        except Exception:
            return []

class WorkerThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, mode: str, source: str, method: str, n_clusters: int, use_tsne: bool):
        super().__init__()
        self.mode = mode
        self.source = source
        self.method = method
        self.n_clusters = n_clusters
        self.use_tsne = use_tsne
        self.mutex = QMutex()
        self._stop = False

    def stop(self):
        with QMutexLocker(self.mutex):
            self._stop = True

    def run(self):
        try:
            self.progress.emit(5, "Validating input...")
            texts = []
            numeric = None
            if self.mode == 'file':
                if not SecureValidator.validate_extension(self.source):
                    self.error.emit("Invalid file type. Allowed: CSV, PDF, DOCX, TXT")
                    return
                if not SecureValidator.validate_file_size(self.source):
                    self.error.emit("File size exceeds limit or is empty")
                    return
                ext = Path(self.source).suffix.lower()
                self.progress.emit(15, "Extracting content...")
                if ext == '.csv':
                    texts, numeric = TextExtractor.from_csv(self.source)
                elif ext == '.pdf':
                    texts = TextExtractor.from_pdf(self.source)
                elif ext == '.docx':
                    texts = TextExtractor.from_docx(self.source)
                elif ext == '.txt':
                    with open(self.source, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(MAX_TEXT_LENGTH)
                    texts = [content[i:i+2000] for i in range(0, len(content), 2000)][:MAX_DOCUMENTS]
            elif self.mode == 'url':
                if not SecureValidator.validate_url(self.source):
                    self.error.emit("Invalid or unsafe URL")
                    return
                self.progress.emit(15, "Fetching URL content...")
                texts = TextExtractor.from_url(self.source)
            else:
                self.error.emit("Unknown mode")
                return

            if not texts and numeric is None:
                self.error.emit("No extractable content found")
                return

            analyzer = ContextAnalyzer()
            self.progress.emit(30, "Analyzing context and building features...")
            if numeric is not None and numeric.shape[0] > 0 and numeric.shape[1] > 0:
                embeddings = numeric
                if embeddings.shape[0] != len(texts) and texts:
                    texts = texts[:embeddings.shape[0]]
                elif not texts:
                    texts = [f"Item {i}" for i in range(embeddings.shape[0])]
            else:
                embeddings = analyzer.compute_tfidf_embeddings(texts)
                if embeddings.size == 0:
                    self.error.emit("Failed to generate embeddings")
                    return

            if embeddings.shape[0] < 2:
                self.error.emit("Insufficient data points for analysis")
                return

            self.progress.emit(50, "Scaling and reducing dimensions...")
            scaler = StandardScaler()
            try:
                scaled = scaler.fit_transform(embeddings)
            except Exception:
                scaled = embeddings

            if self.use_tsne and scaled.shape[0] >= 5:
                coords = DimensionalityReducer.apply_tsne(scaled, 2)
            else:
                coords = DimensionalityReducer.apply_pca(scaled, 2)

            self.progress.emit(70, "Clustering data...")
            if self.method == 'kmeans':
                labels = ClusterEngine.kmeans(scaled, self.n_clusters)
            elif self.method == 'dbscan':
                labels = ClusterEngine.dbscan(scaled)
            else:
                labels, _ = ClusterEngine.auto_cluster(scaled, self.n_clusters)

            labels = analyzer.reduce_false_positives(scaled, labels, texts)

            self.progress.emit(85, "Computing nearest neighbors...")
            nn_search = NeighborSearch(scaled)
            neighbors = {}
            for i in range(min(len(texts), 100)):
                neighbors[i] = nn_search.find_neighbors(i, 5)

            self.progress.emit(95, "Finalizing...")
            result = {
                'coords': coords,
                'labels': labels,
                'texts': texts[:len(coords)],
                'neighbors': neighbors,
                'n_clusters': len(np.unique(labels[labels >= 0])),
                'n_outliers': int(np.sum(labels == -1))
            }
            self.progress.emit(100, "Done")
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Worker error: {type(e).__name__}: {str(e)[:200]}")
            self.error.emit("Processing failed due to an internal error. Check logs.")

class OceanOfDataApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ocean of Data")
        self.setMinimumSize(1100, 700)
        self.worker = None
        self.current_data = None
        self.init_ui()
        self.setStyleSheet("""
            QMainWindow { background-color: #0a1628; }
            QLabel { color: #e0f0ff; }
            QPushButton {
                background-color: #1a3a5c; color: #e0f0ff; border: 1px solid #2a5a8c;
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2a5a8c; }
            QPushButton:disabled { background-color: #0a2030; color: #608090; }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #0d2137; color: #e0f0ff; border: 1px solid #2a5a8c;
                padding: 6px; border-radius: 3px;
            }
            QTextEdit, QTableWidget {
                background-color: #0d2137; color: #c0d8f0; border: 1px solid #2a5a8c;
            }
            QGroupBox {
                color: #80c0ff; border: 1px solid #2a5a8c; border-radius: 5px;
                margin-top: 10px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QProgressBar {
                border: 1px solid #2a5a8c; border-radius: 3px; text-align: center;
                background-color: #0d2137; color: #e0f0ff;
            }
            QProgressBar::chunk { background-color: #2a8ac0; }
            QStatusBar { background-color: #0a1628; color: #80c0ff; }
            QTabWidget::pane { border: 1px solid #2a5a8c; }
            QTabBar::tab {
                background-color: #1a3a5c; color: #e0f0ff; padding: 8px 16px;
                border: 1px solid #2a5a8c; border-bottom: none;
            }
            QTabBar::tab:selected { background-color: #2a5a8c; }
        """)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(340)

        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)

        self.file_btn = QPushButton("Upload File (CSV / PDF / DOCX)")
        self.file_btn.clicked.connect(self.select_file)
        source_layout.addWidget(self.file_btn)

        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        source_layout.addWidget(self.file_label)

        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://example.com/news-article")
        url_layout.addWidget(self.url_input)
        self.url_btn = QPushButton("Load URL")
        self.url_btn.clicked.connect(self.load_url)
        url_layout.addWidget(self.url_btn)
        source_layout.addLayout(url_layout)

        left_layout.addWidget(source_group)

        algo_group = QGroupBox("Algorithms")
        algo_layout = QVBoxLayout(algo_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems(["Auto (Silhouette)", "K-Means", "DBSCAN"])
        algo_layout.addWidget(QLabel("Clustering:"))
        algo_layout.addWidget(self.method_combo)

        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 30)
        self.cluster_spin.setValue(5)
        algo_layout.addWidget(QLabel("Max / Target Clusters:"))
        algo_layout.addWidget(self.cluster_spin)

        self.tsne_check = QCheckBox("Use t-SNE (else PCA)")
        self.tsne_check.setChecked(True)
        algo_layout.addWidget(self.tsne_check)

        left_layout.addWidget(algo_group)

        self.run_btn = QPushButton("Explore Ocean")
        self.run_btn.clicked.connect(self.start_analysis)
        self.run_btn.setEnabled(False)
        left_layout.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        info_group = QGroupBox("Landscape Info")
        info_layout = QVBoxLayout(info_group)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        left_layout.addWidget(info_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        right_panel = QTabWidget()

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0b1420')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color='#6a9abf', width=1))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color='#6a9abf', width=1))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color='#a0c8e8'))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color='#a0c8e8'))
        self.plot_widget.setLabel('left', 'Embedding Dimension 2', color='#c0d8f0')
        self.plot_widget.setLabel('bottom', 'Embedding Dimension 1', color='#c0d8f0')
        self.plot_widget.setTitle(
            "Clusters as Islands · k-NN Bridges · Outliers Isolated",
            color='#e0f0ff', size='12pt'
        )
        self.plot_widget.setAspectLocked(False)
        self.edge_item = pg.PlotDataItem(pen=pg.mkPen(color=(100, 140, 180, 60), width=0.8))
        self.plot_widget.addItem(self.edge_item)
        self.centroid_scatter = pg.ScatterPlotItem(
            size=18, symbol='s', pen=pg.mkPen(color='w', width=1.5),
            brush=pg.mkBrush(255, 255, 255, 40)
        )
        self.plot_widget.addItem(self.centroid_scatter)
        self.scatter = pg.ScatterPlotItem(
            size=11, pen=pg.mkPen(width=0.6, color=(255, 255, 255, 80)),
            hoverable=True, tip=None
        )
        self.plot_widget.addItem(self.scatter)
        self.scatter.sigClicked.connect(self.on_point_clicked)
        self.legend_label = QLabel("")
        self.legend_label.setStyleSheet(
            "color: #b0d0f0; background-color: #0d2137; padding: 6px; border: 1px solid #2a5a8c; font-size: 11px;"
        )
        self.legend_label.setWordWrap(True)
        plot_layout.addWidget(self.plot_widget, stretch=1)
        plot_layout.addWidget(self.legend_label)
        right_panel.addTab(plot_widget, "Landscape View")

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(3)
        self.detail_table.setHorizontalHeaderLabels(["Index", "Cluster", "Preview"])
        self.detail_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.detail_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.detail_table.itemSelectionChanged.connect(self.on_table_select)
        detail_layout.addWidget(self.detail_table)
        right_panel.addTab(detail_widget, "Data Points")

        main_layout.addWidget(right_panel, stretch=1)

        self.statusBar().showMessage("Bilingual PT/EN ")

        self.selected_file = None
        self.selected_url = None
        self.mode = None

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Data File", "",
            "Supported (*.csv *.pdf *.docx *.txt);;CSV (*.csv);;PDF (*.pdf);;DOCX (*.docx);;Text (*.txt)"
        )
        if path:
            if not SecureValidator.validate_extension(path):
                self.show_dark_dialog("Invalid", "Unsupported file type.", QMessageBox.Warning)
                return
            if not SecureValidator.validate_file_size(path):
                self.show_dark_dialog("Invalid", "File too large or empty (max 10MB).", QMessageBox.Warning)
                return
            self.selected_file = path
            self.selected_url = None
            self.mode = 'file'
            self.file_label.setText(SecureValidator.sanitize_filename(path))
            self.run_btn.setEnabled(True)
            self.status_label.setText("File ready for analysis")

    def load_url(self):
        url = self.url_input.text().strip()
        if not SecureValidator.validate_url(url):
            self.show_dark_dialog("Invalid URL", "Provide a valid public http/https URL.", QMessageBox.Warning)
            return
        self.selected_url = url
        self.selected_file = None
        self.mode = 'url'
        self.file_label.setText(f"URL: {url[:60]}...")
        self.run_btn.setEnabled(True)
        self.status_label.setText("URL ready for analysis")

    def start_analysis(self):
        if self.worker and self.worker.isRunning():
            return
        source = self.selected_file if self.mode == 'file' else self.selected_url
        if not source:
            return
        method_map = {0: 'auto', 1: 'kmeans', 2: 'dbscan'}
        method = method_map.get(self.method_combo.currentIndex(), 'auto')
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_label.setText("Processing...")
        self.info_text.clear()
        self.worker = WorkerThread(
            self.mode, source, method,
            self.cluster_spin.value(),
            self.tsne_check.isChecked()
        )
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.on_progress)
        self.worker.start()

    def on_progress(self, value: int, msg: str):
        self.progress.setValue(value)
        self.status_label.setText(msg)

    def on_error(self, msg: str):
        self.run_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status_label.setText("Error")
        self.show_dark_dialog("Error", msg, QMessageBox.Critical)
        logger.warning(f"User-facing error: {msg}")

    def on_finished(self, result: dict):
        self.run_btn.setEnabled(True)
        self.current_data = result
        coords = result['coords']
        labels = result['labels']
        texts = result['texts']
        n_clust = result['n_clusters']
        n_out = result['n_outliers']
        neighbors = result.get('neighbors', {})

        scientific_cmap = [
            (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
            (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
            (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120)
        ]

        spots = []
        for i in range(len(coords)):
            lab = int(labels[i])
            if lab == -1:
                brush = pg.mkBrush(160, 160, 160, 140)
                symbol = 't'
                size = 9
            else:
                c = scientific_cmap[lab % len(scientific_cmap)]
                brush = pg.mkBrush(c[0], c[1], c[2], 210)
                symbol = 'o'
                size = 11
            spots.append({
                'pos': (float(coords[i, 0]), float(coords[i, 1])),
                'brush': brush,
                'symbol': symbol,
                'size': size,
                'data': i
            })
        self.scatter.setData(spots)

        edge_x = []
        edge_y = []
        drawn = set()
        for i, neighs in neighbors.items():
            if i >= len(coords):
                continue
            for j in neighs:
                if j >= len(coords):
                    continue
                key = (min(i, j), max(i, j))
                if key in drawn:
                    continue
                drawn.add(key)
                edge_x.extend([float(coords[i, 0]), float(coords[j, 0]), np.nan])
                edge_y.extend([float(coords[i, 1]), float(coords[j, 1]), np.nan])
        if edge_x:
            self.edge_item.setData(x=edge_x, y=edge_y)
        else:
            self.edge_item.setData(x=[], y=[])

        centroids = []
        centroid_labels = []
        unique_labs = sorted(set(int(l) for l in labels if l >= 0))
        for lab in unique_labs:
            mask = labels == lab
            if mask.sum() == 0:
                continue
            cx = float(np.mean(coords[mask, 0]))
            cy = float(np.mean(coords[mask, 1]))
            c = scientific_cmap[lab % len(scientific_cmap)]
            centroids.append({
                'pos': (cx, cy),
                'brush': pg.mkBrush(c[0], c[1], c[2], 90),
                'symbol': 's',
                'size': 16,
                'data': -1 - lab
            })
            centroid_labels.append(f"C{lab}")
        self.centroid_scatter.setData(centroids)

        legend_parts = []
        for lab in unique_labs:
            c = scientific_cmap[lab % len(scientific_cmap)]
            legend_parts.append(
                f'<span style="color:rgb({c[0]},{c[1]},{c[2]})">■</span> Island {lab}'
            )
        if n_out > 0:
            legend_parts.append('<span style="color:rgb(160,160,160)">▲</span> Outliers')
        legend_parts.append(f'<span style="color:#6a9abf">─</span> k-NN bridges ({len(drawn)})')
        self.legend_label.setText("  ·  ".join(legend_parts) if legend_parts else "No clusters")

        method = 't-SNE' if self.tsne_check.isChecked() else 'PCA'
        self.plot_widget.setTitle(
            f"Scientific Data Landscape  |  {method}  |  {self.method_combo.currentText()}  |  "
            f"n={len(texts)}  islands={n_clust}  outliers={n_out}",
            color='#e0f0ff', size='11pt'
        )

        self.plot_widget.autoRange()
        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None)
            r = vb.viewRange()
            pad_x = (r[0][1] - r[0][0]) * 0.08
            pad_y = (r[1][1] - r[1][0]) * 0.08
            vb.setRange(xRange=(r[0][0] - pad_x, r[0][1] + pad_x),
                        yRange=(r[1][0] - pad_y, r[1][1] + pad_y), padding=0)

        self.detail_table.setRowCount(len(texts))
        for i, t in enumerate(texts):
            self.detail_table.setItem(i, 0, QTableWidgetItem(str(i)))
            lab = labels[i]
            lab_str = "Outlier" if lab == -1 else f"Island {lab}"
            self.detail_table.setItem(i, 1, QTableWidgetItem(lab_str))
            preview = t[:120].replace('\n', ' ') + ('...' if len(t) > 120 else '')
            self.detail_table.setItem(i, 2, QTableWidgetItem(preview))

        info = (
            f"Observations: {len(texts)}\n"
            f"Islands (clusters): {n_clust}\n"
            f"Isolated (outliers): {n_out}\n"
            f"Embedding: {method}\n"
            f"Clustering: {self.method_combo.currentText()}\n"
            f"k-NN bridges drawn: {len(drawn)}\n"
            f"False-positive refinement: active (centroid distance)"
        )
        self.info_text.setText(info)
        self.status_label.setText("Exploration complete")
        self.statusBar().showMessage(
            f"Landscape ready · {len(texts)} pts · {n_clust} islands · {n_out} outliers · {len(drawn)} bridges"
        )

    def show_dark_dialog(self, title: str, body: str, icon=None):
        if icon is None:
            icon = QMessageBox.Information
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setTextFormat(Qt.PlainText)
        dlg.setText(body)
        dlg.setIcon(icon)
        dlg.setStandardButtons(QMessageBox.Ok)
        dlg.setStyleSheet("""
            QMessageBox {
                background-color: #0d2137;
                color: #e0f0ff;
            }
            QMessageBox QLabel {
                color: #e0f0ff;
                background-color: #0d2137;
                min-width: 420px;
                font-size: 12px;
            }
            QMessageBox QPushButton {
                background-color: #1a3a5c;
                color: #e0f0ff;
                border: 1px solid #2a5a8c;
                padding: 6px 18px;
                border-radius: 4px;
                min-width: 80px;
            }
            QMessageBox QPushButton:hover {
                background-color: #2a5a8c;
            }
        """)
        dlg.exec_()

    def on_point_clicked(self, plot, points):
        try:
            if points is None or self.current_data is None:
                return
            if hasattr(points, '__len__') and len(points) == 0:
                return
            point = points[0] if hasattr(points, '__getitem__') else points
            idx = point.data() if hasattr(point, 'data') else None
            if idx is None:
                return
            try:
                idx = int(idx)
            except (TypeError, ValueError):
                return
            texts = self.current_data.get('texts', [])
            labels = self.current_data.get('labels', [])
            neighbors = self.current_data.get('neighbors', {})
            if idx < 0 or idx >= len(texts):
                return
            lab = int(labels[idx]) if idx < len(labels) else -1
            lab_str = "Outlier (isolated object)" if lab == -1 else f"Island {lab}"
            neigh = neighbors.get(idx, [])
            neigh_str = ", ".join(str(n) for n in neigh) if neigh else "None"
            text_preview = texts[idx][:1200] if texts[idx] else "(empty)"
            body = (
                f"Index: {idx}\n"
                f"Cluster: {lab_str}\n"
                f"Nearest neighbors (bridges): {neigh_str}\n"
                f"{'─' * 40}\n"
                f"{text_preview}"
            )
            self.show_dark_dialog(f"Point {idx} — Point Detail", body)
        except Exception as e:
            logger.error(f"Point click error: {type(e).__name__}")

    def on_table_select(self):
        try:
            rows = self.detail_table.selectionModel().selectedRows()
            if not rows or self.current_data is None:
                return
            idx = rows[0].row()
            texts = self.current_data.get('texts', [])
            labels = self.current_data.get('labels', [])
            neighbors = self.current_data.get('neighbors', {})
            if idx < 0 or idx >= len(texts):
                return
            lab = int(labels[idx]) if idx < len(labels) else -1
            lab_str = "Outlier (isolated object)" if lab == -1 else f"Island {lab}"
            neigh = neighbors.get(idx, [])
            neigh_str = ", ".join(str(n) for n in neigh) if neigh else "None"
            text_preview = texts[idx][:1500] if texts[idx] else "(empty)"
            body = (
                f"Index: {idx}\n"
                f"Cluster: {lab_str}\n"
                f"Nearest neighbors (bridges): {neigh_str}\n"
                f"{'─' * 40}\n"
                f"{text_preview}"
            )
            self.show_dark_dialog(f"Point {idx} — Scientific Detail", body)
        except Exception as e:
            logger.error(f"Table select error: {type(e).__name__}")

def main():
    if sys.platform.startswith('win'):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('OceanOfData.1.0')
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setApplicationName("Ocean of Data")
    app.setOrganizationName("OceanOfData")
    window = OceanOfDataApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
