import sys
import os
import json
import numpy as np
from numpy.linalg import norm
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QStackedWidget, QScrollArea, QMainWindow, QToolBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from screen.PlotScreen import PlotScreen
from screen.SearchTermScreen import SearchTermScreen
from screen.SearchDocsScreen import SearchDocsScreen
from screen.SearchTermDocsScreen import SearchTermDocsScreen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Big data Wikipedia")
        self.resize(800, 600)
        self.term_list = []
        self.term_dict = {}
        self.doc_list = []

        self.read_file()

        # Create stacked widget
        self.stacked = QStackedWidget()
        self.search_term_screen = SearchTermScreen(self.term_list, self.mV)
        self.search_doc_screen = SearchDocsScreen(self.mU, self.doc_list)
        self.search_termdoc_screen = SearchTermDocsScreen(self.mU, self.mV, self.doc_list, self.term_list)
        self.plot_screen = PlotScreen(self.term_dict, self.term_list, self.term_emb_data, self.doc_emb_data, self.topic_data)

        self.stacked.addWidget(self.plot_screen)
        self.stacked.addWidget(self.search_term_screen)
        self.stacked.addWidget(self.search_doc_screen)
        self.stacked.addWidget(self.search_termdoc_screen)

        self.setCentralWidget(self.stacked)

        # Toolbar to switch screens
        toolbar = QToolBar("Navigation")
        self.addToolBar(toolbar)

        plot_action = QAction("Plot", self)
        plot_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.plot_screen))

        search_term_action = QAction("Term Relevance", self)
        search_term_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.search_term_screen))

        search_doc_action = QAction("Doc Relevance", self)
        search_doc_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.search_doc_screen))

        search_termdoc_action = QAction("TermDoc Relevance", self)
        search_termdoc_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.search_termdoc_screen))


        toolbar.addAction(plot_action)
        toolbar.addAction(search_term_action)
        toolbar.addAction(search_doc_action)
        toolbar.addAction(search_termdoc_action)
    
    def read_file(self):
        def read_json_file(path):
            data = []
            with open(path, "r", encoding="utf-8") as infile:
                for line in infile:
                    data.append(json.loads(line))
            return data
        
        """Đọc embedding từ file"""
        base_path = Path(__file__).resolve().parents[1]
        base_path = base_path / "db"
        term_emb_path = base_path / "term_embeddings.json"
        doc_emb_path = base_path / "doc_embeddings.json"
        topic_path = base_path / "topics.json"


        term_emb_data = read_json_file(term_emb_path)
        doc_emb_data = read_json_file(doc_emb_path)
        topic_data = read_json_file(topic_path)
        
        self.term_emb_data = term_emb_data
        self.doc_emb_data = doc_emb_data
        self.topic_data = topic_data

        self.mV = np.zeros((len(term_emb_data), len(term_emb_data[0]["embedding"]))) # V matrix
        for index, item in enumerate(term_emb_data):
            if "term" in item and "embedding" in item:
                self.term_list.append(item["term"])
                self.term_dict[item["term"]] = np.array(item["embedding"])
                self.mV[index] = item["embedding"]

        self.mU = np.zeros((len(doc_emb_data), len(doc_emb_data[0]["embedding"]))) # V matrix
        for index, item in enumerate(doc_emb_data):
            if "embedding" in item:
                self.doc_list.append(item["title"].strip().lower())
                self.mU[index] = item["embedding"]


# ========== RUN ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
