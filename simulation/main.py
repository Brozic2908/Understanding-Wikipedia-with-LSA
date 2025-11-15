import sys
import os
import json
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QStackedWidget, QScrollArea, QMainWindow, QToolBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from screen.PlotScreen import PlotScreen
from screen.SearchScreen import SearchScreen

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Big data Wikipedia")
        self.resize(800, 600)
        self.term_list = []
        self.term_dict = {}

        self.read_file()

        # Create stacked widget
        self.stacked = QStackedWidget()
        self.search_screen = SearchScreen(self.term_dict, self.term_list, self.term_emb_data, self.doc_emb_data, self.topic_data)
        self.plot_screen = PlotScreen(self.term_dict, self.term_list, self.term_emb_data, self.doc_emb_data, self.topic_data)

        self.stacked.addWidget(self.plot_screen)
        self.stacked.addWidget(self.search_screen)

        self.setCentralWidget(self.stacked)

        # Toolbar to switch screens
        toolbar = QToolBar("Navigation")
        self.addToolBar(toolbar)

        plot_action = QAction("Plot", self)
        plot_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.plot_screen))

        search_action = QAction("Search", self)
        search_action.triggered.connect(lambda: self.stacked.setCurrentWidget(self.search_screen))


        toolbar.addAction(plot_action)
        toolbar.addAction(search_action)
    
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

        for item in term_emb_data:
            if "term" in item and "embedding" in item:
                self.term_list.append(item["term"])
                self.term_dict[item["term"]] = np.array(item["embedding"])


# ========== RUN ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
