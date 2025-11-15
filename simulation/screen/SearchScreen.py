import os
import sys
import json
import numpy as np
from numpy.linalg import norm

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QScrollArea
)
from PyQt6.QtCore import Qt


class SearchScreen(QWidget):
    def __init__(self, term_dict, term_list, term_emb_data, doc_emb_data, topic_data):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        # Init data
        self.term_dict = term_dict
        self.term_list = term_list
        self.term_emb_data = term_emb_data
        self.doc_emb_data = doc_emb_data
        self.topics_data = topic_data

        # --- Search bar ---
        search_bar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter search term...")
        self.search_input.setFixedHeight(35)
        self.search_input.setStyleSheet("""
            QLineEdit { 
                border: 1px solid #aaa; 
                border-radius: 10px; 
                padding: 0 12px;
                font-size: 15px;
            }
        """)

        search_button = QPushButton("Search")
        search_button.setFixedHeight(35)
        search_button.setStyleSheet("""
            QPushButton { 
                background: #1A73E8; 
                color: white; 
                border-radius: 10px; 
                padding: 0 16px; 
                font-size: 15px;
            }
        """)
        search_button.clicked.connect(self.handle_search)

        search_bar.addWidget(self.search_input)
        search_bar.addWidget(search_button)
        layout.addLayout(search_bar)

        # --- Scroll area ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        layout.addWidget(scroll)

        # --- Results container ---
        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_layout.setSpacing(12)  
        scroll.setWidget(container)

        self.container_layout.addStretch()


    def handle_search(self):
        vector = self._get_search_text()
        if vector is None:
            return

        self._clear_results()

        searching_label = QLabel("<i style='color: gray;'>Searching...</i>")
        searching_label.setStyleSheet("font-size: 16px; margin: 8px 0;")
        self.container_layout.insertWidget(0, searching_label)

        QApplication.processEvents()  

        # Compute similarity
        scores = []
        for doc in self.doc_emb_data:
            title = doc["title"]
            value = doc["embedding"]
            score = self._cosine(vector, value)
            scores.append((score, title))

        scores.sort(reverse=True)

        self._clear_results()

        for i in range(min(20, len(scores))):
            title = scores[i][1]
            wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            title_label = QLabel(f'<a href="{wiki_url}"><b style="color:#1A0DAB; font-size:16px;">{title}</b></a>')
            title_label.setOpenExternalLinks(True)
            title_label.setWordWrap(True)
            title_label.setStyleSheet("margin-bottom: 6px; padding: 4px;")

            self.container_layout.addWidget(title_label)

        self.container_layout.addStretch()


    def _get_search_text(self):
        text = self.search_input.text().strip().lower()

        vectors = []
        for t in text.split():
            if t in self.term_list:
                vectors.append(self.term_dict[t])

        if not vectors:
            return None

        return np.mean(vectors, axis=0)


    def _clear_results(self):
        # Remove all items (widgets + stretch)
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()

        # Add stretch again
        self.container_layout.addStretch()


    def _cosine(self, a, b):
        na = norm(a)
        nb = norm(b)
        if na == 0 or nb == 0:
            return 0
        return float(np.dot(a, b) / (na * nb + 1e-9))
