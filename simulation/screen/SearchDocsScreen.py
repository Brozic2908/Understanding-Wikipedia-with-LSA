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


class SearchDocsScreen(QWidget):
    def __init__(self, mU, doc_list):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        self.doc_list = doc_list
        self.mU = mU

        search_bar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter document title to get similar documents (at less 3 characters)...")
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        layout.addWidget(scroll)

        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(10, 10, 10, 10)
        self.container_layout.setSpacing(12)
        scroll.setWidget(container)

        self.container_layout.addStretch()


    def handle_search(self):
        result = self._get_search_text()

        if isinstance(result, list):
            self._show_doc_choices(result)
            return

        index, value = result
        if value is None:
            self._clear_results()
            self._add_error("Document not found.")
            return

        self._search_similar_documents(index, value)


    def _get_search_text(self):
        text = self.search_input.text().strip().lower()

        if not text or len(text) < 3:
            return None, None

        matches = [title for title in self.doc_list if text in title.lower()]

        if len(matches) == 0:
            return None, None

        if len(matches) > 1:
            return matches

        selected = matches[0]
        idx = self.doc_list.index(selected)
        return idx, self.mU[idx]


    def _show_doc_choices(self, matches):
        self._clear_results()

        label = QLabel("<b>Multiple documents found. Select one:</b>")
        label.setStyleSheet("font-size: 16px; margin-bottom: 2px;")
        self.container_layout.addWidget(label)

        for title in matches:
            btn = QPushButton(title)
            btn.setStyleSheet("padding: 6px; font-size: 15px; text-align: left;")
            btn.clicked.connect(lambda _, t=title: self._select_doc(t))
            self.container_layout.addWidget(btn)

        self.container_layout.addStretch()


    def _select_doc(self, title):
        index = self.doc_list.index(title)
        value = self.mU[index]
        self.search_input.setText(self.doc_list[index])
        self._search_similar_documents(index, value)


    def _search_similar_documents(self, index, value):
        self._clear_results()

        searching_label = QLabel("<i style='color: gray;'>Searching...</i>")
        searching_label.setStyleSheet("font-size: 16px; margin: 8px 0;")
        self.container_layout.insertWidget(0, searching_label)

        QApplication.processEvents()

        scores = []
        for i, v in enumerate(self.mU):
            if i == index:
                continue
            score = self._cosine(value, v)
            scores.append((score, self.doc_list[i]))

        scores.sort(reverse=True)

        self._clear_results()
        top_k = min(20, len(scores))

        header = QLabel("<b>Top related documents:</b>")
        header.setStyleSheet("font-size: 17px; margin-bottom: 12px;")
        self.container_layout.addWidget(header)

        for i in range(top_k):
            title = scores[i][1]
            title_label = QLabel(f'<b style="color:#1A0DAB; font-size:16px;">{title}</b>')
            title_label.setWordWrap(True)
            title_label.setStyleSheet("margin-bottom: 6px; padding: 4px;")
            self.container_layout.addWidget(title_label)

        self.container_layout.addStretch()


    def _clear_results(self):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            if item:
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        self.container_layout.addStretch()

    def _add_error(self, msg):
        label = QLabel(f"<b style='color:red;'>{msg}</b>")
        label.setStyleSheet("font-size: 15px; margin: 8px 0;")
        self.container_layout.insertWidget(0, label)

    def _cosine(self, a, b):
        na = norm(a)
        nb = norm(b)
        if na == 0 or nb == 0:
            return 0
        return float(np.dot(a, b) / (na * nb + 1e-9))
