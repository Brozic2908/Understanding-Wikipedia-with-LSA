import sys
import os
import json
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QPushButton, QLabel, QStackedWidget, QScrollArea, QMainWindow, QToolBar,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle

class SilhouetteWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list, int, list, list, list, list, object, object, object)

    def __init__(self, X, topics):
        super().__init__()
        self.X = X
        self.top_topics = topics

    def run(self):
        """Tính silhouette score trong background"""
        K = range(2, 15)
        scores = []
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X)
            score = silhouette_score(self.X, kmeans.labels_)
            scores.append(score)
        # best_k = K[int(np.argmax(scores))]
        best_k = 5
        """Tính độ mạnh topic trong background"""
        topics = []
        values = []
        for t in self.top_topics:
            topics.append(t["topic"])
            values.append(t["singular_value"])
        """Dùng kmean để chia cụm trong background"""
        X_norm = normalize(self.X)
        # X_norm = self.X
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_norm)
        centers = kmeans.cluster_centers_
        unique, counts = np.unique(labels, return_counts=True)

        self.finished.emit(scores, best_k, topics, values, unique, counts, X_norm, centers, labels)

class PlotScreen(QWidget):
    def __init__(self,term_dict, term_list, term_emb_data, doc_emb_data, topic_data):
        super().__init__()
        #init variabels
        self.term_dict = term_dict
        self.term_list = term_list
        self.term_emb_data = term_emb_data
        self.doc_emb_data = doc_emb_data
        self.topics_data = topic_data

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Label to notification
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFixedHeight(40) #40px
        self.status_label.setStyleSheet("color: red; font-weight: bold; margin-top: 8px;")
        layout.addWidget(self.status_label)


        # Create matplotlib figure
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        layout.addWidget(scroll)
        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(container)

        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.container_layout.addWidget(self.canvas)
        
        # Plot
        if self.term_dict:
            self.X = np.array(list(self.term_dict.values()))
            self.start_plot_thread(self.X, self.topics_data)


    def start_plot_thread(self, X, topics):
        """Chạy tính toán silhouette score trong thread riêng"""
        self.status_label.setText("Plotting ...")
        self.worker = SilhouetteWorker(X, topics)
        self.worker.finished.connect(self.plot)
        self.worker.start()

    def plot(self, scores, best_k, topics, values, unique, counts, X_norm, centers, labels):
        """Vẽ biểu đồ sau khi thread hoàn tất"""
        self.silhouette_scores = scores
        self.best_k = best_k
        self.strengthen_topics =  topics
        self.strengthen_values =  values
        self.kmean_unique = unique
        self.kmean_counts = counts
        self.X_norm = X_norm
        self.kmeans_centers = centers
        self.kmeans_labels = labels

        self.add_plot_section("📊 Biểu đồ 1: Silhouette Score", self.plot_silhouette)
        self.add_plot_section("", self.plot_kmeans, False)
        self.add_plot_section("", self.plot_kmeans_scatter, False)
        self.add_plot_section("📈 Biểu đồ 2: Topic Strength", self.plot_topics)

        self.status_label.setText(f"Plot completed")

    def add_plot_section(self, title: str, plot_func, with_label = True):
        """Tạo một section gồm label + canvas riêng"""
        # Label mô tả
        if with_label:
            label = QLabel(title)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setStyleSheet("font-weight: bold; font-size: 14px; color: #1a237e; margin-bottom: 6px;")
            self.container_layout.addWidget(label)

        # Figure & canvas
        figure = Figure(figsize=(6, 3))
        canvas = FigureCanvas(figure)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        canvas.setMinimumSize(400, 300)
        self.container_layout.addWidget(canvas)

        # Gọi hàm vẽ cụ thể
        plot_func(figure)


    def plot_silhouette(self, figure):
        """Vẽ biểu đồ Silhouette"""
        ax = figure.add_subplot(111)
        K = np.arange(2, 15)
        ax.plot(K, self.silhouette_scores, "bo-", linewidth=2)
        ax.set_xlabel("Số cụm (k)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Đánh giá số cụm KMeans bằng Silhouette Score")
        ax.grid(True, linestyle="--", alpha=0.6)
        figure.tight_layout()


    def plot_kmeans(self, figure):
        """Vẽ biểu đồ số phần tử trong mỗi phần"""
        ax2 = figure.add_subplot(111)
        bars = ax2.bar(self.kmean_unique, self.kmean_counts, color="#43a047", alpha=0.8)
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Số phần tử")
        ax2.set_title(f"Phân bố số phần tử trong {self.best_k} cụm")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
        figure.tight_layout()

        self.on_hover(figure, ax2, bars, self.kmean_unique)


    def plot_topics(self, figure):
        """Vẽ biểu đồ Topic Strength"""
        ax3 = figure.add_subplot(111)
        bars = ax3.bar(self.strengthen_topics, self.strengthen_values, color="#1f77b4", alpha=0.8)
        ax3.set_xlabel("Topic ID")
        ax3.set_ylabel("Singular Value (Topic Strength)")
        ax3.set_title("Topic Importance from SVD")
        ax3.grid(True, axis="y", linestyle="--", alpha=0.5)
        figure.tight_layout()

        self.on_hover(figure, ax3, bars, self.strengthen_topics)

    def on_hover(self, figure, ax, bars, cluster_ids):
        canvas = figure.canvas

        
        tooltip = QLabel(canvas)
        tooltip.setStyleSheet("""
            QLabel {
                background-color: rgba(40, 40, 40, 220);
                color: white;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
        """)
        tooltip.hide()

        def on_motion(event):
            if event.inaxes == ax and event.guiEvent:
                pos = event.guiEvent.position()
                hovered = False

                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        count = int(bar.get_height())
                        cluster_label = cluster_ids[i] if i < len(cluster_ids) else i

                        tooltip.setText(f"Cụm {cluster_label}: {count} từ")

                        
                        x = int(pos.x()) + 12
                        y = int(pos.y()) - 25
                        tooltip.move(x, y)

                        tooltip.show()
                        hovered = True
                        break

                if not hovered:
                    if tooltip.isVisible():
                        tooltip.hide()
                        canvas.draw_idle()  
            else:
                if tooltip.isVisible():
                    tooltip.hide()
                    canvas.draw_idle()  

        canvas.mpl_connect("motion_notify_event", on_motion)

    def plot_kmeans_scatter(self, figure):
        """Vẽ scatter các điểm + tâm cụm + vòng tròn KMeans"""
        pca = PCA(n_components=2)
        self.X2d = pca.fit_transform(self.X_norm)
        self.centers2d = pca.transform(self.kmeans_centers)


        ax = figure.add_subplot(111)

        X2d = self.X2d
        labels = self.kmeans_labels
        centers2d = self.centers2d
        k = self.best_k

        # Bảng màu
        colors = cm.get_cmap('tab10')(np.linspace(0, 1, k))

        # Vẽ điểm theo từng cụm
        for i in range(k):
            pts = X2d[labels == i]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=25, alpha=0.7,
                color=colors[i],
                label=f"Cluster {i}"
            )

        # Vẽ centroid
        ax.scatter(
            centers2d[:, 0], centers2d[:, 1],
            s=140, marker='X',
            color='black', label="Centroids"
        )

        # Vẽ vòng tròn bao cụm (bán kính = khoảng cách trung bình các điểm trong cụm)
        # for i in range(k):
        #     pts = X2d[labels == i]
        #     center = centers2d[i]

        #     if len(pts) == 0:
        #         continue

        #     dists = np.sqrt(np.sum((pts - center) ** 2, axis=1))
        #     radius = np.mean(dists)

        #     circle = Circle(
        #         center, radius,
        #         color=colors[i],
        #         alpha=0.18,
        #         linewidth=1.5,
        #         fill=True
        #     )
        #     ax.add_patch(circle)

        ax.set_title("Phân cụm KMeans (PCA 2D)")
        ax.set_xlabel("PCA Dimension 1")
        ax.set_ylabel("PCA Dimension 2")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)
        figure.tight_layout()