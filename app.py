import os
# Ensure Qt loads correct platform plugin path
from PyQt5.QtCore import QLibraryInfo
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

import sys
import json
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# Optional InsightFace for image import feature
try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

class LandmarkItem(QtWidgets.QGraphicsEllipseItem):
    """Draggable landmark point"""
    def __init__(self, index, x, y, radius=4, *args, **kwargs):
        super().__init__(-radius/2, -radius/2, radius, radius, *args, **kwargs)
        self.setBrush(QtGui.QBrush(QtCore.Qt.red))
        self.setPen(QtGui.QPen(QtCore.Qt.black))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setPos(x, y)
        self.index = index

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            newPos = value
            scene = self.scene()
            if scene is not None:
                rect = scene.sceneRect()
                x = max(rect.left(), min(rect.right(), newPos.x()))
                y = max(rect.top(), min(rect.bottom(), newPos.y()))
                return QtCore.QPointF(x, y)
        return super().itemChange(change, value)

class EditorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FasterLivePortrait Source Data Editor")
        self.data = None
        self.landmark_items = []

        # Central UI setup
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Graphics view for image + landmarks
        self.scene = QtWidgets.QGraphicsScene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.Antialiasing)
        layout.addWidget(self.view, stretch=1)

        # Side panel tabs
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, stretch=0)

        # Info & Pose tab
        info_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(info_widget)
        self.orig_size_label = QtWidgets.QLabel("N/A")
        self.bbox_label = QtWidgets.QLabel("N/A")
        form.addRow("Original Size (WxH):", self.orig_size_label)
        form.addRow("Detected BBox:", self.bbox_label)
        self.yaw_spin = QtWidgets.QDoubleSpinBox(); self.pitch_spin = QtWidgets.QDoubleSpinBox(); self.roll_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.yaw_spin, self.pitch_spin, self.roll_spin): s.setRange(-180, 180); s.setSingleStep(1)
        form.addRow("Yaw (°):", self.yaw_spin)
        form.addRow("Pitch (°):", self.pitch_spin)
        form.addRow("Roll (°):", self.roll_spin)
        self.yaw_spin.valueChanged.connect(lambda v: self.update_pose('yaw', v))
        self.pitch_spin.valueChanged.connect(lambda v: self.update_pose('pitch', v))
        self.roll_spin.valueChanged.connect(lambda v: self.update_pose('roll', v))
        tabs.addTab(info_widget, "Info & Pose")

        # Shape coeffs tab
        shape_widget = QtWidgets.QWidget(); shape_layout = QtWidgets.QVBoxLayout(shape_widget)
        self.shape_table = QtWidgets.QTableWidget(0,2)
        self.shape_table.setHorizontalHeaderLabels(["Index","Value"])
        self.shape_table.horizontalHeader().setStretchLastSection(True)
        shape_layout.addWidget(self.shape_table)
        tabs.addTab(shape_widget, "Shape Coeffs")

        # Expression coeffs tab
        expr_widget = QtWidgets.QWidget(); expr_layout = QtWidgets.QVBoxLayout(expr_widget)
        self.expr_table = QtWidgets.QTableWidget(0,2)
        self.expr_table.setHorizontalHeaderLabels(["Index","Value"])
        self.expr_table.horizontalHeader().setStretchLastSection(True)
        expr_layout.addWidget(self.expr_table)
        tabs.addTab(expr_widget, "Expression Coeffs")

        # Status bar
        self.statusBar().showMessage("Ready")

        # Menu actions
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QtWidgets.QAction("Open JSON", self)
        save_action = QtWidgets.QAction("Save JSON", self)
        import_action = QtWidgets.QAction("Import Image...", self)
        file_menu.addAction(open_action); file_menu.addAction(save_action); file_menu.addAction(import_action)
        open_action.triggered.connect(self.open_file)
        save_action.triggered.connect(self.save_file)
        import_action.triggered.connect(self.import_image)

        self.scene.selectionChanged.connect(self.on_selection_changed)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open JSON", "", "JSON Files (*.json)")
        if path:
            self.load_data(path)

    def save_file(self):
        if not self.data:
            return
        # update landmarks
        updated = [[item.pos().x(), item.pos().y()] for item in self.landmark_items]
        self.data['landmarks_aligned'] = updated
        # update coeffs
        shape = [float(self.shape_table.item(i,1).text()) for i in range(self.shape_table.rowCount())]
        expr  = [float(self.expr_table.item(i,1).text())  for i in range(self.expr_table.rowCount())]
        if 'flame_model' in self.data:
            self.data['flame_model']['shape'] = shape
            self.data['flame_model']['expression'] = expr
        path, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON Files (*.json)")
        if path:
            with open(path,'w') as f: json.dump(self.data, f, indent=2)
            self.statusBar().showMessage(f"Saved {path}", 5000)

    def import_image(self):
        if not FaceAnalysis:
            QMessageBox.critical(self, "Error", "Please install insightface for import.")
            return
        img_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not img_path:
            return
        # reuse CLI logic for detect+crop
        from argparse import Namespace
        tmp_args = Namespace(image=img_path, output=os.devnull)
        # call separate function or inline simplified detection
        analyzer = FaceAnalysis(providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        analyzer.prepare(ctx_id=0, det_size=(640,640))
        img = cv2.imread(img_path)
        faces = analyzer.get(img)
        if not faces:
            QMessageBox.warning(self, "No face", "No face detected.")
            return
        face = faces[0]
        # Build minimal JSON by calling CLI-like logic; skip here for brevity
        QMessageBox.information(self, "Import", "Image import feature is pending implementation.")

    def load_data(self, json_path):
        with open(json_path,'r') as f:
            self.data = json.load(f)
        # load aligned image
        base_dir = os.path.dirname(json_path)
        img_file = os.path.join(base_dir, self.data.get('aligned_face_image'))
        pix = QtGui.QPixmap(img_file)
        self.scene.clear(); self.scene.addPixmap(pix)
        self.scene.setSceneRect(QtCore.QRectF(pix.rect()))
        # labels
        orig = self.data.get('original_size',['?','?'])
        self.orig_size_label.setText(f"{orig[0]} x {orig[1]}")
        bb = self.data.get('detected_bbox',{})
        self.bbox_label.setText(f"{bb.get('x',0):.1f},{bb.get('y',0):.1f},{bb.get('width',0):.1f},{bb.get('height',0):.1f}")
        # landmarks
        for item in self.landmark_items:
            self.scene.removeItem(item)
        self.landmark_items = []
        for i,pt in enumerate(self.data.get('landmarks_aligned',[])):
            item = LandmarkItem(i, pt[0], pt[1])
            self.scene.addItem(item)
            self.landmark_items.append(item)
        # pose
        pose = self.data.get('flame_model',{}).get('pose',{})
        self.yaw_spin.setValue(pose.get('yaw',0))
        self.pitch_spin.setValue(pose.get('pitch',0))
        self.roll_spin.setValue(pose.get('roll',0))
        # tables
        shape = self.data.get('flame_model',{}).get('shape',[])
        self.shape_table.setRowCount(len(shape))
        for i,v in enumerate(shape):
            self.shape_table.setItem(i,0,QtWidgets.QTableWidgetItem(str(i)))
            self.shape_table.setItem(i,1,QtWidgets.QTableWidgetItem(f"{v:.6f}"))
        expr = self.data.get('flame_model',{}).get('expression',[])
        self.expr_table.setRowCount(len(expr))
        for i,v in enumerate(expr):
            self.expr_table.setItem(i,0,QtWidgets.QTableWidgetItem(str(i)))
            self.expr_table.setItem(i,1,QtWidgets.QTableWidgetItem(f"{v:.6f}"))

    def update_pose(self, name, val):
        if not self.data: return
        pose = self.data.setdefault('flame_model',{}).setdefault('pose',{})
        pose[name] = float(val)
        self.statusBar().showMessage(f"Pose {name} set to {val:.1f}°",2000)

    def on_selection_changed(self):
        items = self.scene.selectedItems()
        if len(items)==1 and isinstance(items[0],LandmarkItem):
            p=items[0].pos(); idx=items[0].index
            self.statusBar().showMessage(f"Landmark {idx}: ({p.x():.1f},{p.y():.1f})")
        else:
            self.statusBar().clearMessage()

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = EditorWindow()
    win.resize(1200,800)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
