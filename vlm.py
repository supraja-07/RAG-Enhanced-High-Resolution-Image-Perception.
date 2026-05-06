import torch
import numpy as np
from ultralytics import YOLO
import easyocr
from collections import defaultdict
from PIL import Image


class SceneGraphBuilder:

    def __init__(self, device="cuda", debug=True):
        self.device = device
        self.debug = debug

        print("[GRAPH INIT] Loading YOLOv3...")
        self.detector = YOLO("yolov3.pt")
        self.detector.to(device)

        print("[GRAPH INIT] Loading EasyOCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))

    # ------------------------------------------------
    # Extract dominant color (average RGB)
    # ------------------------------------------------
    def extract_color(self, image_crop):
        img = np.array(image_crop)
        avg_color = img.mean(axis=(0, 1))
        return tuple(avg_color.astype(int))

    # ------------------------------------------------
    # Detect objects in patch
    # ------------------------------------------------
    def detect_objects(self, patch_img, patch_id):

        results = self.detector(patch_img, verbose=False)[0]
        objects = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = self.detector.names[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            obj_crop = patch_img.crop((x1, y1, x2, y2))
            color = self.extract_color(obj_crop)

            ocr_text = self.ocr_reader.readtext(
                np.array(obj_crop),
                detail=0
            )

            obj_node = {
                "patch_id": patch_id,
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "color": color,
                "ocr_text": ocr_text
            }
            objects.append(obj_node)

        if self.debug:
            print(f"[GRAPH] Patch {patch_id} → {len(objects)} objects")

        return objects

    # ------------------------------------------------
    # Build full scene graph
    # ------------------------------------------------
    def build_graph(self, image_patch_list, rows, cols):

        graph = {
            "patch_nodes": {},
            "object_nodes": {},
            "patch_edges": defaultdict(list),
            "contains_edges": defaultdict(list),
            "object_edges": defaultdict(list)
        }

        patch_id = 1

        for r in range(rows):
            for c in range(cols):

                graph["patch_nodes"][patch_id] = {
                    "row": r,
                    "col": c
                }

                # Detect objects in this patch
                patch_img = image_patch_list[patch_id]
                objects = self.detect_objects(patch_img, patch_id)

                for obj in objects:
                    obj_id = len(graph["object_nodes"]) + 1
                    # Compute global center
                    x1, y1, x2, y2 = obj["bbox"]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    global_x = graph["patch_nodes"][patch_id]["col"] * patch_img.size[0] + center_x
                    global_y = graph["patch_nodes"][patch_id]["row"] * patch_img.size[1] + center_y
                    obj["global_center"] = (global_x, global_y)
                    
                    graph["object_nodes"][obj_id] = obj

                    # Contains edge: patch contains object
                    graph["contains_edges"][patch_id].append(obj_id)

                patch_id += 1

        # Build spatial edges between patches
        for pid1 in graph["patch_nodes"]:
            r1 = graph["patch_nodes"][pid1]["row"]
            c1 = graph["patch_nodes"][pid1]["col"]

            # Check neighbors
            directions = [
                (-1, 0, "above"),
                (1, 0, "below"),
                (0, -1, "left_of"),
                (0, 1, "right_of")
            ]

            for dr, dc, rel in directions:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < rows and 0 <= c2 < cols:
                    pid2 = r2 * cols + c2 + 1
                    graph["patch_edges"][pid1].append({
                        "target": pid2,
                        "relation": rel
                    })

        # Build object-object relations (simplified)
        for obj1_id, obj1 in graph["object_nodes"].items():
            for obj2_id, obj2 in graph["object_nodes"].items():
                if obj1_id != obj2_id:
                    # Simple spatial relation based on centers
                    center1 = obj1.get("global_center", (0, 0))
                    center2 = obj2.get("global_center", (0, 0))

                    dx = center2[0] - center1[0]
                    dy = center2[1] - center1[1]

                    if abs(dx) > abs(dy):
                        rel = "left_of" if dx < 0 else "right_of"
                    else:
                        rel = "above" if dy < 0 else "below"

                    graph["object_edges"][obj1_id].append({
                        "target": obj2_id,
                        "relation": rel
                    })

        if self.debug:
            print(f"[GRAPH] Built graph with {len(graph['patch_nodes'])} patches, {len(graph['object_nodes'])} objects")

        return graph