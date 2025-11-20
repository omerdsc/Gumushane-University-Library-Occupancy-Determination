import cv2
import time
import math

# --- PyTorch 2.6 fix ---
try:
    import torch.serialization as ts
    from ultralytics.nn.tasks import DetectionModel
    ts.add_safe_globals([DetectionModel])
except:
    pass

from ultralytics import YOLO


# ==============================
# AYARLAR
# ==============================
VIDEO_PATH = "kutuphaneVideo5.mp4"
OUTPUT_VIDEO = "output.mp4"
MAX_DIST = 80
MAX_MISS_TIME = 1.0


# ==============================
# TAKİP SINIFI
# ==============================
class PersonTrack:
    def __init__(self, track_id, bbox, start_time):
        self.id = track_id
        self.bbox = bbox
        self.start_time = start_time
        self.last_seen = start_time

    def update(self, bbox, t):
        self.bbox = bbox
        self.last_seen = t

    def elapsed(self, t):
        return t - self.start_time


def bbox_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


def dist(a, b):
    return math.dist(a, b)


def point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


# ==============================
# YOLO MODELİ
# ==============================
model = YOLO("yolov8n.pt")
names = model.model.names
person_id = [i for i, n in names.items() if n == "person"][0]

# ==============================
# VİDEO YÜKLE
# ==============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Video acilamadi")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
ret, first_frame = cap.read()
if not ret:
    print("Video kare okunamadi")
    exit()

h, w, _ = first_frame.shape

# ==============================
# CIKTI VİDEOSU
# ==============================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

# ==============================
# SANDALYE SEÇME
# ==============================
chairs = []
drawing = False
ix, iy = -1, -1

def mouse_draw(event, x, y, flags, img):
    global ix, iy, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = img.copy()
        cv2.rectangle(temp, (ix, iy), (x, y), (255, 200, 0), 2)
        for bx in chairs:
            cv2.rectangle(temp, (bx[0], bx[1]), (bx[2], bx[3]), (255, 200, 0), 2)
        cv2.imshow("Sandalyeleri Isaretle (ENTER)", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, x2 = sorted([ix, x])
        y1, y2 = sorted([iy, y])
        chairs.append((x1, y1, x2, y2))

        temp = img.copy()
        for bx in chairs:
            cv2.rectangle(temp, (bx[0], bx[1]), (bx[2], bx[3]), (255, 200, 0), 2)
        cv2.imshow("Sandalyeleri Isaretle (ENTER)", temp)


clone = first_frame.copy()
cv2.namedWindow("Sandalyeleri Isaretle (ENTER)")
cv2.setMouseCallback("Sandalyeleri Isaretle (ENTER)", mouse_draw, clone)

print("Sandalyeleri çiz: sürükle-bırak. ENTER ile başla.")

while True:
    temp = clone.copy()
    for bx in chairs:
        cv2.rectangle(temp, (bx[0], bx[1]), (bx[2], bx[3]), (255, 200, 0), 2)
    cv2.imshow("Sandalyeleri Isaretle (ENTER)", temp)

    key = cv2.waitKey(20) & 0xFF
    if key == 13:
        break
    elif key == ord('q'):
        exit()

cv2.destroyWindow("Sandalyeleri Isaretle (ENTER)")

chair_count = len(chairs)
print("Toplam sandalye:", chair_count)

# ==============================
# ANALİZ
# ==============================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_idx = 0

tracks = {}
next_id = 0

seat_prev = [False] * chair_count
seat_now = [False] * chair_count

seat_start = [None] * chair_count
seat_total = [0.0] * chair_count

print("Video işleniyor... (q ile çık)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    t = frame_idx / fps

    # YOLO TESPİT
    res = model(frame, verbose=False)[0]

    dets = []
    for box in res.boxes:
        if int(box.cls) == person_id:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append(((x1, y1, x2, y2), bbox_center((x1, y1, x2, y2))))

    # TRACK GÜNCELLEME
    for tr in tracks.values():
        tr.matched = False

    for bbox, center in dets:
        best_track = None
        best_d = 1e9

        for tr in tracks.values():
            if tr.matched:
                continue

            d = dist(center, bbox_center(tr.bbox))
            if d < best_d:
                best_d = d
                best_track = tr

        if best_track and best_d <= MAX_DIST:
            best_track.update(bbox, t)
            best_track.matched = True
        else:
            new = PersonTrack(next_id, bbox, t)
            new.matched = True
            tracks[next_id] = new
            next_id += 1

    # KAYBOLAN TRACK SİL
    delete_ids = []
    for tid, tr in tracks.items():
        if (t - tr.last_seen) > MAX_MISS_TIME:
            delete_ids.append(tid)

    for tid in delete_ids:
        del tracks[tid]

    # SANDALYE DURUMU
    seat_now = [False] * chair_count

    for tr in tracks.values():
        cx, cy = bbox_center(tr.bbox)
        for i, box in enumerate(chairs):
            if point_in_box((cx, cy), box):
                seat_now[i] = True

    # SÜRE HESABI
    for i in range(chair_count):
        if seat_now[i] and not seat_prev[i]:
            seat_start[i] = t
        elif not seat_now[i] and seat_prev[i]:
            if seat_start[i] is not None:
                seat_total[i] += t - seat_start[i]
                seat_start[i] = None

    seat_prev = seat_now.copy()

    # YAZILAR
    occupied = sum(seat_now)
    empty = chair_count - occupied

    cv2.putText(frame, f"Kisi: {len(tracks)}  Dolu: {occupied}  Bos: {empty}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Sandalyeleri çiz
    for i, (x1, y1, x2, y2) in enumerate(chairs):
        color = (0, 255, 255) if seat_now[i] else (255, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"S{i}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Kişileri çiz
    for tr in tracks.values():
        x1, y1, x2, y2 = tr.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elapsed = tr.elapsed(t)
        cv2.putText(frame, f"ID{tr.id} {int(elapsed)}s",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    # ÇIKTI VİDEOSU YAZ
    out.write(frame)

    cv2.imshow("Kutuphanede Doluluk Takibi", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# KAPAT
out.release()
cap.release()
cv2.destroyAllWindows()

print("İşlenmiş video kaydedildi: output.mp4")
print("Bitti!")
