import cv2
import json

VIDEO = "demo.mp4"
OUTPUT = "calibration.json"

# Real-world dimensions of the selected road rectangle.
# Change these after deciding what the 4 clicked points represent.
ROAD_WIDTH_METERS = 10.0
ROAD_LENGTH_METERS = 30.0

points = []
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise SystemExit(f"Could not open {VIDEO}")

ret, frame = cap.read()
cap.release()

if not ret:
    raise SystemExit("Could not read first frame from video")

display = frame.copy()

def redraw():
    global display
    display = frame.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 7, (0, 255, 255), -1)
        cv2.putText(
            display, str(i + 1), (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i + 1], (0, 255, 255), 2)

    if len(points) == 4:
        cv2.line(display, points[3], points[0], (0, 255, 255), 2)

    cv2.putText(
        display,
        "Click 4 road corners: TL, TR, BR, BL | R=reset | ENTER=save | Q=quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2
    )

def mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        redraw()

cv2.namedWindow("Road Calibration")
cv2.setMouseCallback("Road Calibration", mouse)
redraw()

while True:
    cv2.imshow("Road Calibration", display)
    key = cv2.waitKey(20) & 0xFF

    if key == ord("r"):
        points.clear()
        redraw()

    elif key == 13 and len(points) == 4:
        break

    elif key == ord("q"):
        cv2.destroyAllWindows()
        raise SystemExit("Calibration cancelled.")

cv2.destroyAllWindows()

data = {
    "source_points": points,
    "road_width_meters": ROAD_WIDTH_METERS,
    "road_length_meters": ROAD_LENGTH_METERS,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"Saved {OUTPUT}")
print("Source points:", points)
print("Now run: streamlit run dashboard.py")
