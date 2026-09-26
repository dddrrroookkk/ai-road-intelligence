
import cv2
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Road Intelligence",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.ri-title {font-size: 42px; font-weight: 800;}
.ri-sub {color: #9ca3af; font-size: 17px; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="ri-title">🚦 Road Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ri-sub">'
    'AI-powered real-time traffic monitoring, speed estimation and congestion analytics'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = "demo.mp4"

FALLBACK_METERS_PER_PIXEL = 0.05

SMOOTHING_ALPHA = 0.25

DIRECTION_HISTORY_SIZE = 10

DIRECTION_THRESHOLD = 5

SLOW_SPEED_KMH = 15.0

STOPPED_SPEED_KMH = 3.0

COUNTING_LINE_Y = 400

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


# ============================================================
# LOAD CALIBRATION
# ============================================================

def load_calibration():

    path = Path("calibration.json")

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


calibration = load_calibration()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return YOLO("yolo11n.pt")


model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ System Controls")

show_video = st.sidebar.checkbox(
    "Show Live Video",
    value=True
)

if calibration:

    st.sidebar.success(
        "Perspective calibration loaded"
    )

else:

    st.sidebar.warning(
        "Using fallback scale: 0.05 m/px"
    )

st.sidebar.markdown("---")

st.sidebar.markdown("""
**AI Pipeline**

YOLO → Detection  
ByteTrack → Tracking  
ROI → Road Filtering  
Perspective → Calibration  
EMA → Speed smoothing  
Direction → Movement  
Congestion → Analysis
""")


# ============================================================
# VIDEO
# ============================================================

video = cv2.VideoCapture(VIDEO)

if not video.isOpened():

    st.error(f"Could not open {VIDEO}")

    st.stop()


fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# ============================================================
# PERSPECTIVE CALIBRATION
# ============================================================

H = None

world_meters_per_pixel = FALLBACK_METERS_PER_PIXEL

roi_polygon = None

if calibration:

    source_points = np.float32(
        calibration["source_points"]
    )

    # This is the exact road polygon selected
    # during calibration.
    roi_polygon = source_points.astype(
        np.int32
    )

    road_width_meters = float(
        calibration["road_width_meters"]
    )

    road_length_meters = float(
        calibration["road_length_meters"]
    )

    # World coordinate canvas.
    # 100 pixels represent one real-world meter.
    world_scale = 100.0

    destination_points = np.float32([
        [0, 0],
        [
            road_width_meters * world_scale,
            0
        ],
        [
            road_width_meters * world_scale,
            road_length_meters * world_scale
        ],
        [
            0,
            road_length_meters * world_scale
        ]
    ])

    H = cv2.getPerspectiveTransform(
        source_points,
        destination_points
    )

    world_meters_per_pixel = (
        1.0 / world_scale
    )


# ============================================================
# HELPER: ROI CHECK
# ============================================================

def point_inside_roi(x, y):

    if roi_polygon is None:

        return True

    result = cv2.pointPolygonTest(
        roi_polygon,
        (float(x), float(y)),
        False
    )

    return result >= 0


# ============================================================
# HELPER: PERSPECTIVE POINT
# ============================================================

def transform_point(x, y):

    if H is None:

        return (
            float(x),
            float(y)
        )

    point = np.array(
        [[[float(x), float(y)]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        H
    )[0][0]

    return (
        float(transformed[0]),
        float(transformed[1])
    )


# ============================================================
# TRACKING DATA
# ============================================================

previous_world_positions = {}
previous_image_positions = {}

vehicle_speeds = {}

vehicle_directions = {}

position_history = {}

display_id_map = {}

next_display_id = 1

counted_raw_ids = set()


# ============================================================
# CROSSED VEHICLES
# ============================================================

crossed_count = {

    "car": 0,

    "motorcycle": 0,

    "bus": 0,

    "truck": 0
}


# ============================================================
# ANALYTICS HISTORY
# ============================================================

vehicle_history = deque(
    maxlen=120
)

speed_history = deque(
    maxlen=120
)

congestion_history = deque(
    maxlen=120
)


# ============================================================
# CONGESTION
# ============================================================

def calculate_congestion(
    current_vehicles,
    speeds
):

    if (
        current_vehicles == 0
        or not speeds
    ):

        return (
            0,
            "LOW",
            0.0,
            0,
            0
        )

    average_speed = (
        sum(speeds)
        /
        len(speeds)
    )

    slow_vehicles = sum(
        speed < SLOW_SPEED_KMH
        for speed in speeds
    )

    stopped_vehicles = sum(
        speed < STOPPED_SPEED_KMH
        for speed in speeds
    )

    score = 0

    # Density
    if current_vehicles >= 25:

        score += 40

    elif current_vehicles >= 15:

        score += 25

    elif current_vehicles >= 8:

        score += 10

    # Average speed
    if average_speed < 10:

        score += 40

    elif average_speed < 20:

        score += 25

    elif average_speed < 30:

        score += 10

    # Slow vehicle percentage
    slow_percentage = (
        slow_vehicles
        /
        len(speeds)
    ) * 100

    if slow_percentage >= 70:

        score += 20

    elif slow_percentage >= 40:

        score += 10

    if score >= 60:

        level = "HIGH"

    elif score >= 30:

        level = "MEDIUM"

    else:

        level = "LOW"

    return (
        score,
        level,
        average_speed,
        slow_vehicles,
        stopped_vehicles
    )


# ============================================================
# DASHBOARD PLACEHOLDERS
# ============================================================

st.subheader(
    "📊 Live Traffic Overview"
)

m1, m2, m3, m4 = st.columns(4)

metric_vehicles = m1.empty()

metric_speed = m2.empty()

metric_congestion = m3.empty()

metric_score = m4.empty()


st.subheader(
    "🎥 Live Traffic Monitoring"
)

video_placeholder = st.empty()


st.subheader(
    "🚗 Current Vehicles"
)

c1, c2, c3, c4 = st.columns(4)

metric_car = c1.empty()

metric_bike = c2.empty()

metric_bus = c3.empty()

metric_truck = c4.empty()


st.subheader(
    "🚧 Vehicles Crossed Counting Line"
)

x1, x2, x3, x4 = st.columns(4)

metric_cross_car = x1.empty()

metric_cross_bike = x2.empty()

metric_cross_bus = x3.empty()

metric_cross_truck = x4.empty()


st.subheader(
    "🚦 Traffic Status"
)

s1, s2, s3 = st.columns(3)

metric_slow = s1.empty()

metric_stopped = s2.empty()

metric_calibration = s3.empty()


st.subheader(
    "🧭 Current Traffic Direction"
)

d1, d2, d3, d4 = st.columns(4)

metric_up = d1.empty()

metric_down = d2.empty()

metric_left = d3.empty()

metric_right = d4.empty()


st.subheader(
    "📈 Traffic Analytics"
)

chart_vehicle = st.empty()

chart_speed = st.empty()

chart_congestion = st.empty()


# ============================================================
# MAIN VIDEO LOOP
# ============================================================

while True:

    ret, frame = video.read()

    if not ret:

        break


    # ========================================================
    # YOLO + BYTE TRACK
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]


    current_vehicles = 0

    current_distribution = {

        "car": 0,

        "motorcycle": 0,

        "bus": 0,

        "truck": 0
    }


    active_raw_ids = set()


    # ========================================================
    # DRAW ONLY VEHICLES INSIDE ROI
    # ========================================================

    # Start with the original frame instead of result.plot().
    # This prevents YOLO from drawing boxes on the opposite
    # carriageway before the ROI filter is applied.
    annotated = frame.copy()

    roi_vehicle_ids = set()

    if result.boxes.id is not None:

        draw_track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        draw_class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        draw_confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )

        draw_boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )

        for raw_id, class_id, confidence, box in zip(
            draw_track_ids,
            draw_class_ids,
            draw_confidences,
            draw_boxes
        ):

            # Only draw vehicles
            if class_id not in vehicle_classes:
                continue

            x1, y1, x2, y2 = box

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Only vehicles inside the calibrated road ROI
            if not point_inside_roi(
                center_x,
                center_y
            ):
                continue

            roi_vehicle_ids.add(raw_id)

            vehicle_type = vehicle_classes[class_id]

            # Vehicle-specific colors
            if vehicle_type == "car":
                box_color = (255, 255, 255)
            elif vehicle_type == "motorcycle":
                box_color = (255, 255, 0)
            elif vehicle_type == "bus":
                box_color = (255, 0, 255)
            else:
                box_color = (0, 255, 255)

            # Bounding box
            cv2.rectangle(
                annotated,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                box_color,
                3
            )

            # Compact display ID
            display_id = display_id_map.get(
                raw_id,
                raw_id
            )

            label = (
                f"id:{display_id} "
                f"{vehicle_type} "
                f"{confidence:.2f}"
            )

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )

            label_y = max(
                int(y1),
                text_h + 8
            )

            # Label background
            cv2.rectangle(
                annotated,
                (
                    int(x1),
                    label_y - text_h - 8
                ),
                (
                    int(x1) + text_w + 8,
                    label_y + baseline
                ),
                box_color,
                -1
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (
                    int(x1) + 4,
                    label_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2
            )

    # ========================================================
    # DRAW ROI BORDER
    # ========================================================

    if roi_polygon is not None:

        cv2.polylines(
            annotated,
            [roi_polygon],
            True,
            (0, 255, 255),
            3
        )


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )


        for raw_id, class_id, box in zip(
            track_ids,
            class_ids,
            boxes
        ):

            # Only vehicles
            if class_id not in vehicle_classes:

                continue


            x1, y1, x2, y2 = box


            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # =================================================
            # ROI FILTER
            # =================================================

            if not point_inside_roi(
                center_x,
                center_y
            ):

                # Vehicle is outside monitored road.
                # It does NOT contribute to analytics.
                continue


            vehicle_type = (
                vehicle_classes[class_id]
            )


            current_vehicles += 1

            current_distribution[
                vehicle_type
            ] += 1

            active_raw_ids.add(
                raw_id
            )


            # =================================================
            # COMPACT DISPLAY ID
            # =================================================

            if raw_id not in display_id_map:

                display_id_map[
                    raw_id
                ] = next_display_id

                next_display_id += 1


            # =================================================
            # WORLD POSITION
            # =================================================

            world_x, world_y = (
                transform_point(
                    center_x,
                    center_y
                )
            )


            if raw_id not in position_history:

                position_history[
                    raw_id
                ] = deque(
                    maxlen=
                    DIRECTION_HISTORY_SIZE
                )


            position_history[
                raw_id
            ].append(
                (
                    world_x,
                    world_y
                )
            )


            # =================================================
            # SPEED
            # =================================================

            if raw_id in previous_world_positions:

                old_x, old_y = (
                    previous_world_positions[
                        raw_id
                    ]
                )


                dx = (
                    world_x - old_x
                )

                dy = (
                    world_y - old_y
                )


                distance_world_pixels = math.sqrt(
                    dx ** 2 +
                    dy ** 2
                )


                distance_meters = (
                    distance_world_pixels
                    *
                    world_meters_per_pixel
                )


                raw_speed = (
                    distance_meters
                    *
                    fps
                    *
                    3.6
                )


                # Ignore tiny/noisy movement
                if distance_world_pixels >= 0.5:

                    if raw_id in vehicle_speeds:

                        old_speed = (
                            vehicle_speeds[
                                raw_id
                            ]
                        )


                        smoothed_speed = (
                            SMOOTHING_ALPHA
                            *
                            raw_speed
                            +
                            (
                                1 -
                                SMOOTHING_ALPHA
                            )
                            *
                            old_speed
                        )

                    else:

                        smoothed_speed = (
                            raw_speed
                        )


                    # Safety cap
                    smoothed_speed = min(
                        max(
                            smoothed_speed,
                            0.0
                        ),
                        120.0
                    )


                    vehicle_speeds[
                        raw_id
                    ] = smoothed_speed


                # =================================================
                # DIRECTION
                # =================================================

                history = (
                    position_history[
                        raw_id
                    ]
                )


                if len(history) >= 3:

                    old_hx, old_hy = (
                        history[0]
                    )

                    new_hx, new_hy = (
                        history[-1]
                    )


                    total_dx = (
                        new_hx - old_hx
                    )

                    total_dy = (
                        new_hy - old_hy
                    )


                    if (
                        abs(total_dx)
                        < DIRECTION_THRESHOLD
                        and
                        abs(total_dy)
                        < DIRECTION_THRESHOLD
                    ):

                        direction = (
                            "STATIONARY"
                        )

                    elif (
                        abs(total_dy)
                        >
                        abs(total_dx)
                    ):

                        if total_dy > 0:

                            direction = "DOWN"

                        else:

                            direction = "UP"

                    else:

                        if total_dx > 0:

                            direction = "RIGHT"

                        else:

                            direction = "LEFT"


                    vehicle_directions[
                        raw_id
                    ] = direction


                # =================================================
                # LINE CROSSING
                # =================================================

                if raw_id in previous_image_positions:

                    previous_image_y = (
                        previous_image_positions[
                            raw_id
                        ][1]
                    )

                    if (
                        previous_image_y < COUNTING_LINE_Y
                        and center_y >= COUNTING_LINE_Y
                        and raw_id not in counted_raw_ids
                    ):

                        counted_raw_ids.add(raw_id)

                        crossed_count[
                            vehicle_type
                        ] += 1


            # =================================================
            # SAVE POSITIONS
            # =================================================

            previous_image_positions[
                raw_id
            ] = (
                center_x,
                center_y
            )

            previous_world_positions[
                raw_id
            ] = (
                world_x,
                world_y
            )


    # ========================================================
    # ACTIVE SPEEDS
    # ========================================================

    active_speeds = [

        vehicle_speeds[raw_id]

        for raw_id in active_raw_ids

        if raw_id in vehicle_speeds
    ]


    # ========================================================
    # CONGESTION
    # ========================================================

    (
        congestion_score,
        congestion_level,
        average_speed,
        slow_vehicles,
        stopped_vehicles
    ) = calculate_congestion(
        current_vehicles,
        active_speeds
    )


    # ========================================================
    # DRAW COUNTING LINE
    # ========================================================

    cv2.line(
        annotated,
        (0, COUNTING_LINE_Y),
        (
            annotated.shape[1],
            COUNTING_LINE_Y
        ),
        (0, 0, 255),
        3
    )


    # ========================================================
    # DRAW DIRECTION ARROWS
    # ========================================================

    if result.boxes.id is not None:

        for raw_id, box in zip(
            result.boxes.id.int().cpu().tolist(),
            result.boxes.xyxy.cpu().tolist()
        ):

            if raw_id not in active_raw_ids:

                continue

            # Only draw arrows for vehicles inside the ROI
            if raw_id not in roi_vehicle_ids:

                continue


            direction = vehicle_directions.get(
                raw_id,
                "UNKNOWN"
            )


            if direction == "STATIONARY":

                continue


            x1, y1, x2, y2 = box

            px = int(
                (x1 + x2) / 2
            )

            py = int(
                (y1 + y2) / 2
            )


            arrow_dx = 0
            arrow_dy = 0


            if direction == "UP":

                arrow_dy = -35

            elif direction == "DOWN":

                arrow_dy = 35

            elif direction == "LEFT":

                arrow_dx = -35

            elif direction == "RIGHT":

                arrow_dx = 35


            if arrow_dx != 0 or arrow_dy != 0:

                cv2.arrowedLine(
                    annotated,
                    (px, py),
                    (
                        px + arrow_dx,
                        py + arrow_dy
                    ),
                    (0, 255, 255),
                    3,
                    tipLength=0.3
                )


    # ========================================================
    # TOP METRICS
    # ========================================================

    metric_vehicles.metric(
        "🚗 Vehicles Now",
        current_vehicles
    )

    metric_speed.metric(
        "⚡ Average Speed",
        f"{average_speed:.1f} km/h"
    )

    metric_congestion.metric(
        "🚦 Congestion",
        congestion_level
    )

    metric_score.metric(
        "📊 Congestion Score",
        congestion_score
    )


    # ========================================================
    # CURRENT DISTRIBUTION
    # ========================================================

    metric_car.metric(
        "🚗 Cars",
        current_distribution["car"]
    )

    metric_bike.metric(
        "🏍️ Motorcycles",
        current_distribution["motorcycle"]
    )

    metric_bus.metric(
        "🚌 Buses",
        current_distribution["bus"]
    )

    metric_truck.metric(
        "🚚 Trucks",
        current_distribution["truck"]
    )


    # ========================================================
    # CROSSED COUNTS
    # ========================================================

    metric_cross_car.metric(
        "🚗 Cars Crossed",
        crossed_count["car"]
    )

    metric_cross_bike.metric(
        "🏍️ Motorcycles Crossed",
        crossed_count["motorcycle"]
    )

    metric_cross_bus.metric(
        "🚌 Buses Crossed",
        crossed_count["bus"]
    )

    metric_cross_truck.metric(
        "🚚 Trucks Crossed",
        crossed_count["truck"]
    )


    # ========================================================
    # TRAFFIC STATUS
    # ========================================================

    metric_slow.metric(
        "🐢 Slow Vehicles",
        slow_vehicles
    )

    metric_stopped.metric(
        "🛑 Stopped Vehicles",
        stopped_vehicles
    )

    metric_calibration.metric(
        "📐 Calibration",
        "Perspective"
        if H is not None
        else "Fallback 0.05 m/px"
    )


    # ========================================================
    # DIRECTIONS
    # ========================================================

    direction_counts = {

        "UP": 0,

        "DOWN": 0,

        "LEFT": 0,

        "RIGHT": 0
    }


    for raw_id in active_raw_ids:

        direction = (
            vehicle_directions.get(
                raw_id
            )
        )


        if direction in direction_counts:

            direction_counts[
                direction
            ] += 1


    metric_up.metric(
        "⬆️ UP",
        direction_counts["UP"]
    )

    metric_down.metric(
        "⬇️ DOWN",
        direction_counts["DOWN"]
    )

    metric_left.metric(
        "⬅️ LEFT",
        direction_counts["LEFT"]
    )

    metric_right.metric(
        "➡️ RIGHT",
        direction_counts["RIGHT"]
    )


    # ========================================================
    # HISTORY
    # ========================================================

    vehicle_history.append(
        current_vehicles
    )

    speed_history.append(
        average_speed
    )

    congestion_history.append(
        congestion_score
    )


    chart_vehicle.line_chart(
        pd.DataFrame(
            {
                "Vehicles": list(
                    vehicle_history
                )
            }
        )
    )


    chart_speed.line_chart(
        pd.DataFrame(
            {
                "Average Speed (km/h)":
                list(speed_history)
            }
        )
    )


    chart_congestion.line_chart(
        pd.DataFrame(
            {
                "Congestion Score":
                list(congestion_history)
            }
        )
    )


    # ========================================================
    # VIDEO
    # ========================================================

    if show_video:

        frame_rgb = cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )

        video_placeholder.image(
            frame_rgb,
            channels="RGB",
            width="stretch"
        )


    time.sleep(0.01)


# ============================================================
# CLEANUP
# ============================================================

video.release()
