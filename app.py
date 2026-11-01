
import streamlit as st
import cv2
import tempfile
import os
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="TactiVision AI",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ TactiVision AI")
st.caption("AI-powered Football Video Analysis | Prototype v0.2")

st.info(
    "Upload a short football clip to detect people "
    "and visualize their estimated positions."
)


@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")


def draw_pitch(pitch_positions):

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")

    ax.set_facecolor("#267a43")
    fig.patch.set_facecolor("#0b1220")

    white = "white"

    # Outer boundary
    ax.plot(
        [0, 105, 105, 0, 0],
        [0, 0, 68, 68, 0],
        color=white,
        linewidth=2
    )

    # Halfway line
    ax.plot(
        [52.5, 52.5],
        [0, 68],
        color=white
    )

    # Centre circle
    centre = plt.Circle(
        (52.5, 34),
        9.15,
        fill=False,
        color=white
    )
    ax.add_patch(centre)

    # Centre spot
    ax.scatter([52.5], [34], color=white, s=15)

    # Penalty areas
    ax.plot(
        [0, 16.5, 16.5, 0],
        [13.84, 13.84, 54.16, 54.16],
        color=white
    )

    ax.plot(
        [105, 88.5, 88.5, 105],
        [13.84, 13.84, 54.16, 54.16],
        color=white
    )

    # Goal areas
    ax.plot(
        [0, 5.5, 5.5, 0],
        [24.84, 24.84, 43.16, 43.16],
        color=white
    )

    ax.plot(
        [105, 99.5, 99.5, 105],
        [24.84, 24.84, 43.16, 43.16],
        color=white
    )

    # Plot estimated positions
    if pitch_positions:
        positions = np.array(pitch_positions)

        pitch_x = positions[:, 0] * 105
        pitch_y = (1 - positions[:, 1]) * 68

        ax.scatter(
            pitch_x,
            pitch_y,
            s=45,
            color="#ffd54f",
            edgecolors="black",
            linewidths=0.7
        )

    ax.set_title(
        "Estimated Player Positions",
        color=white,
        fontsize=15,
        pad=15
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


uploaded_video = st.file_uploader(
    "Upload a football video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video:

    st.video(uploaded_video)

    if st.button("Analyse Video", type="primary"):

        temp_path = None

        try:

            suffix = Path(uploaded_video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:

                temp_file.write(uploaded_video.getvalue())
                temp_path = temp_file.name

            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                st.error("Could not read this video.")
                st.stop()

            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0:
                st.error("No readable frames found.")
                cap.release()
                st.stop()

            duration = total_frames / fps if fps > 0 else 0

            st.subheader("Video Information")

            col1, col2, col3 = st.columns(3)

            col1.metric("Duration", f"{duration:.1f} sec")
            col2.metric("Total Frames", total_frames)
            col3.metric("Frame Rate", f"{fps:.1f} FPS")

            st.subheader("AI Person Detection")

            model = load_model()

            sample_count = min(20, total_frames)
            frame_step = max(1, total_frames // sample_count)

            results_images = []
            detection_counts = []
            pitch_positions = []

            progress = st.progress(0)

            for i in range(sample_count):

                frame_number = min(
                    i * frame_step,
                    total_frames - 1
                )

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_number
                )

                success, frame = cap.read()

                if not success:
                    continue

                result = model.predict(
                    frame,
                    classes=[0],
                    conf=0.35,
                    imgsz=640,
                    verbose=False
                )[0]

                boxes = result.boxes

                count = len(boxes) if boxes is not None else 0
                detection_counts.append(count)

                # Collect estimated player positions
                if boxes is not None:

                    for box in boxes:

                        x1, y1, x2, y2 = (
                            box.xyxy[0].cpu().tolist()
                        )

                        center_x = (x1 + x2) / 2
                        bottom_y = y2

                        normalized_x = center_x / frame.shape[1]
                        normalized_y = bottom_y / frame.shape[0]

                        pitch_positions.append(
                            (normalized_x, normalized_y)
                        )

                annotated = cv2.cvtColor(
                    result.plot(),
                    cv2.COLOR_BGR2RGB
                )

                results_images.append(annotated)

                progress.progress((i + 1) / sample_count)

            cap.release()

            if results_images:

                st.success("Analysis completed!")

                a, b, c = st.columns(3)

                a.metric(
                    "Frames Analysed",
                    len(results_images)
                )

                b.metric(
                    "Average Detections",
                    f"{sum(detection_counts) / len(detection_counts):.1f}"
                )

                c.metric(
                    "Maximum Detections",
                    max(detection_counts)
                )

                st.caption(
                    "Detections may include referees, coaches "
                    "and spectators. Counts are not unique players."
                )

                st.subheader("Detection Results")

                columns = st.columns(2)

                for i, image in enumerate(results_images):

                    with columns[i % 2]:

                        st.image(
                            image,
                            caption=f"Sample frame {i + 1}",
                            use_container_width=True
                        )

                # 2D Pitch Visualization
                st.divider()

                st.header("⚽ 2D Football Pitch")

                st.caption(
                    "Experimental projection from video-frame positions. "
                    "Not calibrated to actual football pitch coordinates."
                )

                if pitch_positions:

                    fig = draw_pitch(pitch_positions)

                    st.pyplot(fig)

                    plt.close(fig)

                    st.metric(
                        "Recorded position samples",
                        len(pitch_positions)
                    )

                else:

                    st.warning(
                        "No player positions were detected."
                    )

            else:

                st.warning("No frames could be analysed.")

        except Exception as error:

            st.error(f"Analysis failed: {error}")

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


st.divider()

st.caption(
    "TactiVision AI | Early prototype | "
    "AI detections may contain errors."
)
