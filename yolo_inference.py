import cv2
import numpy as np
import pickle
from ultralytics import YOLO


def get_upper_body_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    upper_body = crop[:int(crop.shape[0] * 0.6), :]
    if upper_body.size == 0:
        return None
    mean_color = cv2.mean(upper_body)[:3]
    return np.array(mean_color, dtype=np.float32)


def cluster_teams(colors, k=3):
    colors = np.array(colors, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(colors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return labels.flatten(), centers.astype(int)


def main():
    video_path = "before/rugby.mp4"
    model = YOLO("weights/yolo26n.pt")

    import os
    os.makedirs("output", exist_ok=True)

    # Check if cached data exists
    cache_exists = os.path.exists("output/detections.pkl") and os.path.exists("output/team_mapping.pkl")

    if cache_exists:
        print("Loading cached detections and team mapping...")
        with open("output/detections.pkl", "rb") as f:
            data = pickle.load(f)
            detections = data["detections"]
            player_colors = data["player_colors"]
        with open("output/team_mapping.pkl", "rb") as f:
            data = pickle.load(f)
            team_mapping = data["team_mapping"]
            centers = data["centers"]
        print(f"Loaded {len(detections)} detections, {len(team_mapping)} players")
        print(f"Team centers (BGR): {centers}")
    else:
        # PASS 1: Collecte des détections
        print("Pass 1: Collecting detections...")
        player_colors = {}
        detections = []

        results = model.track(
            source=video_path,
            stream=True,
            tracker="botsort.yaml"
        )

        for frame_idx, result in enumerate(results):
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                track_id = int(box.id[0]) if box.id is not None else -1
                if track_id == -1:
                    continue

                bbox = box.xyxy[0].cpu().numpy()
                color = get_upper_body_color(result.orig_img, bbox)

                if color is not None:
                    if track_id not in player_colors:
                        player_colors[track_id] = []
                    player_colors[track_id].append(color)
                    detections.append((frame_idx, track_id, bbox))

        # Save detections
        with open("output/detections.pkl", "wb") as f:
            pickle.dump({"detections": detections, "player_colors": player_colors}, f)

        print(f"Collected {len(detections)} detections for {len(player_colors)} unique players")

        if len(player_colors) < 3:
            print("Not enough players detected")
            return

        # K-means sur les couleurs moyennes par joueur
        track_ids = list(player_colors.keys())
        colors = [np.mean(player_colors[tid], axis=0) for tid in track_ids]
        colors = np.array(colors, dtype=np.float32)
        labels, centers = cluster_teams(colors, k=3)

        team_mapping = {tid: label for tid, label in zip(track_ids, labels)}

        print(f"Team centers (BGR): {centers}")

        # Save team mapping
        with open("output/team_mapping.pkl", "wb") as f:
            pickle.dump({"team_mapping": team_mapping, "centers": centers}, f)

    # PASS 2: Rendu vidéo
    print("Pass 2: Rendering video...")

    team_colors = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (255, 255, 0)
    }

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs("output/rugby_tracked", exist_ok=True)
    out = cv2.VideoWriter("output/rugby_tracked/result.mp4", fourcc, fps, (width, height))

    frame_detections = {}
    for frame_idx, track_id, bbox in detections:
        if frame_idx not in frame_detections:
            frame_detections[frame_idx] = []
        frame_detections[frame_idx].append((track_id, bbox))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_detections:
            for track_id, bbox in frame_detections[frame_idx]:
                team_id = team_mapping.get(track_id, 0)
                color = team_colors[team_id]

                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"ID:{track_id} T{team_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Done! Output: output/rugby_tracked/result.mp4")


if __name__ == "__main__":
    main()
