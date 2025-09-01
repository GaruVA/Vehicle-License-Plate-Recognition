from ultralytics import YOLO
import cv2
import os
import tempfile
import time
import numpy as np

def run_plate_and_character_models_video(plate_model_path, char_model_path, source, conf_thresh=0.5):
    # Validate model paths
    if not os.path.exists(plate_model_path) or not os.path.exists(char_model_path):
        print("Model paths are invalid")
        return

    # Determine if source is a file or IP camera stream
    if not (os.path.isfile(source) or source.startswith("rtsp://") or source.startswith("http://")):
        print("Video source not found or invalid:", source)
        return

    # Load models
    plate_model = YOLO(plate_model_path)
    char_model = YOLO(char_model_path)

    # Open video or camera stream
    cap = cv2.VideoCapture(source)
    
    # Set buffer size to reduce latency for live streams
    if source.startswith("rtsp://") or source.startswith("http://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get video properties for performance tracking
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = 0
    start_time = time.time()
    
    # Initialize windows once
    cv2.namedWindow("License Plate Detection", cv2.WINDOW_AUTOSIZE)
    
    print("🚗 Vehicle License Plate Detection System Started")
    print("Press 'q' to quit, 's' to save current frame")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()
        
        # Skip frames for better performance (process every 3rd frame for live streams)
        if source.startswith("rtsp://") or source.startswith("http://"):
            if frame_count % 3 != 0:
                continue

        # Plate detection
        plate_results = plate_model.predict(source=frame, imgsz=640, conf=conf_thresh, verbose=False)
        plate_res = plate_results[0]

        annotated_frame = frame.copy()
        plates_detected = 0
        detected_plates = []

        # Process detected plates
        if len(plate_res.boxes) > 0:
            for i, box in enumerate(plate_res.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                confidence = float(plate_res.boxes.conf[i])
                
                # Only process high-confidence detections
                if confidence < conf_thresh:
                    continue
                    
                plates_detected += 1
                
                # Draw plate bounding box with confidence
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Plate {i+1}: {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Crop plate safely with padding
                plate_height = y2 - y1
                plate_width = x2 - x1
                
                # Add padding and ensure bounds
                pad_x = int(plate_width * 0.1)
                pad_y = int(plate_height * 0.1)
                x1_crop = max(0, x1 - pad_x)
                y1_crop = max(0, y1 - pad_y)
                x2_crop = min(frame.shape[1], x2 + pad_x)
                y2_crop = min(frame.shape[0], y2 + pad_y)
                
                plate_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                if plate_crop.size == 0 or plate_crop.shape[0] < 20 or plate_crop.shape[1] < 50:
                    continue  # Skip if crop is too small

                # Character recognition on cropped plate
                temp_crop_path = os.path.join(tempfile.gettempdir(), f"temp_plate_{frame_count}_{i}.jpg")
                cv2.imwrite(temp_crop_path, plate_crop)
                
                try:
                    char_results = char_model.predict(temp_crop_path, imgsz=640, conf=0.3, verbose=False)
                    char_res = char_results[0]

                    # Process character detections
                    characters = []
                    if len(char_res.boxes) > 0:
                        # Sort characters by x-coordinate (left to right)
                        char_boxes = []
                        for j, cbox in enumerate(char_res.boxes.xyxy):
                            x1c, y1c, x2c, y2c = cbox.cpu().numpy().astype(int)
                            char_conf = float(char_res.boxes.conf[j])
                            char_cls = int(char_res.boxes.cls[j]) if char_res.boxes.cls is not None else 0
                            char_boxes.append((x1c, char_conf, char_cls, (x1c, y1c, x2c, y2c)))
                        
                        # Sort by x-coordinate
                        char_boxes.sort(key=lambda x: x[0])
                        
                        # Draw character boxes and collect text
                        for x1c, char_conf, char_cls, (x1c, y1c, x2c, y2c) in char_boxes:
                            if char_conf > 0.3:  # Only high-confidence characters
                                cv2.rectangle(plate_crop, (x1c, y1c), (x2c, y2c), (0, 0, 255), 1)
                                characters.append(f"{char_cls}")
                    
                    # Display detected text
                    plate_text = "".join(characters) if characters else "No text detected"
                    detected_plates.append({
                        'text': plate_text,
                        'confidence': confidence,
                        'bbox': (x1, y1, x2, y2),
                        'crop': plate_crop
                    })
                    
                    # Draw text on main frame
                    cv2.putText(annotated_frame, f"Text: {plate_text}", 
                               (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                except Exception as e:
                    print(f"Character recognition error: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_crop_path):
                        os.remove(temp_crop_path)

        # Add performance info
        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Status overlay
        status_text = f"FPS: {current_fps:.1f} | Frame: {frame_count} | Plates: {plates_detected}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Show main detection window
        cv2.imshow("License Plate Detection", annotated_frame)
        
        # Show individual plate crops in separate windows (only if plates detected)
        for idx, plate_info in enumerate(detected_plates[:3]):  # Limit to 3 plates max
            if plate_info['crop'].size > 0:
                # Resize crop for better visibility
                crop_resized = cv2.resize(plate_info['crop'], (300, 100), interpolation=cv2.INTER_CUBIC)
                window_name = f"Plate {idx+1}: {plate_info['text']}"
                cv2.imshow(window_name, crop_resized)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and detected_plates:
            # Save current frame with detections
            timestamp = int(time.time())
            save_path = f"detected_plates_{timestamp}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"💾 Saved frame: {save_path}")
            
            # Save individual plate crops
            for idx, plate_info in enumerate(detected_plates):
                crop_path = f"plate_crop_{timestamp}_{idx}.jpg"
                cv2.imwrite(crop_path, plate_info['crop'])
                print(f"💾 Saved plate crop: {crop_path}")

        # Print detection info periodically
        if frame_count % 30 == 0 and detected_plates:
            print(f"🎯 Frame {frame_count}: Detected {len(detected_plates)} plates")
            for idx, plate in enumerate(detected_plates):
                print(f"   Plate {idx+1}: '{plate['text']}' (confidence: {plate['confidence']:.3f})")

    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n" + "="*50)
    print("🏁 Detection Session Complete")
    print(f"📊 Processed {frame_count} frames in {total_time:.1f} seconds")
    print(f"⚡ Average FPS: {avg_fps:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_plate_and_character_models_video(
        "/home/gvassalaarachchi/Documents/vlpr/models/plate_detection.pt",
        "/home/gvassalaarachchi/Documents/vlpr/models/character_recognition.pt",
        "rtsp://admin:Web@doc122@172.30.30.194:554/Streaming/Channels/101"
    )
