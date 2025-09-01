from ultralytics import YOLO
import cv2
import os
import tempfile
import time
import numpy as np
from collections import defaultdict, deque
import re

# Configuration Section - Easy to modify
CONFIG = {
    'PLATE_MODEL_PATH': "/home/gvassalaarachchi/Documents/vlpr/models/plate_detection.pt",
    'CHAR_MODEL_PATH': "/home/gvassalaarachchi/Documents/vlpr/models/character_recognition.pt",
    'VIDEO_SOURCE': "rtsp://admin:Web@doc122@172.30.30.194:554/Streaming/Channels/101",  # Your current RTSP
    'CONFIDENCE_THRESHOLD': 0.5,
    'TRACKER_MAX_AGE': 30,
    'TRACKER_MIN_HITS': 3,
    'TRACKER_IOU_THRESHOLD': 0.3,
    'FRAME_SKIP': 3,  # Process every 3rd frame for performance
    'SAVE_DETECTIONS': True,
    'SHOW_INDIVIDUAL_PLATES': True
}

class PlateTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        self.frame_count += 1
        
        # Update existing tracks
        updated_tracks = {}
        for track_id, track in self.tracks.items():
            # Find best matching detection
            best_match_idx, best_iou = -1, 0
            for i, det in enumerate(detections):
                iou = self.calculate_iou(track['bbox'], det['bbox'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_match_idx = i
            
            if best_match_idx >= 0:
                # Update track with detection
                det = detections[best_match_idx]
                track['bbox'] = det['bbox']
                track['text_history'].append(det['text'])
                track['confidence_history'].append(det['confidence'])
                track['last_seen'] = self.frame_count
                track['hits'] += 1
                track['crop'] = det['crop']  # Store latest crop
                
                # Calculate consensus text
                track['consensus_text'] = self.get_consensus_text(track['text_history'])
                track['avg_confidence'] = np.mean(track['confidence_history'])
                
                updated_tracks[track_id] = track
                # Remove matched detection
                detections.pop(best_match_idx)
            elif self.frame_count - track['last_seen'] < self.max_age:
                # Keep track alive but don't update
                updated_tracks[track_id] = track
        
        # Create new tracks for unmatched detections
        for det in detections:
            new_track = {
                'bbox': det['bbox'],
                'text_history': [det['text']],
                'confidence_history': [det['confidence']],
                'consensus_text': det['text'],
                'avg_confidence': det['confidence'],
                'hits': 1,
                'last_seen': self.frame_count,
                'first_seen': self.frame_count,
                'crop': det['crop']
            }
            updated_tracks[self.next_id] = new_track
            self.next_id += 1
        
        self.tracks = updated_tracks
        return self.tracks
    
    def calculate_iou(self, box1, box2):
        # Calculate Intersection over Union between two bounding boxes
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0
    
    def get_consensus_text(self, text_history):
        # Use the most frequent text in recent history
        if not text_history:
            return ""
        
        # Consider only recent entries (last 10 frames)
        recent_texts = text_history[-10:]
        
        # Count occurrences
        text_counts = {}
        for text in recent_texts:
            if text and text != "No text detected":
                text_counts[text] = text_counts.get(text, 0) + 1
        
        if not text_counts:
            return recent_texts[-1] if recent_texts else ""
        
        # Return the most frequent text
        consensus = max(text_counts.items(), key=lambda x: x[1])[0]
        
        # Additional validation: if we have a properly formatted plate, prefer it
        formatted_plates = []
        for text, count in text_counts.items():
            if re.match(r'^[A-Z]{2,3}-\d{1,4}[A-Z]?$', text):
                formatted_plates.append((text, count))
        
        if formatted_plates:
            # Prefer properly formatted plates
            return max(formatted_plates, key=lambda x: x[1])[0]
        
        return consensus

class SriLankanPlateValidator:
    def __init__(self):
        # Character model class mapping (based on diagnostic results)
        self.class_to_char = {
            0: '-', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
            11: 'A', 12: 'B', 13: 'C', 14: 'D', 15: 'E', 16: 'F', 17: 'G', 18: 'H', 19: 'I', 20: 'J',
            21: 'K', 22: 'L', 23: 'M', 24: 'N', 25: 'O', 26: 'P', 27: 'Q', 28: 'R', 29: 'S', 30: 'T',
            31: 'U', 32: 'V', 33: 'W', 34: 'X', 35: 'Y', 36: 'Z'
        }
        
        # Sri Lankan plate patterns - strict format (exactly 4 digits required)
        self.patterns = [
            r'^[A-Z]{2}-\d{4}$',            # Standard 2-letter format: AB-1234
            r'^[A-Z]{3}-\d{4}$',            # Standard 3-letter format: CBN-5808, CBX-1234  
        ]
        
        # Invalid patterns - things that should NOT be valid
        self.invalid_patterns = [
            r'^[A-Z]{1}-.*',               # Single letter prefixes
            r'^.*-\d{1,3}$',               # Less than 4 digits (DBT-11, CAC-77, KIA-737)
            r'^.*-\d{5,}$',                # More than 4 digits
            r'^.*-\d{1,4}[A-Z]+.*',        # Numbers followed by letters (BCG-875F)
            r'^[A-Z]{4,}-.*',              # Too many letters at start (more than 3)
            r'^.*-[A-Z]{2,}$',             # Ends with multiple letters
        ]
        
        # Generate all valid Sri Lankan plate prefixes systematically
        self.valid_prefixes = []
        
        # 2-letter combinations (all valid letter combinations)
        # Format: AA-0000 to ZZ-9999 (older system, still valid)
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for first in letters:
            for second in letters:
                self.valid_prefixes.append(first + second)
        
        # 3-letter combinations - systematic generation for Sri Lankan plates
        # Current system follows alphabetical progression: AAA-0001 to ZZZ-9999
        for first in letters:
            for second in letters:
                for third in letters:
                    prefix = first + second + third
                    self.valid_prefixes.append(prefix)
        
        # Positional OCR corrections based on Sri Lankan plate format
        # First 2-3 characters should be letters, next 4 should be numbers
        self.letter_corrections = {
            '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', 
            '6': 'G', '7': 'T', '8': 'B', '9': 'P'
        }
        
        self.number_corrections = {
            'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 
            'G': '6', 'T': '7', 'B': '8', 'P': '9', 'D': '0', 'Q': '0'
        }
    
    def validate_and_correct(self, plate_text, confidence):
        # Skip if no text detected
        if not plate_text or plate_text == "No text detected":
            return plate_text, confidence, False
        
        # Clean up the text (remove extra spaces, normalize case)
        cleaned_text = plate_text.strip().upper()
        
        # Apply positional OCR corrections
        corrected_text = self.apply_positional_corrections(cleaned_text)
        
        # Format to Sri Lankan standard (add dash if missing)
        formatted_text = self.format_sri_lankan_plate(corrected_text)
        
        # Check if the plate matches any pattern
        is_valid = any(re.match(pattern, formatted_text) for pattern in self.patterns)
        
        # Check against invalid patterns - if it matches any, mark as invalid
        is_invalid = any(re.match(pattern, formatted_text) for pattern in self.invalid_patterns)
        
        # Check if prefix is realistic for Sri Lankan plates
        prefix_realistic = False
        dash_pos = formatted_text.find('-')
        if dash_pos > 0:
            prefix = formatted_text[:dash_pos]
            prefix_realistic = prefix in self.valid_prefixes
        
        # Final validation - must match valid pattern AND not match invalid pattern AND have realistic prefix
        is_valid = is_valid and not is_invalid and prefix_realistic
        
        # Adjust confidence based on validation
        adjusted_confidence = confidence * (1.2 if is_valid else 0.8)
        
        return formatted_text, min(adjusted_confidence, 1.0), is_valid
    
    def apply_positional_corrections(self, text):
        """Apply OCR corrections based on character position in Sri Lankan plates"""
        if not text:
            return text
        
        # Handle text with existing dashes - preserve the dash position but fix misplaced characters
        if '-' in text:
            parts = text.split('-')
            if len(parts) == 2:
                prefix, suffix = parts
                
                # Special case: Handle misplaced characters due to dash positioning
                # If prefix has 3+ characters and last one should be a number, move it to suffix
                if len(prefix) >= 3:
                    last_char = prefix[-1]
                    if last_char.isalpha() and last_char in self.number_corrections:
                        # Move the last character from prefix to beginning of suffix
                        corrected_char = self.number_corrections[last_char]
                        prefix = prefix[:-1]  # Remove last character from prefix
                        suffix = corrected_char + suffix  # Add corrected character to start of suffix
                
                # Correct remaining prefix (should be letters)
                corrected_prefix = ""
                for char in prefix:
                    if char.isdigit():
                        corrected_prefix += self.letter_corrections.get(char, char)
                    else:
                        corrected_prefix += char
                
                # Correct suffix (should be numbers, except possibly the very last one)
                corrected_suffix = ""
                for i, char in enumerate(suffix):
                    # All characters in suffix should be digits, except possibly the very last one
                    if char.isalpha():
                        corrected_suffix += self.number_corrections.get(char, char)
                    else:
                        corrected_suffix += char
                
                return f"{corrected_prefix}-{corrected_suffix}"
        
        # Handle text without dashes - remove any spaces and apply corrections
        clean_text = text.replace(' ', '').upper()
        corrected = []
        
        for i, char in enumerate(clean_text):
            if i < 3:  # First 2-3 positions should be letters
                if char.isdigit():
                    # Convert digit to most likely letter
                    corrected.append(self.letter_corrections.get(char, char))
                else:
                    corrected.append(char)
            else:  # Remaining positions should be numbers (except possible suffix letter)
                if char.isalpha() and i < len(clean_text) - 1:  # Not the last character
                    # Convert letter to most likely digit
                    corrected.append(self.number_corrections.get(char, char))
                else:
                    corrected.append(char)
        
        return ''.join(corrected)

    def format_sri_lankan_plate(self, text):
        """Format text to Sri Lankan plate standard"""
        if not text:
            return text
            
        # Remove any existing dashes or spaces
        clean_text = text.replace('-', '').replace(' ', '')
        
        # Pattern: Letters followed by numbers
        import re
        match = re.match(r'^([A-Z]{2,3})(\d{1,4})([A-Z]?)$', clean_text)
        
        if match:
            letters, numbers, suffix = match.groups()
            # Format as LLL-DDDD or LL-DDDD
            formatted = f"{letters}-{numbers}{suffix}"
            return formatted
        
        # If no clear pattern, return original
        return text

def is_reasonable_plate_text(text):
    """Check if detected text looks like a reasonable license plate"""
    if not text or len(text) < 5 or len(text) > 10:
        return False
    
    # Should have at least some letters and some numbers
    has_letters = any(c.isalpha() for c in text)
    has_numbers = any(c.isdigit() for c in text)
    
    if not (has_letters and has_numbers):
        return False
    
    # Check for excessive repetition (like 'GGGGGG')
    for char in set(text):
        if text.count(char) > len(text) * 0.6:  # More than 60% same character
            return False
    
    return True

def smart_character_ordering(char_boxes, plate_shape):
    """
    Intelligent character ordering for both single-row and two-row license plates
    """
    if not char_boxes:
        return [], []
    
    plate_height, plate_width = plate_shape[:2]
    
    # Group characters by row based on Y-coordinate
    # Use relative positions to handle different plate sizes
    row_threshold = plate_height * 0.3  # 30% of plate height as threshold
    
    # Find the median Y position to separate rows
    y_positions = [box['y'] + box['h']/2 for box in char_boxes]
    y_positions.sort()
    median_y = y_positions[len(y_positions)//2]
    
    # Check for two-row layout
    top_row = [box for box in char_boxes if (box['y'] + box['h']/2) < median_y]
    bottom_row = [box for box in char_boxes if (box['y'] + box['h']/2) >= median_y]
    
    # Determine if this is likely a two-row plate
    is_two_row = (len(top_row) >= 2 and len(bottom_row) >= 2 and 
                  len(top_row) + len(bottom_row) >= 5)
    
    if is_two_row:
        # Two-row layout: letters on top, numbers on bottom (typical Sri Lankan format)
        # Sort each row by x-coordinate
        top_row.sort(key=lambda x: x['x'])
        bottom_row.sort(key=lambda x: x['x'])
        
        # Verify the pattern: top row should be mostly letters, bottom row mostly numbers
        top_letters = sum(1 for box in top_row if box['is_letter'])
        bottom_numbers = sum(1 for box in bottom_row if not box['is_letter'])
        
        if top_letters >= len(top_row) * 0.6 and bottom_numbers >= len(bottom_row) * 0.6:
            # Valid two-row pattern: combine top row + bottom row
            ordered_boxes = top_row + bottom_row
        else:
            # Pattern doesn't match expectation, fall back to single-row
            char_boxes.sort(key=lambda x: x['x'])
            ordered_boxes = char_boxes
    else:
        # Single-row layout: sort by x-coordinate only
        char_boxes.sort(key=lambda x: x['x'])
        ordered_boxes = char_boxes
    
    # Extract characters and confidences
    chars = [box['char'] for box in ordered_boxes]
    confidences = [box['conf'] for box in ordered_boxes]
    
    return chars, confidences

def run_enhanced_plate_detection():
    """Enhanced plate detection with tracking and consensus"""
    
    # Validate model paths
    if not os.path.exists(CONFIG['PLATE_MODEL_PATH']) or not os.path.exists(CONFIG['CHAR_MODEL_PATH']):
        print("❌ Model paths are invalid")
        print(f"Plate model: {CONFIG['PLATE_MODEL_PATH']}")
        print(f"Char model: {CONFIG['CHAR_MODEL_PATH']}")
        return

    # Determine if source is a file or IP camera stream
    source = CONFIG['VIDEO_SOURCE']
    if not (os.path.isfile(source) or source.startswith("rtsp://") or source.startswith("http://")):
        print("❌ Video source not found or invalid:", source)
        return

    print("🚗 Enhanced Vehicle License Plate Detection System")
    print("=" * 60)
    print(f"📹 Video Source: {source}")
    print(f"🎯 Plate Model: {os.path.basename(CONFIG['PLATE_MODEL_PATH'])}")
    print(f"🔤 Char Model: {os.path.basename(CONFIG['CHAR_MODEL_PATH'])}")
    print(f"🎛️  Confidence Threshold: {CONFIG['CONFIDENCE_THRESHOLD']}")
    print("=" * 60)

    # Load models
    try:
        plate_model = YOLO(CONFIG['PLATE_MODEL_PATH'])
        char_model = YOLO(CONFIG['CHAR_MODEL_PATH'])
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return
    
    # Initialize tracker and validator
    tracker = PlateTracker(
        max_age=CONFIG['TRACKER_MAX_AGE'],
        min_hits=CONFIG['TRACKER_MIN_HITS'],
        iou_threshold=CONFIG['TRACKER_IOU_THRESHOLD']
    )
    validator = SriLankanPlateValidator()

    # Open video or camera stream
    cap = cv2.VideoCapture(source)
    
    # Set buffer size to reduce latency for live streams
    if source.startswith("rtsp://") or source.startswith("http://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("📡 Configured for RTSP stream")
    
    # Get video properties for performance tracking
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_count = 0
    start_time = time.time()
    
    # Initialize windows
    cv2.namedWindow("Enhanced License Plate Detection", cv2.WINDOW_AUTOSIZE)
    
    # Track individual plate windows
    plate_windows = {}
    
    print("🎮 Controls: 'q' to quit, 's' to save current frame")
    print("-" * 50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()
        
        # Skip frames for better performance (process every nth frame for live streams)
        if source.startswith("rtsp://") or source.startswith("http://"):
            if frame_count % CONFIG['FRAME_SKIP'] != 0:
                continue

        # Plate detection
        try:
            plate_results = plate_model.predict(source=frame, imgsz=640, conf=CONFIG['CONFIDENCE_THRESHOLD'], verbose=False)
            plate_res = plate_results[0]
        except Exception as e:
            print(f"⚠️ Plate detection error: {e}")
            continue

        annotated_frame = frame.copy()
        detections = []
        
        # Process detected plates
        if len(plate_res.boxes) > 0:
            for i, box in enumerate(plate_res.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                confidence = float(plate_res.boxes.conf[i])
                
                # Only process high-confidence detections
                if confidence < CONFIG['CONFIDENCE_THRESHOLD']:
                    continue
                
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
                    plate_text = "No text detected"
                    char_confidence = 0.0
                    
                    if len(char_res.boxes) > 0:
                        # Smart character sorting for both single-row and two-row plates
                        char_boxes = []
                        for j, cbox in enumerate(char_res.boxes.xyxy):
                            x1c, y1c, x2c, y2c = cbox.cpu().numpy().astype(int)
                            char_conf = float(char_res.boxes.conf[j])
                            char_cls = int(char_res.boxes.cls[j]) if char_res.boxes.cls is not None else 0
                            
                            # Only include high-confidence characters
                            if char_conf > 0.3:
                                char = validator.class_to_char.get(char_cls, str(char_cls))
                                char_boxes.append({
                                    'x': x1c, 'y': y1c, 'w': x2c-x1c, 'h': y2c-y1c,
                                    'char': char, 'conf': char_conf, 'is_letter': char.isalpha()
                                })
                        
                        # Determine if this is a two-row plate layout
                        chars, confidences = smart_character_ordering(char_boxes, plate_crop.shape)
                        
                        if chars and len(chars) >= 4:  # At least 4 characters for valid plate
                            plate_text = "".join(chars)
                            char_confidence = np.mean(confidences) if confidences else 0.0
                            
                            # Filter out obviously wrong detections
                            if is_reasonable_plate_text(plate_text):
                                pass  # Text looks reasonable
                            else:
                                plate_text = "No text detected"
                                char_confidence = 0.0
                    
                    # Validate and correct plate text
                    validated_text, validated_conf, is_valid = validator.validate_and_correct(plate_text, char_confidence)
                    
                    # Create detection object for tracker
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'text': validated_text,
                        'confidence': validated_conf,
                        'is_valid': is_valid,
                        'crop': plate_crop
                    })
                    
                except Exception as e:
                    print(f"⚠️ Character recognition error: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_crop_path):
                        os.remove(temp_crop_path)
        
        # Update tracker with new detections
        tracks = tracker.update(detections)
        
        # Clean up windows for disappeared tracks
        current_track_ids = set(tracks.keys())
        windows_to_remove = []
        for window_name in list(plate_windows.keys()):
            # Extract track ID from window name "Plate {track_id}"
            if window_name.startswith("Plate "):
                try:
                    track_id = int(window_name.split(" ")[1])
                    if track_id not in current_track_ids:
                        # Track disappeared, close its window
                        cv2.destroyWindow(window_name)
                        windows_to_remove.append(window_name)
                        print(f"🗑️  Closed window for disappeared track {track_id}")
                except (ValueError, IndexError):
                    pass  # Skip malformed window names
        
        # Remove closed windows from tracking
        for window_name in windows_to_remove:
            del plate_windows[window_name]
        
        # Re-validate consensus text for all mature tracks (important for accuracy)
        for track_id, track in tracks.items():
            if track['hits'] >= tracker.min_hits:  # Only re-validate mature tracks
                _, _, track['is_valid'] = validator.validate_and_correct(track['consensus_text'], track['avg_confidence'])
        
        # Draw tracked plates on frame
        valid_tracks = 0
        for track_id, track in tracks.items():
            x1, y1, x2, y2 = track['bbox']
            
            # Choose color based on validation status and track maturity
            if track['hits'] >= tracker.min_hits:
                color = (0, 255, 0) if track.get('is_valid', False) else (0, 165, 255)  # Green or Orange
                valid_tracks += 1
            else:
                color = (128, 128, 128)  # Gray for immature tracks
            
            # Draw plate bounding box with track ID and consensus text
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Display track info
            info_text = f"ID:{track_id} {track['consensus_text']} ({track['avg_confidence']:.2f})"
            cv2.putText(annotated_frame, info_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Show hits count
            hits_text = f"Hits: {track['hits']}"
            cv2.putText(annotated_frame, hits_text, (x1, y2+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Update individual plate window if we have enough hits and feature is enabled
            if (track['hits'] >= tracker.min_hits and 
                CONFIG['SHOW_INDIVIDUAL_PLATES'] and 
                'crop' in track):
                
                # Create or update individual plate window
                window_name = f"Plate {track_id}"
                if window_name not in plate_windows:
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    plate_windows[window_name] = True
                
                # Resize crop for consistent display
                crop_resized = cv2.resize(track['crop'], (300, 100), interpolation=cv2.INTER_CUBIC)
                
                # Add text overlay to plate window
                text_overlay = crop_resized.copy()
                cv2.putText(text_overlay, track['consensus_text'], (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(text_overlay, f"Conf: {track['avg_confidence']:.2f}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(text_overlay, f"Hits: {track['hits']}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(text_overlay, "VALID" if track.get('is_valid', False) else "INVALID", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if track.get('is_valid', False) else (0, 0, 255), 1)
                
                cv2.imshow(window_name, text_overlay)

        # Add performance info and statistics
        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Status overlay with enhanced information
        status_text = f"FPS: {current_fps:.1f} | Frame: {frame_count} | Tracks: {len(tracks)} | Valid: {valid_tracks}"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        # Show main detection window
        cv2.imshow("Enhanced License Plate Detection", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and CONFIG['SAVE_DETECTIONS']:
            # Save current frame with detections
            timestamp = int(time.time())
            save_path = f"enhanced_detected_plates_{timestamp}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"💾 Saved frame: {save_path}")
            
            # Save individual plate crops
            for track_id, track in tracks.items():
                if 'crop' in track and track['hits'] >= tracker.min_hits:
                    crop_path = f"enhanced_plate_{track_id}_{timestamp}.jpg"
                    cv2.imwrite(crop_path, track['crop'])
                    print(f"💾 Saved plate crop: {crop_path} - '{track['consensus_text']}'")

        # Print detection info periodically
        if frame_count % 60 == 0 and tracks:  # Every 60 frames (about every 2 seconds)
            print(f"🎯 Frame {frame_count}: Tracking {len(tracks)} plates ({valid_tracks} mature)")
            for track_id, track in tracks.items():
                if track['hits'] >= tracker.min_hits:
                    status = "VALID" if track.get('is_valid', False) else "INVALID"
                    print(f"   Track {track_id}: '{track['consensus_text']}' ({status}, {track['avg_confidence']:.2f}, {track['hits']} hits)")
                    
                    # Debug: Show recent text history for troubleshooting
                    if len(track['text_history']) > 1:
                        recent_texts = track['text_history'][-5:]  # Last 5 detections
                        print(f"      Recent detections: {recent_texts}")

    # Clean up
    cap.release()
    for window_name in plate_windows:
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
    
    # Final statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print("\n" + "="*60)
    print("🏁 Enhanced Detection Session Complete")
    print(f"📊 Processed {frame_count} frames in {total_time:.1f} seconds")
    print(f"⚡ Average FPS: {avg_fps:.2f}")
    print(f"🔢 Maximum track ID reached: {tracker.next_id-1}")
    print(f"🎯 Total tracks created: {tracker.next_id-1}")
    print("="*60)

if __name__ == "__main__":
    # Allow configuration override from command line or environment
    import sys
    
    if len(sys.argv) > 1:
        # Override video source if provided as argument
        CONFIG['VIDEO_SOURCE'] = sys.argv[1]
        print(f"📹 Using video source from command line: {CONFIG['VIDEO_SOURCE']}")
    
    run_enhanced_plate_detection()
