# HandDetector

A Python module for real-time hand detection and tracking using MediaPipe.

## Description

HandDetector is a lightweight wrapper around Google's MediaPipe Hands solution that simplifies the process of detecting and tracking hands in images or video streams. This module is designed to be easy to integrate into any computer vision project requiring hand pose estimation.

## Features

- Real-time hand detection in images and video streams
- Support for tracking multiple hands (configurable, default: 2)
- Hand landmark detection (21 points per hand)
- Landmark visualization
- FPS calculation and display
- Identification of hand type (Left/Right)
- Easy-to-use API

## Installation

### Prerequisites

- Python 3.6+
- OpenCV
- MediaPipe

### Install dependencies

```bash
pip install opencv-python mediapipe
```

### Install the module

Clone this repository:

```bash
git clone https://github.com/yourusername/hand-detector.git
cd hand-detector
```

## Usage

### Basic Example

```python
import cv2
from hand_detector import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the hand detector
detector = HandDetector(detection_confidence=0.7)

while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        break
        
    # Find hands in the image
    img = detector.find_hands(img)
    
    # Find positions of all hand landmarks
    hands = detector.find_positions(img)
    
    # Update and display FPS
    detector.update_fps()
    img = detector.display_fps(img)
    
    # If hands are detected, print the position of index finger tip (landmark 8)
    if hands:
        for hand in hands:
            index_finger_tip = hand[8]  # [id, x, y, z, hand_type]
            print(f"Index finger tip: {index_finger_tip[1]}, {index_finger_tip[2]}")
    
    # Display the image
    cv2.imshow("Hand Tracking", img)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### Advanced Usage

```python
import cv2
from hand_detector import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize detector with custom parameters
detector = HandDetector(
    mode=False,               # Set to True for static images
    max_hands=2,              # Maximum number of hands to detect
    detection_confidence=0.7, # Minimum detection confidence
    tracking_confidence=0.5   # Minimum tracking confidence
)

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Find hands but don't draw landmarks
    img = detector.find_hands(img, draw=False)
    
    # Get hand positions
    hands = detector.find_positions(img)
    
    # Process each detected hand
    for hand in hands:
        # Get hand type
        hand_type = hand[0][4]  # Right or Left
        
        # Draw circles at specific landmarks
        if len(hand) > 0:
            # Wrist position
            wrist = hand[0]
            cv2.circle(img, (wrist[1], wrist[2]), 10, (255, 0, 0), cv2.FILLED)
            
            # Thumb tip
            thumb_tip = hand[4]
            cv2.circle(img, (thumb_tip[1], thumb_tip[2]), 10, (0, 255, 0), cv2.FILLED)
            
            # Index finger tip
            index_tip = hand[8]
            cv2.circle(img, (index_tip[1], index_tip[2]), 10, (0, 0, 255), cv2.FILLED)
    
    # Update and display FPS
    fps = detector.update_fps()
    img = detector.display_fps(img)
    
    # Display the image
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## API Reference

### `HandDetector` Class

```python
detector = HandDetector(mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5)
```

#### Parameters

- `mode` (bool): If True, detection runs on every frame. If False, detection runs once and then tracking takes over.
- `max_hands` (int): Maximum number of hands to detect
- `detection_confidence` (float): Minimum confidence value (0-1) for hand detection
- `tracking_confidence` (float): Minimum confidence value (0-1) for landmark tracking

#### Methods

##### `find_hands(img, draw=True, flip_type=True)`

Process an image and detect hands.

- `img` (numpy.ndarray): Input image (BGR format)
- `draw` (bool): Whether to draw landmarks and connections
- `flip_type` (bool): Flip image horizontally for mirror effect
- Returns: Processed image

##### `find_positions(img)`

Find positions of all hand landmarks from all detected hands.

- `img` (numpy.ndarray): Input image
- Returns: List of hand data, where each hand is a list of landmark positions [id, x, y, z, hand_type]

##### `find_position(img, hand_no=0, draw=True, draw_radius=5)`

Legacy method for backward compatibility. Find landmark positions of a specific hand.

- `img` (numpy.ndarray): Input image
- `hand_no` (int): Index of the hand to track
- `draw` (bool): Whether to draw circles at landmark positions
- `draw_radius` (int): Radius of the circles
- Returns: List of [id, x, y] for each landmark

##### `update_fps()`

Calculate and update the frames per second.

- Returns: Current FPS value

##### `display_fps(img, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, color=(0, 255, 0), thickness=2)`

Display FPS on the image.

- `img` (numpy.ndarray): Input image
- `position` (tuple): Position to display the FPS text
- `font`: OpenCV font type
- `scale` (float): Font scale
- `color` (tuple): RGB color
- `thickness` (int): Text thickness
- Returns: Image with FPS displayed

## Hand Landmark Reference

```
HAND LANDMARKS:
0: WRIST
1-4: THUMB (1:CMC, 2:MCP, 3:IP, 4:TIP)
5-8: INDEX FINGER (5:MCP, 6:PIP, 7:DIP, 8:TIP)
9-12: MIDDLE FINGER (9:MCP, 10:PIP, 11:DIP, 12:TIP)
13-16: RING FINGER (13:MCP, 14:PIP, 15:DIP, 16:TIP)
17-20: PINKY (17:MCP, 18:PIP, 19:DIP, 20:TIP)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request