import os
import cv2
from ultralytics import YOLO

# Simply put your images here
image_paths = ['1.jpeg', '2.jpeg', '3.jpeg', "4.jpg", "5.jpeg"]
pest_weights = 'weights/pest.pt'
weed_weights = 'weights/weed.pt'
output_dir = 'analysis_results'

def run_analysis():
    # Load models once
    print("Loading models...")
    pest_model = YOLO(pest_weights)
    weed_model = YOLO(weed_weights)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Skipping: {img_path} not found")
            continue
            
        print(f"\nAnalyzing {img_path}...")
        img = cv2.imread(img_path)
        
        # Run detection
        p_res = pest_model(img, conf=0.25)
        w_res = weed_model(img, conf=0.25)
        
        # Draw results
        for r in p_res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, f"Pest {box.conf[0]:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for r in w_res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"Weed {box.conf[0]:.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save
        out_path = os.path.join(output_dir, f"result_{img_path}")
        cv2.imwrite(out_path, img)
        print(f"Saved to {out_path}")

if __name__ == "__main__":
    run_analysis()
