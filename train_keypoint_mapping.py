#!/usr/bin/env python3
"""
Train Keypoint Mapping Model
Processes all frames in data/ directory, extracts keypoints, and learns
an improved mapping/calibration for better accuracy.

Usage:
    python train_keypoint_mapping.py [--data-dir data/] [--output mapping_model.json]
"""

import sys
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import mediapipe as mp
import os
import pickle

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not found. Install with: pip install scikit-learn")
    print("   Will use simple statistical mapping instead.")


class KeypointMappingTrainer:
    """Train a mapping model to improve keypoint normalization accuracy."""
    
    # MediaPipe landmark indices
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    
    def __init__(self, model_path=None):
        """Initialize MediaPipe pose detector."""
        self.fps = 30.0
        self.use_new_api = False
        self.pose_landmarker = None
        self.pose = None
        
        # Try new API first
        if hasattr(mp, 'tasks'):
            try:
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode
                
                # Find model
                if model_path is None:
                    local_model = os.path.join(os.getcwd(), 'pose_landmarker_full.task')
                    if os.path.exists(local_model):
                        model_path = local_model
                    else:
                        mp_path = os.path.dirname(mp.__file__)
                        model_path = os.path.join(mp_path, 'tasks', 'data', 'pose_landmarker_full.task')
                
                if os.path.exists(model_path):
                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=model_path),
                        running_mode=VisionRunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.3,
                        min_pose_presence_confidence=0.3,
                    )
                    self.pose_landmarker = PoseLandmarker.create_from_options(options)
                    self.use_new_api = True
                    print(f"✅ Using MediaPipe Tasks API with {os.path.basename(model_path)}")
            except Exception as e:
                print(f"⚠️ Failed to initialize new API: {e}")
        
        # Fall back to old API
        if not self.use_new_api:
            if hasattr(mp, 'solutions'):
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                self.use_new_api = False
                print("✅ Using MediaPipe Solutions API (legacy)")
            else:
                raise ImportError("MediaPipe API not available")
        
        # Training data storage
        self.training_data = []
        
    def extract_keypoints_from_image(self, image_path):
        """Extract raw keypoint data from a single image."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            return None
        
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        if self.use_new_api:
            ImageFormat = mp.ImageFormat
            mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = self.pose_landmarker.detect(mp_image)
            
            if not result.pose_landmarks or len(result.pose_landmarks) == 0:
                return None
            
            landmarks_list = result.pose_landmarks[0]
            
            # Convert to similar format
            class Landmark:
                def __init__(self, x, y, visibility):
                    self.x = x
                    self.y = y
                    self.visibility = visibility
            
            class Landmarks:
                def __init__(self, landmarks_list):
                    self.landmark = [None] * 33
                    for i, lm in enumerate(landmarks_list):
                        if i < 33:
                            self.landmark[i] = Landmark(lm.x, lm.y, lm.visibility)
            
            landmarks = Landmarks(landmarks_list)
        else:
            results = self.pose.process(frame_rgb)
            if not results.pose_landmarks:
                return None
            landmarks = results.pose_landmarks
        
        # Get key landmarks
        shoulder = landmarks.landmark[self.LEFT_SHOULDER]
        elbow = landmarks.landmark[self.LEFT_ELBOW]
        wrist = landmarks.landmark[self.LEFT_WRIST]
        
        # Check visibility
        if shoulder.visibility < 0.3 or wrist.visibility < 0.3:
            return None
        
        # Convert to pixel coordinates
        shoulder_pixel = np.array([shoulder.x * w, shoulder.y * h])
        elbow_pixel = np.array([elbow.x * w, elbow.y * h])
        wrist_pixel = np.array([wrist.x * w, wrist.y * h])
        
        # Calculate arm length
        arm_vec = wrist_pixel - shoulder_pixel
        arm_length_pixels = np.linalg.norm(arm_vec)
        
        if arm_length_pixels < 1:
            return None
        
        # Calculate raw normalized coordinates
        x_raw = (wrist_pixel[0] - shoulder_pixel[0]) / arm_length_pixels
        y_raw = (shoulder_pixel[1] - wrist_pixel[1]) / arm_length_pixels
        
        # Store raw data for training
        return {
            'shoulder_pixel': shoulder_pixel,
            'elbow_pixel': elbow_pixel,
            'wrist_pixel': wrist_pixel,
            'arm_length_pixels': arm_length_pixels,
            'x_raw': x_raw,
            'y_raw': y_raw,
            'shoulder_visibility': shoulder.visibility,
            'elbow_visibility': elbow.visibility,
            'wrist_visibility': wrist.visibility,
            'image_size': (w, h),
            'image_path': str(image_path)
        }
    
    def process_training_data(self, data_dir, max_images=1000):
        """Process images in data directory."""
        self.max_images = max_images
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        image_files = sorted(data_path.glob("*.jpg")) + sorted(data_path.glob("*.png"))
        
        # Limit to first N images
        max_images = getattr(self, 'max_images', 1000)
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
            print(f"Found {len(image_files)} images in {data_dir} (limited to first {max_images})")
        else:
            print(f"Found {len(image_files)} images in {data_dir}")
        
        print("Processing images...")
        
        successful = 0
        failed = 0
        
        for i, img_path in enumerate(image_files):
            if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
                print(f"  Processed {i + 1}/{len(image_files)} images... ({successful} successful, {failed} failed)")
            
            keypoint_data = self.extract_keypoints_from_image(img_path)
            
            if keypoint_data:
                self.training_data.append(keypoint_data)
                successful += 1
            else:
                failed += 1
        
        print(f"\n✅ Processing complete:")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total training samples: {len(self.training_data)}")
        
        return len(self.training_data) > 0
    
    def analyze_data_distribution(self):
        """Analyze the distribution of keypoint data."""
        if not self.training_data:
            return None
        
        x_raw = np.array([d['x_raw'] for d in self.training_data])
        y_raw = np.array([d['y_raw'] for d in self.training_data])
        arm_lengths = np.array([d['arm_length_pixels'] for d in self.training_data])
        
        stats = {
            'x_raw': {
                'min': float(np.min(x_raw)),
                'max': float(np.max(x_raw)),
                'mean': float(np.mean(x_raw)),
                'std': float(np.std(x_raw)),
                'median': float(np.median(x_raw))
            },
            'y_raw': {
                'min': float(np.min(y_raw)),
                'max': float(np.max(y_raw)),
                'mean': float(np.mean(y_raw)),
                'std': float(np.std(y_raw)),
                'median': float(np.median(y_raw))
            },
            'arm_length_pixels': {
                'min': float(np.min(arm_lengths)),
                'max': float(np.max(arm_lengths)),
                'mean': float(np.mean(arm_lengths)),
                'std': float(np.std(arm_lengths))
            },
            'num_samples': len(self.training_data)
        }
        
        return stats
    
    def train_mapping_model(self, method='polynomial'):
        """
        Train a mapping model to improve normalization.
        
        Methods:
        - 'linear': Simple linear regression
        - 'polynomial': Polynomial features (degree 2)
        - 'ridge': Ridge regression with regularization
        - 'statistical': Simple statistical mapping (no sklearn needed)
        """
        if not self.training_data:
            raise ValueError("No training data available. Process images first.")
        
        # Prepare training data
        X_raw = np.array([[d['x_raw'], d['y_raw']] for d in self.training_data])
        
        # Current normalization (baseline)
        X_current = (X_raw + 1.0) / 2.0  # Maps [-1, 1] to [0, 1]
        X_current = np.clip(X_current, 0.0, 1.0)
        
        if not HAS_SKLEARN or method == 'statistical':
            # Simple statistical mapping without sklearn
            # Learn optimal offset and scale from data distribution
            x_min, x_max = np.min(X_raw[:, 0]), np.max(X_raw[:, 0])
            y_min, y_max = np.min(X_raw[:, 1]), np.max(X_raw[:, 1])
            
            # Calculate optimal mapping to [0, 1]
            x_scale = 1.0 / (x_max - x_min + 1e-6)
            x_offset = -x_min
            y_scale = 1.0 / (y_max - y_min + 1e-6)
            y_offset = -y_min
            
            models = {
                'x': {'scale': float(x_scale), 'offset': float(x_offset), 'min': float(x_min), 'max': float(x_max)},
                'y': {'scale': float(y_scale), 'offset': float(y_offset), 'min': float(y_min), 'max': float(y_max)},
                'type': 'statistical'
            }
            
            # Evaluate
            X_pred = np.column_stack([
                np.clip((X_raw[:, 0] + x_offset) * x_scale, 0.0, 1.0),
                np.clip((X_raw[:, 1] + y_offset) * y_scale, 0.0, 1.0)
            ])
            mse = np.mean((X_pred - X_current) ** 2)
            method = 'statistical'
            
        elif method == 'linear':
            model_x = LinearRegression()
            model_y = LinearRegression()
            
            # Train separate models for x and y
            model_x.fit(X_raw, X_current[:, 0])
            model_y.fit(X_raw, X_current[:, 1])
            
            models = {'x': model_x, 'y': model_y, 'type': 'linear'}
            
            # Evaluate
            X_pred = np.column_stack([
                model_x.predict(X_raw),
                model_y.predict(X_raw)
            ])
            mse = np.mean((X_pred - X_current) ** 2)
            
        elif method == 'polynomial':
            # Use polynomial features for non-linear mapping
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X_raw)
            
            model_x = Ridge(alpha=0.1)
            model_y = Ridge(alpha=0.1)
            
            model_x.fit(X_poly, X_current[:, 0])
            model_y.fit(X_poly, X_current[:, 1])
            
            models = {
                'x': Pipeline([('poly', poly), ('reg', model_x)]),
                'y': Pipeline([('poly', poly), ('reg', model_y)]),
                'type': 'polynomial'
            }
            
            # Evaluate
            X_pred = np.column_stack([
                models['x'].predict(X_raw),
                models['y'].predict(X_raw)
            ])
            mse = np.mean((X_pred - X_current) ** 2)
            
        elif method == 'ridge':
            model_x = Ridge(alpha=1.0)
            model_y = Ridge(alpha=1.0)
            
            model_x.fit(X_raw, X_current[:, 0])
            model_y.fit(X_raw, X_current[:, 1])
            
            models = {'x': model_x, 'y': model_y, 'type': 'ridge'}
            
            # Evaluate
            X_pred = np.column_stack([
                model_x.predict(X_raw),
                model_y.predict(X_raw)
            ])
            mse = np.mean((X_pred - X_current) ** 2)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"\n✅ Model trained ({method})")
        print(f"  Mean Squared Error: {mse:.6f}")
        
        return models, {
            'method': method,
            'mse': float(mse),
            'num_samples': len(self.training_data)
        }
    
    def save_mapping_model(self, models, stats, output_path):
        """Save the trained mapping model and statistics."""
        # Save model using pickle
        model_data = {
            'models': models,
            'stats': stats,
            'training_stats': self.analyze_data_distribution(),
            'landmark_indices': {
                'shoulder': self.LEFT_SHOULDER,
                'elbow': self.LEFT_ELBOW,
                'wrist': self.LEFT_WRIST
            }
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save human-readable JSON with stats
        json_path = output_path.replace('.pkl', '.json').replace('.pickle', '.json')
        json_data = {
            'method': stats['method'],
            'mse': stats['mse'],
            'num_samples': stats['num_samples'],
            'training_stats': model_data['training_stats'],
            'landmark_indices': model_data['landmark_indices']
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✅ Model saved:")
        print(f"  Binary: {output_path}")
        print(f"  JSON stats: {json_path}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.use_new_api and self.pose_landmarker:
            self.pose_landmarker.close()
        elif self.pose:
            self.pose.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train keypoint mapping model from data directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python train_keypoint_mapping.py
  
  # Train with custom data directory
  python train_keypoint_mapping.py --data-dir data/ --output mapping_model.pkl
  
  # Use polynomial regression
  python train_keypoint_mapping.py --method polynomial
        """
    )
    parser.add_argument("--data-dir", default="data/", help="Directory containing training images")
    parser.add_argument("--output", default="keypoint_mapping_model.pkl", help="Output model file")
    parser.add_argument("--method", choices=['linear', 'polynomial', 'ridge', 'statistical'], default='statistical',
                       help="Regression method to use (statistical works without sklearn)")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum number of images to process (default: 1000)")
    
    args = parser.parse_args()
    
    try:
        trainer = KeypointMappingTrainer()
        
        # Process training data
        if not trainer.process_training_data(args.data_dir, max_images=args.max_images):
            print("ERROR: No valid keypoint data extracted from images")
            sys.exit(1)
        
        # Analyze data distribution
        stats = trainer.analyze_data_distribution()
        print("\n📊 Data Distribution:")
        print(f"  X range: [{stats['x_raw']['min']:.3f}, {stats['x_raw']['max']:.3f}]")
        print(f"  Y range: [{stats['y_raw']['min']:.3f}, {stats['y_raw']['max']:.3f}]")
        print(f"  Arm length: {stats['arm_length_pixels']['mean']:.1f} ± {stats['arm_length_pixels']['std']:.1f} pixels")
        
        # Train model
        models, training_stats = trainer.train_mapping_model(method=args.method)
        
        # Save model
        trainer.save_mapping_model(models, training_stats, args.output)
        
        trainer.cleanup()
        
        print("\n✅ Training complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
