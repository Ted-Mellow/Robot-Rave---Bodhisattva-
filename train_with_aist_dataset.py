#!/usr/bin/env python3
"""
Train Keypoint Mapping using AIST++ Dataset
Downloads and uses the AIST++ dance dataset to train improved keypoint mappings.

Usage:
    python train_with_aist_dataset.py [--download-dir ./aist_data] [--output mapping_model.pkl]
"""

import sys
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pickle
import urllib.request
import zipfile
import shutil

try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  scikit-learn not found. Install with: pip install scikit-learn")
    print("   Will use simple statistical mapping instead.")


class AISTDatasetTrainer:
    """Train keypoint mapping using AIST++ dataset."""
    
    # COCO keypoint format indices (AIST++ uses COCO format)
    # Left arm: 5 (left_shoulder), 7 (left_elbow), 9 (left_wrist)
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 7
    LEFT_WRIST = 9
    
    def __init__(self, download_dir="./aist_data"):
        """Initialize trainer with download directory."""
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.training_data = []
        
    def download_aist_keypoints(self, force=False):
        """Download AIST++ 2D keypoints dataset."""
        keypoints_url = "https://storage.googleapis.com/aist_plusplus_public/20210324/keypoints2d.zip"
        keypoints_file = self.download_dir / "keypoints2d.zip"
        keypoints_dir = self.download_dir / "keypoints2d"
        
        if keypoints_dir.exists() and not force:
            print(f"✅ AIST++ keypoints already downloaded at {keypoints_dir}")
            return keypoints_dir
        
        print(f"Downloading AIST++ 2D keypoints dataset...")
        print(f"URL: {keypoints_url}")
        print(f"This may take a while (~1.2GB)...")
        print(f"\nNote: This downloads only the keypoint annotations.")
        print(f"      To download videos, use: python download_aist_videos.py --download_folder <folder>")
        
        try:
            # Download with progress
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB)")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(keypoints_url, keypoints_file, show_progress)
            print("\n✅ Download complete, extracting...")
            
            # Extract
            with zipfile.ZipFile(keypoints_file, 'r') as zip_ref:
                zip_ref.extractall(self.download_dir)
            
            # Clean up zip file
            keypoints_file.unlink()
            
            print(f"✅ Extracted to {keypoints_dir}")
            return keypoints_dir
            
        except Exception as e:
            print(f"\n❌ Failed to download: {e}")
            print("\nYou can manually download from:")
            print("  https://storage.googleapis.com/aist_plusplus_public/20210324/keypoints2d.zip")
            print(f"  And extract to: {keypoints_dir}")
            return None
    
    def load_keypoints_file(self, pkl_path):
        """Load a single AIST++ keypoints file."""
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # AIST++ format: keypoints2d shape is (9, N, 17, 3)
            # 9 cameras, N frames, 17 COCO keypoints, 3 values (x, y, confidence)
            if 'keypoints2d' in data:
                keypoints2d = data['keypoints2d']
                # Use first camera view (camera 0)
                camera_keypoints = keypoints2d[0]  # Shape: (N, 17, 3)
                return camera_keypoints
            return None
            
        except Exception as e:
            print(f"⚠️  Error loading {pkl_path}: {e}")
            return None
    
    def extract_arm_keypoints(self, keypoints_array):
        """
        Extract left arm keypoints from COCO format keypoints.
        
        Args:
            keypoints_array: Array of shape (N, 17, 3) where last dim is (x, y, confidence)
        
        Returns:
            List of training samples with normalized coordinates
        """
        samples = []
        
        for frame_keypoints in keypoints_array:
            # Get left arm keypoints (COCO format)
            shoulder = frame_keypoints[self.LEFT_SHOULDER]  # [x, y, confidence]
            elbow = frame_keypoints[self.LEFT_ELBOW]
            wrist = frame_keypoints[self.LEFT_WRIST]
            
            # Check visibility/confidence
            if (shoulder[2] < 0.3 or elbow[2] < 0.3 or wrist[2] < 0.3):
                continue
            
            # Calculate arm vector
            shoulder_pos = np.array([shoulder[0], shoulder[1]])
            elbow_pos = np.array([elbow[0], elbow[1]])
            wrist_pos = np.array([wrist[0], wrist[1]])
            
            # Calculate arm length
            arm_vec = wrist_pos - shoulder_pos
            arm_length = np.linalg.norm(arm_vec)
            
            if arm_length < 1:
                continue
            
            # Calculate raw normalized coordinates (same as our extraction)
            x_raw = (wrist_pos[0] - shoulder_pos[0]) / arm_length
            y_raw = (shoulder_pos[1] - wrist_pos[1]) / arm_length  # Y increases downward in images
            
            samples.append({
                'x_raw': float(x_raw),
                'y_raw': float(y_raw),
                'arm_length_pixels': float(arm_length),
                'shoulder_pos': shoulder_pos.tolist(),
                'wrist_pos': wrist_pos.tolist(),
                'confidence': min(shoulder[2], elbow[2], wrist[2])
            })
        
        return samples
    
    def process_aist_dataset(self, keypoints_dir, max_files=None):
        """Process all keypoint files in AIST++ dataset."""
        keypoints_path = Path(keypoints_dir)
        if not keypoints_path.exists():
            print(f"❌ Keypoints directory not found: {keypoints_dir}")
            return False
        
        # Find all .pkl files
        pkl_files = sorted(keypoints_path.glob("*.pkl"))
        
        if max_files:
            pkl_files = pkl_files[:max_files]
        
        print(f"Found {len(pkl_files)} keypoint files")
        print("Processing AIST++ dataset...")
        
        total_samples = 0
        successful_files = 0
        
        for i, pkl_file in enumerate(pkl_files):
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(pkl_files)} files... ({total_samples} samples extracted)")
            
            keypoints_array = self.load_keypoints_file(pkl_file)
            if keypoints_array is None:
                continue
            
            samples = self.extract_arm_keypoints(keypoints_array)
            if samples:
                self.training_data.extend(samples)
                total_samples += len(samples)
                successful_files += 1
        
        print(f"\n✅ Processing complete:")
        print(f"  Successful files: {successful_files}/{len(pkl_files)}")
        print(f"  Total training samples: {len(self.training_data)}")
        
        return len(self.training_data) > 0
    
    def analyze_data_distribution(self):
        """Analyze the distribution of keypoint data."""
        if not self.training_data:
            return None
        
        x_raw = np.array([d['x_raw'] for d in self.training_data])
        y_raw = np.array([d['y_raw'] for d in self.training_data])
        arm_lengths = np.array([d['arm_length_pixels'] for d in self.training_data])
        confidences = np.array([d['confidence'] for d in self.training_data])
        
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
            'confidence': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences))
            },
            'num_samples': len(self.training_data)
        }
        
        return stats
    
    def train_mapping_model(self, method='polynomial'):
        """Train mapping model (same as train_keypoint_mapping.py)."""
        if not self.training_data:
            raise ValueError("No training data available.")
        
        # Prepare training data
        X_raw = np.array([[d['x_raw'], d['y_raw']] for d in self.training_data])
        X_current = (X_raw + 1.0) / 2.0
        X_current = np.clip(X_current, 0.0, 1.0)
        
        if not HAS_SKLEARN or method == 'statistical':
            # Statistical mapping
            x_min, x_max = np.min(X_raw[:, 0]), np.max(X_raw[:, 0])
            y_min, y_max = np.min(X_raw[:, 1]), np.max(X_raw[:, 1])
            
            x_scale = 1.0 / (x_max - x_min + 1e-6)
            x_offset = -x_min
            y_scale = 1.0 / (y_max - y_min + 1e-6)
            y_offset = -y_min
            
            models = {
                'x': {'scale': float(x_scale), 'offset': float(x_offset), 'min': float(x_min), 'max': float(x_max)},
                'y': {'scale': float(y_scale), 'offset': float(y_offset), 'min': float(y_min), 'max': float(y_max)},
                'type': 'statistical'
            }
            
            X_pred = np.column_stack([
                np.clip((X_raw[:, 0] + x_offset) * x_scale, 0.0, 1.0),
                np.clip((X_raw[:, 1] + y_offset) * y_scale, 0.0, 1.0)
            ])
            mse = np.mean((X_pred - X_current) ** 2)
            method = 'statistical'
            
        elif method == 'polynomial':
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
            
            X_pred = np.column_stack([
                models['x'].predict(X_raw),
                models['y'].predict(X_raw)
            ])
            mse = np.mean((X_pred - X_current) ** 2)
            
        elif method == 'linear':
            model_x = LinearRegression()
            model_y = LinearRegression()
            
            model_x.fit(X_raw, X_current[:, 0])
            model_y.fit(X_raw, X_current[:, 1])
            
            models = {'x': model_x, 'y': model_y, 'type': 'linear'}
            
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
            'num_samples': len(self.training_data),
            'dataset': 'AIST++'
        }
    
    def save_mapping_model(self, models, stats, output_path):
        """Save the trained mapping model."""
        model_data = {
            'models': models,
            'stats': stats,
            'training_stats': self.analyze_data_distribution(),
            'landmark_indices': {
                'shoulder': self.LEFT_SHOULDER,
                'elbow': self.LEFT_ELBOW,
                'wrist': self.LEFT_WRIST,
                'format': 'COCO'  # AIST++ uses COCO format
            },
            'dataset': 'AIST++'
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        json_path = output_path.replace('.pkl', '.json').replace('.pickle', '.json')
        json_data = {
            'method': stats['method'],
            'mse': stats['mse'],
            'num_samples': stats['num_samples'],
            'dataset': stats.get('dataset', 'custom'),
            'training_stats': model_data['training_stats'],
            'landmark_indices': model_data['landmark_indices']
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"\n✅ Model saved:")
        print(f"  Binary: {output_path}")
        print(f"  JSON stats: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train keypoint mapping using AIST++ dance dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and train with AIST++ dataset
  python train_with_aist_dataset.py
  
  # Use existing download
  python train_with_aist_dataset.py --no-download
  
  # Limit number of files for faster training
  python train_with_aist_dataset.py --max-files 50
        """
    )
    parser.add_argument("--download-dir", default="./aist_data", help="Directory for AIST++ data")
    parser.add_argument("--output", default="keypoint_mapping_aist.pkl", help="Output model file")
    parser.add_argument("--method", choices=['linear', 'polynomial', 'ridge', 'statistical'], 
                       default='statistical', help="Regression method")
    parser.add_argument("--no-download", action="store_true", help="Skip download, use existing data")
    parser.add_argument("--max-files", type=int, help="Maximum number of keypoint files to process")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    parser.add_argument("--download-videos", action="store_true", 
                       help="Also download videos using official downloader (requires agreement to terms)")
    
    args = parser.parse_args()
    
    try:
        trainer = AISTDatasetTrainer(download_dir=args.download_dir)
        
        # Download videos if requested (using official downloader)
        if args.download_videos:
            print("\n📥 Downloading AIST++ videos using official downloader...")
            print("   This will download all videos (~large dataset)")
            print("   You can cancel and use --no-download to skip videos")
            
            import subprocess
            video_dir = Path(args.download_dir) / "videos"
            video_dir.mkdir(exist_ok=True)
            
            try:
                subprocess.run([
                    sys.executable, "download_aist_videos.py",
                    "--download_folder", str(video_dir),
                    "--num_processes", "5"
                ], check=True)
                print("✅ Videos downloaded")
            except subprocess.CalledProcessError:
                print("⚠️  Video download failed or cancelled")
            except FileNotFoundError:
                print("⚠️  download_aist_videos.py not found, skipping video download")
        
        # Download keypoints if needed
        if not args.no_download:
            keypoints_dir = trainer.download_aist_keypoints(force=args.force_download)
            if keypoints_dir is None:
                print("ERROR: Could not download or find AIST++ keypoints")
                sys.exit(1)
        else:
            keypoints_dir = Path(args.download_dir) / "keypoints2d"
            if not keypoints_dir.exists():
                print(f"ERROR: Keypoints directory not found: {keypoints_dir}")
                print("   Run without --no-download to download first")
                sys.exit(1)
        
        # Process dataset
        if not trainer.process_aist_dataset(keypoints_dir, max_files=args.max_files):
            print("ERROR: No valid keypoint data extracted")
            sys.exit(1)
        
        # Analyze data
        stats = trainer.analyze_data_distribution()
        print("\n📊 Data Distribution:")
        print(f"  X range: [{stats['x_raw']['min']:.3f}, {stats['x_raw']['max']:.3f}]")
        print(f"  Y range: [{stats['y_raw']['min']:.3f}, {stats['y_raw']['max']:.3f}]")
        print(f"  Arm length: {stats['arm_length_pixels']['mean']:.1f} ± {stats['arm_length_pixels']['std']:.1f} pixels")
        print(f"  Avg confidence: {stats['confidence']['mean']:.3f}")
        
        # Train model
        models, training_stats = trainer.train_mapping_model(method=args.method)
        
        # Save model
        trainer.save_mapping_model(models, training_stats, args.output)
        
        print("\n✅ Training complete!")
        print(f"   Model trained on {training_stats['num_samples']} samples from AIST++ dataset")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
