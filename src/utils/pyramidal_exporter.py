"""
Pyramidal TIFF exporter optimized for preprocessed RGBA TIFF files
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import logging
import math

# Import QApplication for processEvents
from PyQt6.QtWidgets import QApplication

try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..core.fragment import Fragment

class PyramidalExporter:
    """Handles export of stitched pyramidal TIFF files optimized for RGBA inputs"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not TIFFFILE_AVAILABLE:
            self.logger.error("tifffile is required for pyramidal TIFF export")
            raise ImportError("tifffile is required for pyramidal TIFF export")
        
        if not CV2_AVAILABLE:
            self.logger.warning("opencv-python not available - some transformations may be limited")
            
        self.logger.info("Using tifffile for pyramidal TIFF export")
            
    def export_pyramidal_tiff(self, fragments: List[Fragment], output_path: str,
                             selected_levels: List[int], compression: str = "LZW",
                             tile_size: int = 256, progress_callback: Optional[Callable] = None) -> bool:
        """
        Export fragments as a stitched pyramidal TIFF
        
        Args:
            fragments: List of visible Fragment objects
            output_path: Output file path
            selected_levels: List of pyramid levels to export
            compression: Compression method ("LZW", "JPEG", "Deflate", "None")
            tile_size: Tile size for pyramid (default 256)
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            self.logger.info(f"Starting pyramidal TIFF export to {output_path}")
            
            # Filter visible fragments with source files
            visible_fragments = [f for f in fragments if f.visible and f.file_path and os.path.exists(f.file_path)]
            if not visible_fragments:
                raise ValueError("No visible fragments with valid source files to export")
            
            # Validate selected levels
            if not selected_levels:
                raise ValueError("No pyramid levels selected for export")
                
            if progress_callback:
                progress_callback(5, "Analyzing fragment pyramid levels...")
                QApplication.processEvents()
            
            # Analyze available levels in fragments
            fragment_pyramid_info = self._analyze_fragment_pyramids(visible_fragments)
            
            # Process each selected level
            level_images = {}
            total_levels = len(selected_levels)
            
            for i, level in enumerate(selected_levels):
                if progress_callback:
                    progress = int(10 + (i / total_levels) * 70)
                    progress_callback(progress, f"Processing level {level}...")
                    QApplication.processEvents()
                
                try:
                    # Create composite for this level
                    composite = self._create_level_composite(visible_fragments, level, fragment_pyramid_info)
                    if composite is not None:
                        level_images[level] = composite
                        self.logger.info(f"Successfully processed level {level}, size: {composite.shape}")
                    else:
                        self.logger.warning(f"Failed to create composite for level {level}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing level {level}: {str(e)}")
                    continue
            
            if not level_images:
                raise ValueError("No levels could be processed successfully")
            
            # Save as pyramidal TIFF
            if progress_callback:
                progress_callback(85, "Saving pyramidal TIFF...")
                QApplication.processEvents()
            
            success = self._save_pyramidal_tiff(level_images, output_path, compression, tile_size)
            
            if progress_callback:
                progress_callback(100, "Export complete")
                QApplication.processEvents()
                
            self.logger.info("Pyramidal TIFF export completed successfully")
            return success
            
        except Exception as e:
            self.logger.error(f"Pyramidal TIFF export failed: {str(e)}")
            if progress_callback:
                progress_callback(0, f"Export failed: {str(e)}")
            return False
    
    def _analyze_fragment_pyramids(self, fragments: List[Fragment]) -> Dict[str, Dict]:
        """Analyze pyramid structure of each fragment"""
        fragment_info = {}
        
        for fragment in fragments:
            try:
                with tifffile.TiffFile(fragment.file_path) as tif:
                    if hasattr(tif, 'series') and tif.series:
                        series = tif.series[0]
                        if hasattr(series, 'levels') and len(series.levels) > 1:
                            # Pyramidal TIFF
                            levels = []
                            for level_idx, level in enumerate(series.levels):
                                levels.append({
                                    'index': level_idx,
                                    'shape': level.shape,
                                    'dimensions': (level.shape[1], level.shape[0])  # (width, height)
                                })
                            fragment_info[fragment.id] = {
                                'is_pyramidal': True,
                                'levels': levels,
                                'max_level': len(levels) - 1
                            }
                        else:
                            # Single level TIFF
                            page = tif.pages[0]
                            fragment_info[fragment.id] = {
                                'is_pyramidal': False,
                                'levels': [{
                                    'index': 0,
                                    'shape': page.shape,
                                    'dimensions': (page.shape[1], page.shape[0])
                                }],
                                'max_level': 0
                            }
                    else:
                        self.logger.warning(f"Could not analyze pyramid for {fragment.file_path}")
                        
            except Exception as e:
                self.logger.error(f"Error analyzing {fragment.file_path}: {e}")
                
        return fragment_info
    
    def _create_level_composite(self, fragments: List[Fragment], level: int, 
                               pyramid_info: Dict[str, Dict]) -> Optional[np.ndarray]:
        """Create composite image for a specific pyramid level"""
        try:
            # Calculate composite bounds at this level
            bounds = self._calculate_level_bounds(fragments, level, pyramid_info)
            if not bounds:
                return None
            
            min_x, min_y, max_x, max_y = bounds
            width = int(max_x - min_x)
            height = int(max_y - min_y)
            
            if width <= 0 or height <= 0:
                return None
            
            self.logger.info(f"Creating level {level} composite: {width}x{height}")
            
            # Create blank RGBA canvas
            composite = np.zeros((height, width, 4), dtype=np.uint8)
            downsample = 2 ** level
            
            # Composite each fragment
            for fragment in fragments:
                try:
                    # Load fragment at this level
                    fragment_image = self._load_fragment_at_level(fragment, level, pyramid_info)
                    if fragment_image is None:
                        continue
                    
                    # Apply transformations
                    transformed_image = self._apply_transformations(fragment_image, fragment)
                    if transformed_image is None:
                        continue
                    
                    # Calculate position in composite (scale fragment position to this level)
                    scaled_x = int((fragment.x / downsample) - min_x)
                    scaled_y = int((fragment.y / downsample) - min_y)
                    
                    # Composite the fragment
                    self._composite_fragment(composite, transformed_image, scaled_x, scaled_y, fragment.opacity)
                    
                except Exception as e:
                    self.logger.error(f"Error compositing fragment {fragment.name}: {e}")
                    continue
            
            return composite
            
        except Exception as e:
            self.logger.error(f"Failed to create level {level} composite: {str(e)}")
            return None
    
    def _calculate_level_bounds(self, fragments: List[Fragment], level: int, 
                               pyramid_info: Dict[str, Dict]) -> Optional[Tuple[float, float, float, float]]:
        """Calculate composite bounds for a specific level"""
        if not fragments:
            return None
        
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        downsample = 2 ** level
        
        for fragment in fragments:
            try:
                # Get fragment info
                frag_info = pyramid_info.get(fragment.id)
                if not frag_info:
                    continue
                
                # Get dimensions at this level
                if level <= frag_info['max_level']:
                    level_info = frag_info['levels'][level]
                    orig_width, orig_height = level_info['dimensions']
                else:
                    # Use level 0 and calculate downsampled size
                    level_info = frag_info['levels'][0]
                    base_width, base_height = level_info['dimensions']
                    orig_width = int(base_width / downsample)
                    orig_height = int(base_height / downsample)
                
                # Calculate transformed dimensions
                final_width, final_height = self._calculate_transformed_dimensions(
                    orig_width, orig_height, fragment.rotation
                )
                
                # Fragment positions are at level 0 scale, scale them for this level
                scaled_x = fragment.x / downsample
                scaled_y = fragment.y / downsample
                
                min_x = min(min_x, scaled_x)
                min_y = min(min_y, scaled_y)
                max_x = max(max_x, scaled_x + final_width)
                max_y = max(max_y, scaled_y + final_height)
                
            except Exception as e:
                self.logger.warning(f"Could not get bounds for fragment {fragment.name}: {e}")
                continue
        
        if min_x == float('inf'):
            return None
        
        return (min_x, min_y, max_x, max_y)
    
    def _load_fragment_at_level(self, fragment: Fragment, level: int, 
                               pyramid_info: Dict[str, Dict]) -> Optional[np.ndarray]:
        """Load fragment image at specific pyramid level"""
        try:
            frag_info = pyramid_info.get(fragment.id)
            if not frag_info:
                return None
            
            with tifffile.TiffFile(fragment.file_path) as tif:
                if level <= frag_info['max_level']:
                    # Load at requested level
                    if frag_info['is_pyramidal']:
                        image = tif.series[0].levels[level].asarray()
                    else:
                        image = tif.pages[0].asarray()
                else:
                    # Load at highest available level and downsample
                    if frag_info['is_pyramidal']:
                        image = tif.series[0].levels[frag_info['max_level']].asarray()
                    else:
                        image = tif.pages[0].asarray()
                    
                    # Downsample to requested level
                    additional_downsample = 2 ** (level - frag_info['max_level'])
                    if additional_downsample > 1:
                        new_height = max(1, int(image.shape[0] / additional_downsample))
                        new_width = max(1, int(image.shape[1] / additional_downsample))
                        if CV2_AVAILABLE:
                            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                        else:
                            from PIL import Image as PILImage
                            if len(image.shape) == 3:
                                pil_image = PILImage.fromarray(image)
                                pil_image = pil_image.resize((new_width, new_height), PILImage.LANCZOS)
                                image = np.array(pil_image)
                            else:
                                # Handle grayscale
                                pil_image = PILImage.fromarray(image, mode='L')
                                pil_image = pil_image.resize((new_width, new_height), PILImage.LANCZOS)
                                image = np.array(pil_image)
            
            # Ensure RGBA format
            image = self._ensure_rgba_format(image)
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load fragment {fragment.name} at level {level}: {e}")
            return None
    
    def _ensure_rgba_format(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is in RGBA format"""
        if len(image.shape) == 2:
            # Grayscale to RGBA
            rgb = np.stack([image] * 3, axis=2)
            alpha = np.full(image.shape, 255, dtype=image.dtype)
            return np.dstack([rgb, alpha])
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB to RGBA
                alpha = np.full(image.shape[:2], 255, dtype=image.dtype)
                return np.dstack([image, alpha])
            elif image.shape[2] == 4:
                # Already RGBA
                return image
            else:
                raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
    
    def _apply_transformations(self, image: np.ndarray, fragment: Fragment) -> Optional[np.ndarray]:
        """Apply transformations to fragment image"""
        try:
            result = image.copy()
            
            # Apply horizontal flip
            if fragment.flip_horizontal:
                result = np.fliplr(result)
            
            # Apply vertical flip
            if fragment.flip_vertical:
                result = np.flipud(result)
            
            # Apply rotation
            if abs(fragment.rotation) > 0.01:
                result = self._rotate_image(result, fragment.rotation)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to apply transformations: {e}")
            return None
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image using OpenCV or PIL"""
        if abs(angle) < 0.01:
            return image
        
        if CV2_AVAILABLE:
            return self._rotate_with_opencv(image, angle)
        else:
            return self._rotate_with_pil(image, angle)
    
    def _rotate_with_opencv(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image using OpenCV"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        return rotated
    
    def _rotate_with_pil(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image using PIL (fallback)"""
        from PIL import Image as PILImage
        
        # Convert to PIL Image
        if len(image.shape) == 3 and image.shape[2] == 4:
            pil_image = PILImage.fromarray(image, mode='RGBA')
        else:
            pil_image = PILImage.fromarray(image)
        
        # Rotate with transparent background
        rotated_pil = pil_image.rotate(-angle, expand=True, fillcolor=(0, 0, 0, 0))
        
        # Convert back to numpy
        return np.array(rotated_pil)
    
    def _calculate_transformed_dimensions(self, width: int, height: int, rotation: float) -> Tuple[int, int]:
        """Calculate dimensions after rotation transformation"""
        if abs(rotation) < 0.01:
            return (width, height)
        
        # Calculate rotated bounding box
        angle_rad = math.radians(rotation)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        
        new_width = int(width * cos_a + height * sin_a)
        new_height = int(width * sin_a + height * cos_a)
        
        return (new_width, new_height)
    
    def _composite_fragment(self, composite: np.ndarray, fragment_image: np.ndarray,
                           x: int, y: int, opacity: float):
        """Composite fragment with proper alpha blending"""
        try:
            frag_h, frag_w = fragment_image.shape[:2]
            comp_h, comp_w = composite.shape[:2]
            
            # Calculate intersection
            src_x1 = max(0, -x)
            src_y1 = max(0, -y)
            src_x2 = min(frag_w, comp_w - x)
            src_y2 = min(frag_h, comp_h - y)
            
            dst_x1 = max(0, x)
            dst_y1 = max(0, y)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            if src_x2 <= src_x1 or src_y2 <= src_y1:
                return
            
            # Extract regions
            frag_region = fragment_image[src_y1:src_y2, src_x1:src_x2]
            
            # Apply opacity and alpha blend
            frag_alpha = (frag_region[:, :, 3:4] / 255.0) * opacity
            frag_rgb = frag_region[:, :, :3].astype(np.float32)
            
            comp_region = composite[dst_y1:dst_y2, dst_x1:dst_x2]
            comp_alpha = comp_region[:, :, 3:4] / 255.0
            comp_rgb = comp_region[:, :, :3].astype(np.float32)
            
            # Alpha blending
            out_alpha = frag_alpha + (1 - frag_alpha) * comp_alpha
            
            # Avoid division by zero
            mask = out_alpha[:, :, 0] > 0
            out_rgb = np.zeros_like(frag_rgb)
            
            if np.any(mask):
                out_rgb[mask, :] = (frag_alpha[mask, :] * frag_rgb[mask, :] + 
                                   (1 - frag_alpha[mask, :]) * comp_rgb[mask, :]) / out_alpha[mask, :]
            
            # Update composite
            composite[dst_y1:dst_y2, dst_x1:dst_x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
            composite[dst_y1:dst_y2, dst_x1:dst_x2, 3:4] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Failed to composite fragment: {e}")
    
    def _save_pyramidal_tiff(self, level_images: Dict[int, np.ndarray], output_path: str,
                           compression: str, tile_size: int) -> bool:
        """Save images as a proper pyramidal TIFF structure"""
        try:
            # Configure compression
            compression_map = {
                "LZW": "lzw",
                "JPEG": "jpeg",
                "Deflate": "zlib",
                "None": None
            }
            tiff_compression = compression_map.get(compression)
            
            # Prepare images in pyramid order (level 0 first)
            sorted_levels = sorted(level_images.keys())
            
            if len(sorted_levels) == 1:
                # Single level
                image = level_images[sorted_levels[0]]
                save_kwargs = {
                    'compression': tiff_compression,
                    'photometric': 'rgb',
                    'tile': (tile_size, tile_size),
                    'extrasamples': [1]  # Associated alpha
                }
                save_kwargs = {k: v for k, v in save_kwargs.items() if v is not None}
                
                tifffile.imwrite(output_path, image, **save_kwargs)
            else:
                # Multi-level pyramid
                with tifffile.TiffWriter(output_path, bigtiff=True) as tiff_writer:
                    for i, level in enumerate(sorted_levels):
                        image = level_images[level]
                        
                        # Convert RGBA to RGB for JPEG compression
                        if tiff_compression == "jpeg" and image.shape[2] == 4:
                            image = self._rgba_to_rgb(image)
                            extrasamples = None
                        else:
                            extrasamples = [1]  # Associated alpha
                        
                        save_kwargs = {
                            'compression': tiff_compression,
                            'photometric': 'rgb',
                            'tile': (tile_size, tile_size),
                            'extrasamples': extrasamples
                        }
                        save_kwargs = {k: v for k, v in save_kwargs.items() if v is not None}
                        
                        tiff_writer.write(image, **save_kwargs)
            
            self.logger.info(f"Saved pyramidal TIFF with {len(sorted_levels)} levels")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save pyramidal TIFF: {e}")
            return False
    
    def _rgba_to_rgb(self, rgba_image: np.ndarray, background_color=(255, 255, 255)) -> np.ndarray:
        """Convert RGBA image to RGB with specified background color"""
        if rgba_image.shape[2] != 4:
            return rgba_image[:, :, :3]
        
        rgb = rgba_image[:, :, :3].astype(np.float32)
        alpha = rgba_image[:, :, 3:4].astype(np.float32) / 255.0
        
        background = np.full_like(rgb, background_color, dtype=np.float32)
        result = (alpha * rgb + (1 - alpha) * background).astype(np.uint8)
        
        return result