"""Tests for image processing functions."""

import numpy as np
import pytest
import cv2

from src.processing import (
    detect_edges,
    apply_gaussian_blur,
    filter_colors,
    find_contours,
    draw_contours,
    resize_image,
    convert_to_grayscale,
)


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a 100x100 image with a white square on black background
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = [255, 255, 255]
    return img


@pytest.fixture
def sample_grayscale():
    """Create a simple grayscale test image."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[25:75, 25:75] = 255
    return img


class TestEdgeDetection:
    """Tests for edge detection."""
    
    def test_detect_edges_returns_binary(self, sample_grayscale):
        """Test that edge detection returns a binary image."""
        edges = detect_edges(sample_grayscale)
        assert edges.dtype == np.uint8
        assert set(np.unique(edges)).issubset({0, 255})
    
    def test_detect_edges_color_image(self, sample_image):
        """Test edge detection on color image."""
        edges = detect_edges(sample_image)
        assert edges.dtype == np.uint8
        assert len(edges.shape) == 2  # Output should be grayscale
    
    def test_detect_edges_invalid_input(self):
        """Test edge detection with invalid input."""
        with pytest.raises(ValueError):
            detect_edges(np.array([]))
    
    def test_detect_edges_with_thresholds(self, sample_grayscale):
        """Test edge detection with custom thresholds."""
        edges = detect_edges(sample_grayscale, low_threshold=30, high_threshold=100)
        assert edges.shape == sample_grayscale.shape


class TestBlur:
    """Tests for Gaussian blur."""
    
    def test_apply_gaussian_blur(self, sample_image):
        """Test that blur is applied correctly."""
        blurred = apply_gaussian_blur(sample_image)
        assert blurred.shape == sample_image.shape
        assert blurred.dtype == sample_image.dtype
    
    def test_blur_with_custom_kernel(self, sample_image):
        """Test blur with custom kernel size."""
        blurred = apply_gaussian_blur(sample_image, kernel_size=(7, 7))
        assert blurred.shape == sample_image.shape
    
    def test_blur_invalid_kernel_size(self, sample_image):
        """Test blur with invalid kernel size."""
        with pytest.raises(ValueError):
            apply_gaussian_blur(sample_image, kernel_size=(4, 4))  # Must be odd
    
    def test_blur_invalid_input(self):
        """Test blur with invalid input."""
        with pytest.raises(ValueError):
            apply_gaussian_blur(np.array([]))


class TestColorFiltering:
    """Tests for color filtering."""
    
    def test_filter_colors_hsv(self, sample_image):
        """Test color filtering in HSV space."""
        lower = np.array([0, 0, 200])
        upper = np.array([180, 50, 255])
        mask = filter_colors(sample_image, lower, upper, "HSV")
        assert mask.dtype == np.uint8
        assert len(mask.shape) == 2
    
    def test_filter_colors_invalid_colorspace(self, sample_image):
        """Test color filtering with invalid color space."""
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, 255])
        with pytest.raises(ValueError):
            filter_colors(sample_image, lower, upper, "INVALID")
    
    def test_filter_colors_grayscale_input(self, sample_grayscale):
        """Test color filtering with grayscale input."""
        lower = np.array([0, 0, 0])
        upper = np.array([255, 255, 255])
        with pytest.raises(ValueError):
            filter_colors(sample_grayscale, lower, upper)


class TestContours:
    """Tests for contour detection."""
    
    def test_find_contours(self, sample_grayscale):
        """Test finding contours in binary image."""
        contours = find_contours(sample_grayscale)
        assert isinstance(contours, list)
        assert len(contours) > 0
    
    def test_find_contours_invalid_input(self):
        """Test contour detection with invalid input."""
        with pytest.raises(ValueError):
            find_contours(np.array([]))
    
    def test_draw_contours(self, sample_image):
        """Test drawing contours on image."""
        # Create a binary image
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        contours = find_contours(gray)
        
        output = draw_contours(sample_image, contours)
        assert output.shape == sample_image.shape
        assert output.dtype == sample_image.dtype
    
    def test_draw_contours_with_min_area(self, sample_image):
        """Test drawing contours with minimum area filter."""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        contours = find_contours(gray)
        
        output = draw_contours(sample_image, contours, min_area=1000)
        assert output.shape == sample_image.shape


class TestResize:
    """Tests for image resizing."""
    
    def test_resize_by_width(self, sample_image):
        """Test resizing by specifying width."""
        resized = resize_image(sample_image, width=50)
        assert resized.shape[1] == 50
        assert resized.shape[0] == 50  # Aspect ratio maintained
    
    def test_resize_by_height(self, sample_image):
        """Test resizing by specifying height."""
        resized = resize_image(sample_image, height=200)
        assert resized.shape[0] == 200
        assert resized.shape[1] == 200  # Aspect ratio maintained
    
    def test_resize_both_dimensions(self, sample_image):
        """Test resizing with both width and height specified."""
        resized = resize_image(sample_image, width=150, height=75)
        assert resized.shape[1] == 150
        assert resized.shape[0] == 75
    
    def test_resize_no_dimensions(self, sample_image):
        """Test resize with no dimensions specified."""
        with pytest.raises(ValueError):
            resize_image(sample_image)
    
    def test_resize_invalid_input(self):
        """Test resize with invalid input."""
        with pytest.raises(ValueError):
            resize_image(np.array([]), width=100)


class TestGrayscale:
    """Tests for grayscale conversion."""
    
    def test_convert_to_grayscale(self, sample_image):
        """Test converting color image to grayscale."""
        gray = convert_to_grayscale(sample_image)
        assert len(gray.shape) == 2
        assert gray.dtype == np.uint8
    
    def test_convert_already_grayscale(self, sample_grayscale):
        """Test converting already grayscale image."""
        gray = convert_to_grayscale(sample_grayscale)
        assert np.array_equal(gray, sample_grayscale)
    
    def test_convert_invalid_input(self):
        """Test grayscale conversion with invalid input."""
        with pytest.raises(ValueError):
            convert_to_grayscale(np.array([]))