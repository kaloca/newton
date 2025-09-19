#!/usr/bin/env python3
"""
Test script demonstrating ViewerSRTX optimization options.
Run with: uv run --extra srtx test_viewer_optimizations.py
"""

import newton

if __name__ == "__main__":
    # Fast preview settings (lower quality, faster rendering)
    viewer_fast = newton.viewer.ViewerSRTX(
        output_dir="srtx_fast_preview",
        num_frames=100,
        fps=30,  # Lower FPS for preview
        resolution=(640, 480),  # Lower resolution
        image_format="jpeg",  # JPEG for small file size
        jpeg_quality=70,  # Lower quality for smaller files
        render_every_n_frames=5,  # Only render every 5th frame
        camera_position=(6.0, -8.0, 5.0),
        camera_target=(0.0, 0.0, 1.5),
    )

    # High quality settings (for final output)
    viewer_hq = newton.viewer.ViewerSRTX(
        output_dir="srtx_high_quality",
        num_frames=100,
        fps=60,
        resolution=(1920, 1080),  # Full HD
        image_format="png",  # PNG for lossless quality
        render_every_n_frames=1,  # Render all frames
        camera_position=(6.0, -8.0, 5.0),
        camera_target=(0.0, 0.0, 1.5),
    )

    # Balanced settings (good quality, reasonable speed)
    viewer_balanced = newton.viewer.ViewerSRTX(
        output_dir="srtx_balanced",
        num_frames=100,
        fps=60,
        resolution=(1280, 720),  # 720p
        image_format="webp",  # WebP for good compression with quality
        jpeg_quality=85,  # Good quality
        render_every_n_frames=1,  # Render all frames
        camera_position=(6.0, -8.0, 5.0),
        camera_target=(0.0, 0.0, 1.5),
    )

    # Choose which viewer to use
    viewer = viewer_fast  # Change to viewer_hq or viewer_balanced as needed

    # Run the basic shapes example
    import newton.examples.basic.example_basic_shapes as example

    model = example.build_model()
    example.run_sim(model, viewer)

    print("\n=== Performance Comparison ===")
    print("Fast Preview: 640x480 JPEG @ 70% quality, every 5th frame")
    print("  Estimated file size per frame: ~50-100 KB")
    print("  Total frames rendered: 20 (every 5th)")
    print("\nBalanced: 1280x720 WebP @ 85% quality, all frames")
    print("  Estimated file size per frame: ~200-400 KB")
    print("  Total frames rendered: 100")
    print("\nHigh Quality: 1920x1080 PNG, all frames")
    print("  Estimated file size per frame: ~2-8 MB")
    print("  Total frames rendered: 100")
