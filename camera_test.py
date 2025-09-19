#!/usr/bin/env python3
"""
Quick script to test different camera positions for Newton ViewerSRTX
"""

import newton
import newton.viewer
import warp as wp


def test_camera_positions():
    """Test different camera positions to find good viewing angles."""

    # Camera positions to test (position, target, description)
    camera_configs = [
        # Original
        ((6.0, 6.0, 5.0), (0.0, 0.0, 2.0), "original_diagonal"),
        # Side views
        ((8.0, 0.0, 4.0), (0.0, 0.0, 1.0), "side_view"),
        ((0.0, -8.0, 4.0), (0.0, 0.0, 1.0), "front_view"),
        # Better angled views
        ((5.0, -7.0, 5.0), (0.0, 0.0, 1.5), "front_angle"),
        ((7.0, -5.0, 6.0), (0.0, 1.0, 1.0), "front_right_angle"),
        # Top-down views
        ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), "top_down"),
        ((2.0, -2.0, 8.0), (0.0, 0.0, 1.0), "top_angle"),
    ]

    print("\nTesting different camera positions for Newton shapes scene...")
    print("=" * 60)

    for i, (pos, target, desc) in enumerate(camera_configs):
        print(f"\nTest {i + 1}: {desc}")
        print(f"  Position: {pos}")
        print(f"  Target: {target}")
        print(f"  Output: camera_test_{i}_{desc}/")

        # Create simple scene with a ground and box
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 2.0), q=wp.quat_identity()))
        builder.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
        model = builder.finalize()

        # Create viewer with test camera position
        viewer = newton.viewer.ViewerSRTX(
            output_dir=f"camera_test/camera_test_{i}_{desc}",
            num_frames=1,  # Just one frame for testing
            camera_position=pos,
            camera_target=target,
            resolution=(640, 480),  # Smaller for faster testing
            host="localhost",
            port=8081,
        )

        # Render one frame
        state = model.state()
        viewer.set_model(model)
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
        viewer.close()

        print(f"  ✓ Rendered")

    print("\n" + "=" * 60)
    print("Camera testing complete! Check the output directories to see which angle works best.")
    print("\nRecommended camera position for basic shapes example:")
    print("  camera_position=(5.0, -7.0, 5.0)")
    print("  camera_target=(0.0, 0.0, 1.5)")


if __name__ == "__main__":
    test_camera_positions()
