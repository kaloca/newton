#!/usr/bin/env python3
"""
SensorRTX viewer backend for Newton physics simulations.

This backend creates a USD stage similar to ViewerUSD but instead of saving to disk,
it sends the stage to SensorRTX for photorealistic rendering.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from pathlib import Path

try:
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, UsdRender
    from sensor_rtx import Channel, ChangeNumber, ChannelOptions, StageName
    from sensor_rtx.usd import render, write
except ImportError as e:
    raise ImportError(
        "SensorRTX dependencies not installed. Install with:\npip install usd-core pillow sensor-rtx"
    ) from e

from .viewer_usd import ViewerUSD  # Inherit most USD functionality


class ViewerSRTX(ViewerUSD):
    """
    SensorRTX viewer backend for Newton physics simulations.

    This viewer creates USD stages and renders them using NVIDIA's SensorRTX
    for photorealistic ray-traced visualization.
    """

    def __init__(
        self,
        output_dir="srtx_renders",
        fps=60,
        up_axis="Z",
        num_frames=None,
        host="localhost",
        port=8081,
        camera_position=(5.0, 5.0, 5.0),
        camera_target=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
    ):
        """
        Initialize the SensorRTX viewer backend.

        Args:
            output_dir (str): Directory to save rendered images.
            fps (int): Frames per second for time sampling.
            up_axis (str): USD up axis, either 'Y' or 'Z'.
            num_frames (int, optional): Maximum number of frames to render.
            host (str): SensorRTX server host.
            port (int): SensorRTX server port.
            camera_position (tuple): Initial camera position (x, y, z).
            camera_target (tuple): Camera look-at target (x, y, z).
            resolution (tuple): Render resolution (width, height).
        """
        # Create a USD file in the output directory (not temp!)
        self.usd_path = str(Path(output_dir) / "newton_stage.usd")

        # Initialize parent with the USD path
        super().__init__(output_path=self.usd_path, fps=fps, up_axis=up_axis, num_frames=num_frames)

        # SensorRTX settings
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Camera settings
        self.camera_position = camera_position
        self.camera_target = camera_target
        self.resolution = resolution

        # SensorRTX connections (will be initialized on first frame)
        self.channel = None
        self.write_client = None
        self.render_client = None
        self.runtime_stage = None
        self.view = None

        # Camera path in USD - use simpler path for compatibility
        self.camera_path = "/Camera"
        # Create camera in USD stage
        self._create_camera()
        # Add lighting to the scene
        self._create_lights()

        # Save and ensure the stage is written to disk
        self.stage.GetRootLayer().Save()
        print(f"USD stage saved with camera at {self.camera_path}")

        # Track if we're connected
        self._connected = False

        # Setup async runner thread (following sensorrtx_mujoco pattern)
        self._loop = None
        self._thread = None
        self._initialize_async_runner()

        print(f"ViewerSRTX: Will render to {self.output_dir}")
        print(f"ViewerSRTX: Connecting to SensorRTX at {host}:{port}")

    def _initialize_async_runner(self):
        """Initialize a background thread with event loop for async operations."""
        if self._loop is None and self._thread is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="SRTXAsyncRunner", daemon=True)
            self._thread.start()

    def _run_async(self, coro):
        """Run an async coroutine in the background thread (blocking)."""
        if not self._thread or not self._thread.is_alive():
            raise RuntimeError("Async runner thread is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def _create_camera(self):
        """Create a camera AND RenderProduct in the USD stage (following SensorRTX pattern)."""
        # Create the actual camera
        camera = UsdGeom.Camera.Define(self.stage, self.camera_path)

        # Set camera properties
        camera.GetFocusDistanceAttr().Set(10.0)
        camera.GetFStopAttr().Set(5.6)
        camera.GetFocalLengthAttr().Set(24.0)

        # Set resolution via aperture (sensor size)
        aspect_ratio = self.resolution[0] / self.resolution[1]
        camera.GetHorizontalApertureAttr().Set(36.0)  # 35mm sensor width
        camera.GetVerticalApertureAttr().Set(36.0 / aspect_ratio)

        # Position camera using transform
        xform = UsdGeom.Xformable(camera)

        # Calculate look-at matrix
        eye = Gf.Vec3d(*self.camera_position)
        center = Gf.Vec3d(*self.camera_target)

        # For Newton with Z-up, we want the camera's local Y to align with world Z
        # This ensures the horizon stays level
        world_up = Gf.Vec3d(0, 0, 1) if self.up_axis == "Z" else Gf.Vec3d(0, 1, 0)

        # Compute the view direction (from eye to center)
        view_dir = center - eye
        view_dir.Normalize()

        # Compute right vector (perpendicular to view and up)
        right = Gf.Cross(view_dir, world_up)
        right.Normalize()

        # Recompute up to ensure orthogonality (this keeps horizon level)
        up = Gf.Cross(right, view_dir)
        up.Normalize()

        # Build the camera transform matrix manually
        # In USD, cameras look down -Z with Y up in their local space
        # So we need to map: right->X, up->Y, -view_dir->Z
        mat = Gf.Matrix4d(1.0)
        mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
        mat.SetRow(1, Gf.Vec4d(up[0], up[1], up[2], 0))
        mat.SetRow(2, Gf.Vec4d(-view_dir[0], -view_dir[1], -view_dir[2], 0))
        mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))

        # Apply the transform
        xform.ClearXformOpOrder()
        xform_op = xform.AddTransformOp()
        xform_op.Set(mat)

        # Create RenderProduct (this is what SensorRTX actually renders!)
        # Create Render scope if it doesn't exist
        render_scope = UsdGeom.Scope.Define(self.stage, "/Render")

        # Create RenderProduct that references our camera
        self.render_product_path = "/Render/RGBDCam"
        render_product = UsdRender.Product.Define(self.stage, self.render_product_path)
        render_product.GetCameraRel().SetTargets([camera.GetPath()])
        render_product.GetResolutionAttr().Set(Gf.Vec2i(self.resolution[0], self.resolution[1]))

        # Create RenderVars for output types
        render_vars_path = "/Render/Vars"
        UsdGeom.Scope.Define(self.stage, render_vars_path)

        ldr_color_var = UsdRender.Var.Define(self.stage, f"{render_vars_path}/ldrColor")
        ldr_color_var.GetSourceNameAttr().Set("LdrColor")

        # Add the RenderVar to the RenderProduct
        render_product.GetOrderedVarsRel().AddTarget(ldr_color_var.GetPath())

        print(f"Created Camera at {self.camera_path} and RenderProduct at {self.render_product_path}")

    def _create_lights(self):
        """Create lights in the USD stage for proper rendering."""
        # Add a dome light for ambient lighting
        dome_light = UsdLux.DomeLight.Define(self.stage, "/Render/DomeLight")
        dome_light.GetIntensityAttr().Set(500.0)  # Moderate intensity

        # Add a distant light for directional lighting (like the sun)
        distant_light = UsdLux.DistantLight.Define(self.stage, "/Render/DistantLight")
        distant_light.GetIntensityAttr().Set(1000.0)
        distant_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 0.95))  # Slightly warm white

        # Set the direction of the distant light (pointing down and slightly forward)
        xform = UsdGeom.Xformable(distant_light)
        xform.ClearXformOpOrder()
        xform_op = xform.AddRotateXOp()
        xform_op.Set(-45.0)  # Angle down 45 degrees

        print("Created lighting: DomeLight and DistantLight")

    async def _connect_to_srtx(self):
        """Establish connection to SensorRTX server."""
        if self._connected:
            return

        try:
            self.channel = Channel(ChannelOptions(host=self.host, port=self.port, ssl=False))
            self.write_client = write.UsdWriteClient(self.channel)
            self.render_client = render.UsdRenderClient(self.channel)

            # Open channel and wait for healthy
            await self.channel.__aenter__()
            await self.write_client.wait_for_healthy(max_wait_time=10)

            self._connected = True
            print("Connected to SensorRTX")

        except Exception as e:
            print(f"Failed to connect to SensorRTX: {e}")
            raise

    async def _render_frame_async(self):
        """Render current frame using SensorRTX."""
        # Connect if not already connected
        await self._connect_to_srtx()

        # Save current USD state to file
        self.stage.GetRootLayer().Save()

        # Create or update runtime stage in SensorRTX
        if self.runtime_stage is None:
            # First frame - create new runtime stage with file:// URL
            usd_url = f"file://{os.path.abspath(self.usd_path)}"
            self.runtime_stage = await self.write_client.create_runtime_stage(
                f"newton-sim-{id(self)}", source=write.StorageSource(usd_url=usd_url)
            )
            print(f"Created runtime stage from {usd_url}")

            # Wait for stage to be ready and list available prims
            await asyncio.sleep(1.0)  # Give more time for stage to load

            # Debug: Try to verify what's in the runtime stage
            print(f"Runtime stage name: {self.runtime_stage.name}")

            # Create view for the RenderProduct (NOT the camera directly!)
            render_product_path = getattr(self, "render_product_path", "/Render/RGBDCam")
            print(f"Creating view for RenderProduct at: {render_product_path}")

            try:
                self.view = await self.render_client.create_view(
                    prim_path=render.PrimPath(render_product_path),
                    view_id="newton-render-view",
                    parent=StageName(self.runtime_stage.name),
                )
                print(f"Created view for RenderProduct at {render_product_path}")
            except Exception as e:
                print(f"Failed to create view for RenderProduct: {e}")
                # Try alternative RenderProduct paths
                alt_paths = ["/Render/RenderProduct_0", "/Render/RGBDCam", self.camera_path]
                for alt_path in alt_paths:
                    try:
                        print(f"Trying alternative path: {alt_path}")
                        self.view = await self.render_client.create_view(
                            prim_path=render.PrimPath(alt_path),
                            view_id="newton-render-view",
                            parent=StageName(self.runtime_stage.name),
                        )
                        print(f"Successfully created view with path: {alt_path}")
                        break
                    except Exception as alt_e:
                        print(f"  Failed: {alt_e}")
                        continue
                if not self.view:
                    raise RuntimeError("Could not create view with any RenderProduct path") from None
        else:
            # For updates, we should use write transactions to update transforms
            # But for MVP, let's save and reload the stage
            self.stage.GetRootLayer().Save()

            # Recreate the stage (not optimal, but works for MVP)
            await self.write_client.delete_runtime_stage(self.runtime_stage)
            usd_url = f"file://{os.path.abspath(self.usd_path)}"
            self.runtime_stage = await self.write_client.create_runtime_stage(
                f"newton-sim-{id(self)}-frame{self._frame_index}", source=write.StorageSource(usd_url=usd_url)
            )

            # Need to recreate the view since we deleted the stage
            render_product_path = getattr(self, "render_product_path", "/Render/RGBDCam")
            print(f"Recreating view for frame {self._frame_index} at {render_product_path}")
            self.view = await self.render_client.create_view(
                prim_path=render.PrimPath(render_product_path),
                view_id=f"newton-render-view-frame{self._frame_index}",
                parent=StageName(self.runtime_stage.name),
            )
            print(f"View recreated for frame {self._frame_index}")

        # Create stream for receiving render output
        stream_view = await self.render_client.stream_view(
            self.view,
            outputs=["LdrColor"],  # RGB output
        )

        # Submit render request (use change number 0 for MVP since we recreate stage each frame)
        await self.render_client.render_view(self.view, change_number=ChangeNumber(0))

        # Collect rendered image data
        async with contextlib.aclosing(stream_view) as stream:
            stream_iter = aiter(stream)

            chunks = []
            chunk_count = 0
            metadata = None

            while True:
                try:
                    response = await anext(stream_iter)

                    if hasattr(response, "chunk"):
                        # Store metadata from first response
                        if metadata is None and hasattr(response, "metadata"):
                            metadata = response.metadata

                        # First chunk tells us total count
                        if response.chunk.chunk_count:
                            if chunk_count == 0:
                                chunk_count = response.chunk.chunk_count

                        # Collect data chunks
                        if response.chunk.data:
                            chunks.append(response.chunk.data)

                            # Check if we have all chunks
                            if len(chunks) == chunk_count and chunk_count > 0:
                                # Combine chunks and save image
                                raw_data = b"".join(chunks)

                                # Use metadata for image dimensions
                                if metadata and hasattr(metadata, "width"):
                                    image = Image.frombytes(
                                        mode="RGBA",
                                        size=(metadata.width, metadata.height),
                                        data=raw_data,
                                    )
                                else:
                                    # Fallback to configured resolution
                                    image = Image.frombytes(
                                        mode="RGBA",
                                        size=self.resolution,
                                        data=raw_data,
                                    )

                                # Save frame
                                output_path = self.output_dir / f"frame_{self._frame_index:06d}.png"
                                image.save(output_path, "PNG")
                                print(f"Saved frame {self._frame_index} to {output_path}")
                                break

                    # Check for errors
                    if hasattr(response, "error"):
                        print(f"Render error: {response.error}")
                        break

                except StopAsyncIteration:
                    break
                except Exception as e:
                    print(f"Error during rendering: {e}")
                    break

    def end_frame(self):
        """
        End the current frame and trigger SensorRTX rendering.

        This is called after all geometry has been logged for the frame.
        """
        # Run the async render in the background thread
        try:
            self._run_async(self._render_frame_async())
        except Exception as e:
            print(f"Failed to render frame {self._frame_index}: {e}")

        # Call parent end_frame to handle USD saving (but it doesn't increment frame_count)
        super().end_frame()

    async def _cleanup_async(self):
        """Clean up SensorRTX resources."""
        if self._connected and self.runtime_stage is not None:
            try:
                await self.write_client.delete_runtime_stage(self.runtime_stage)
                print("Cleaned up runtime stage")
            except Exception as e:
                print(f"Error during cleanup: {e}")

        if self.channel is not None:
            try:
                await self.channel.__aexit__(None, None, None)
            except Exception:
                pass

    def close(self):
        """
        Close the viewer and clean up resources.
        """
        # Clean up SensorRTX connection
        if self._connected:
            try:
                self._run_async(self._cleanup_async())
            except Exception as e:
                print(f"Error during cleanup: {e}")

        # Stop the async runner thread
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=2.0)

        # Call parent close (saves final USD)
        super().close()

        print(f"ViewerSRTX: Rendered {self._frame_count} frames to {self.output_dir}")


# Optional: Add convenience function for testing
def test_viewer_srtx():
    """Test the SensorRTX viewer with a simple scene."""
    # Create a simple pendulum model
    # Note: This is just a placeholder test function
    # In real usage, you would create a proper Newton model
    # and simulate it with the ViewerSRTX
    print("Test function placeholder - create a Newton model and use ViewerSRTX")
    return

    # Example usage (commented out as it needs proper Newton model):
    # import newton as nt
    # model = nt.Model()
    # # ... add bodies and shapes ...
    # viewer = ViewerSRTX(...)
    # state = model.state()
    # for i in range(num_frames):
    #     viewer.begin_frame(i * dt)
    #     viewer.log_state(state)
    #     viewer.end_frame()
    # viewer.close()


if __name__ == "__main__":
    # Run test if executed directly
    test_viewer_srtx()
