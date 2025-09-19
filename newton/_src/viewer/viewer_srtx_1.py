from __future__ import annotations

import asyncio
import contextlib
import os
import threading
from pathlib import Path

import numpy as np
import warp as wp

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = Vt = None

try:
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux, UsdRender
    from sensor_rtx import ChangeNumber, Channel, ChannelOptions, StageName
    from sensor_rtx.usd import render, write
except ImportError as e:
    raise ImportError(
        "SensorRTX dependencies not installed. Install with:\npip install usd-core pillow sensor-rtx"
    ) from e

from .viewer import ViewerBase


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


class ViewerSRTXOld(ViewerBase):
    """
    SensorRTX viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(
        self,
        output_dir="srtx_renders",
        fps=60,
        up_axis="Z",
        num_frames=5,
        host="localhost",
        port=8081,
        camera_position=(5.0, 5.0, 5.0),
        camera_target=(0.0, 0.0, 0.0),
        resolution=(1920, 1080),
    ):
        # def __init__(self, output_path, fps=60, up_axis="Z", num_frames=None):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path (str): Path to the output USD file.
            fps (int, optional): Frames per second for time sampling. Default is 60.
            up_axis (str, optional): USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames (int, optional): Maximum number of frames to record. If None, recording is unlimited.

        Raises:
            ImportError: If the usd-core package is not installed.
        """
        self.usd_path = str(Path(output_dir) / "newton_stage.usd")

        output_path = self.usd_path

        if Usd is None:
            raise ImportError("usd-core package is required for ViewerUSD. Install with: pip install usd-core")

        super().__init__()

        self.output_path = output_path
        self.fps = fps
        self.up_axis = up_axis
        self.num_frames = num_frames

        # Create USD stage
        self.stage = Usd.Stage.CreateNew(output_path)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)

        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        # Track meshes and instancers
        self._meshes = {}  # mesh_name -> prototype_path
        self._instancers = {}  # instancer_name -> UsdGeomPointInstancer
        self._points = {}  # point_name -> UsdGeomPoints

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        self.set_model(None)

        # SRTX
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
        self._run_async(self._connect_to_srtx())
        self._run_async(self._create_runtime_stage())

    def begin_frame(self, time):
        """
        Begin a new frame at the given simulation time.

        Parameters:
            time (float): The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_index = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._frame_index > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._frame_index)

    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

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

    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

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

    async def _create_runtime_stage(self):
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
        self.stage.GetRootLayer().Save()
        self.stage = None

        if self.output_path:
            print(f"USD output saved in: {os.path.abspath(self.output_path)}")

        print(f"ViewerSRTX: Rendered {self._frame_count} frames to {self.output_dir}")

    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array = None,
        uvs: wp.array = None,
        hidden=False,
        backface_culling=True,
    ):
        """
        Create a USD mesh prototype from vertex and index data.

        Parameters:
            name (str): Mesh name or Sdf.Path string.
            points (wp.array): Vertex positions as a warp array of wp.vec3.
            indices (wp.array): Triangle indices as a warp array of wp.uint32.
            normals (wp.array, optional): Vertex normals as a warp array of wp.vec3.
            uvs (wp.array, optional): UV coordinates as a warp array of wp.vec2.
            hidden (bool, optional): If True, mesh will be hidden. Default is False.
            backface_culling (bool, optional): If True, enable backface culling. Default is True.

        Returns:
            str: The mesh prototype path.
        """

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        if name not in self._meshes:
            # Store prototypes in a dedicated scope that won't be rendered directly
            # This prevents duplicate meshes from appearing at origin
            prototype_path = f"/Prototypes{name}" if not name.startswith("/Prototypes") else name
            self._ensure_scopes_for_path(self.stage, prototype_path)

            # Hide the /Prototypes scope so prototypes aren't rendered directly
            prototypes_scope = self.stage.GetPrimAtPath("/Prototypes")
            if prototypes_scope:
                prototypes_scope.GetAttribute("visibility").Set("invisible") if prototypes_scope.HasAttribute(
                    "visibility"
                ) else UsdGeom.Imageable(prototypes_scope).CreateVisibilityAttr().Set("invisible")

            mesh_prim = UsdGeom.Mesh.Define(self.stage, prototype_path)

            # setup topology once (do not set every frame)
            face_vertex_counts = [3] * (len(indices_np) // 3)
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)

            # Store the prototype with both the original name and the mesh prim
            # Use original name as key for lookup, but store the actual prototype path
            self._meshes[name] = (prototype_path, mesh_prim)

        # Get the mesh prim from the stored tuple
        prototype_path, mesh_prim = self._meshes[name]
        mesh_prim.GetPointsAttr().Set(points_np, self._frame_index)

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            mesh_prim.GetNormalsAttr().Set(normals_np, self._frame_index)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs if provided (simplified for now)
        if uvs is not None:
            # TODO: Implement UV support for USD meshes
            pass

        # Don't hide the prototype mesh here - it needs to be visible for instances to work
        # The proper solution is to place prototypes in a dedicated location
        # mesh_prim.GetVisibilityAttr().Set("invisible", self._frame_index)

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        """
        Create or update a PointInstancer for mesh instances.

        Parameters:
            name (str): Instancer name or Sdf.Path string.
            mesh (str): Mesh prototype name (must be previously logged).
            xforms (wp.array): Instance transforms as a warp array of wp.transform.
            scales (wp.array): Instance scales as a warp array of wp.vec3.
            colors (wp.array): Instance colors as a warp array of wp.vec3.
            materials (wp.array): Instance materials as a warp array of wp.vec4.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        # Extract the prototype path from the stored tuple
        prototype_path, _ = self._meshes[mesh]

        num_instances = len(xforms)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, name)

            instancer = UsdGeom.PointInstancer.Define(self.stage, name)
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            # Set the prototype relationship to the actual prototype path
            instancer.GetPrototypesRel().AddTarget(prototype_path)

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()

            # Extract positions from warp transforms using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = xforms_np[:, :3].astype(np.float32)

            # Convert quaternion format: Warp (x, y, z, w) → USD (w, (x,y,z))
            # USD expects quaternions as Gf.Quath(real, imag_vec3)
            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(num_instances):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Handle scales with numpy operations
            if scales is None:
                scales = np.ones((num_instances, 3), dtype=np.float32)
            elif isinstance(scales, wp.array):
                scales = scales.numpy().astype(np.float32)

            # Set attributes at current time
            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if scales is not None:
                instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                # Promote colors to proper numpy array format
                colors_np = self._promote_colors_to_array(colors, num_instances)

                # Set color per-instance
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)

                # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                indices = Vt.IntArray(range(num_instances))
                displayColor.SetIndices(indices, self._frame_index)

    # Abstract methods that need basic implementations
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """Debug helper to add a line list as a set of capsules

        Args:
            starts: The vertices of the lines (wp.array)
            ends: The vertices of the lines (wp.array)
            colors: The colors of the lines (wp.array)
            width: The width of the lines (float)
            hidden: Whether the lines are hidden (bool)
        """

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, name)

            instancer = UsdGeom.PointInstancer.Define(self.stage, name)

            # define nested capsule prim
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(width)

            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if starts is not None and ends is not None:
            num_lines = int(len(starts))
            if num_lines > 0:
                # bring to host
                starts = starts.numpy()
                ends = ends.numpy()

                line_positions = []
                line_rotations = []
                line_scales = []

                for i in range(num_lines):
                    pos0 = starts[i]
                    pos1 = ends[i]

                    (pos, rot, scale) = _compute_segment_xform(
                        Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
                    )

                    line_positions.append(pos)
                    line_rotations.append(rot)
                    line_scales.append(scale)

                instancer.GetPositionsAttr().Set(line_positions, self._frame_index)
                instancer.GetOrientationsAttr().Set(line_rotations, self._frame_index)
                instancer.GetScalesAttr().Set(line_scales, self._frame_index)
                instancer.GetProtoIndicesAttr().Set([0] * num_lines, self._frame_index)
                instancer.CreateIdsAttr().Set(list(range(num_lines)))

                if colors is not None:
                    # Promote colors to proper numpy array format
                    colors_np = self._promote_colors_to_array(colors, num_lines)

                    # Set color per-instance
                    displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                    displayColor.Set(colors_np, self._frame_index)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    indices = Vt.IntArray(range(num_lines))
                    displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    def log_points(self, name, points, radii, colors, hidden=False):
        if np.isscalar(radii):
            radius_interp = "constant"
        else:
            radius_interp = "vertex"

        if colors is None:
            color_interp = "constant"
        elif len(colors) == 3 and all(np.isscalar(x) for x in colors):
            color_interp = "constant"
        else:
            color_interp = "vertex"

        instancer = UsdGeom.Points.Get(self.stage, name)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, name)
            instancer = UsdGeom.Points.Define(self.stage, name)

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)

        instancer.GetPointsAttr().Set(points.numpy(), self._frame_index)

        # convert radii to widths for USD
        if np.isscalar(radii):
            widths = (radii * 2.0,)
        elif isinstance(radii, wp.array):
            widths = radii.numpy() * 2.0
        else:
            widths = np.array(radii) * 2.0

        instancer.GetWidthsAttr().Set(widths, self._frame_index)

        if colors is not None:
            if isinstance(colors, wp.array):
                colors = colors.numpy()
            elif isinstance(colors, list | tuple) and len(colors) == 3:
                colors = (colors,)

            instancer.GetDisplayColorAttr().Set(colors, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
        return instancer.GetPath()

    def log_array(self, name, array):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.
        """
        pass

    def log_scalar(self, name, value):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.
        """
        pass

    def _promote_colors_to_array(self, colors, num_items):
        """
        Helper method to promote colors to a numpy array format.

        Parameters:
            colors: Input colors in various formats (wp.array, list/tuple, np.ndarray)
            num_items (int): Number of items that need colors

        Returns:
            np.ndarray: Colors as numpy array with shape (num_items, 3)
        """
        if colors is None:
            return None

        if isinstance(colors, wp.array):
            # Convert warp array to numpy
            return colors.numpy()
        elif isinstance(colors, list | tuple) and len(colors) == 3 and all(np.isscalar(x) for x in colors):
            # Single color (list/tuple of 3 floats) - promote to array with one value per item
            return np.tile(colors, (num_items, 1))
        elif isinstance(colors, np.ndarray):
            # Already numpy array - pass through
            return colors
        else:
            # Fallback for other formats
            return np.array(colors)

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        """
        Ensure that all parent prims in the hierarchy exist as 'Scope' prims.

        If a prim does not exist at the given path, this method creates all
        non-existent parent prims in its hierarchy as 'Scope' prims. This is
        useful for ensuring a valid hierarchy before defining a prim.

        Parameters:
            stage (Usd.Stage): The USD stage to operate on.
            prim_path_str (str): The Sdf.Path string for the target prim.
        """
        # Convert the string to an Sdf.Path object for robust manipulation
        prim_path = Sdf.Path(prim_path_str)

        # First, check if the target prim already exists.
        if stage.GetPrimAtPath(prim_path):
            return

        # We only want to create the parent hierarchy, not the final prim itself.
        parent_path = prim_path.GetParentPath()

        # GetPrefixes() provides a convenient list of all ancestor paths.
        # For "/A/B/C", it returns ["/", "/A", "/A/B"].
        for path in parent_path.GetPrefixes():
            # The absolute root path ('/') always exists, so we can skip it.
            if path == Sdf.Path.absoluteRootPath:
                continue

            # Check if a prim exists at the current ancestor path.
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")
