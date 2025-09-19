from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path

import numpy as np
import warp as wp

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = Vt = None

try:
    from PIL import Image
    from sensor_rtx import ChangeNumber, Channel, ChannelOptions, StageName
    from sensor_rtx.usd import render, write
    from sensor_rtx.usd.value import ValueColumn
except ImportError as e:
    raise ImportError(
        "SensorRTX dependencies not installed. Install with:\npip install usd-core pillow sensor-rtx"
    ) from e

from .viewer import ViewerBase


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()
    if height < 1e-6:  # Avoid division by zero for zero-length segments
        return (mid, Gf.Quath(1.0, Gf.Vec3h(0, 0, 0)), Gf.Vec3f(1.0, 1.0, 0.0))

    direction = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto(Gf.Vec3d(0.0, 0.0, 1.0), Gf.Vec3d(direction))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


class ViewerSRTX(ViewerBase):
    """
    SensorRTX viewer backend for Newton physics simulations.

    This backend creates a USD stage once, then incrementally updates transforms
    and deforming geometry each frame for efficient, high-performance rendering.
    """

    def __init__(
        self,
        output_dir="srtx_renders",
        fps=60,
        up_axis="Z",
        num_frames=None,  # Set to None for unlimited frames
        host="localhost",
        port=8081,
        camera_position=(5.0, 5.0, 5.0),
        camera_target=(0.0, 0.0, 0.0),
        resolution=(1280, 720),  # Reduced default resolution for faster rendering
        image_format="jpeg",  # "jpeg", "png", or "webp"
        jpeg_quality=85,  # Quality for JPEG (1-100, higher is better but larger)
        render_every_n_frames=1,  # Render every Nth frame (1=all, 2=every other, etc.)
    ):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_dir: Directory to save rendered frames
            fps: Frames per second for the simulation
            up_axis: USD up axis ('Y' or 'Z')
            num_frames: Maximum number of frames to render (None for unlimited)
            host: SensorRTX server hostname
            port: SensorRTX server port
            camera_position: Camera position as (x, y, z)
            camera_target: Camera look-at target as (x, y, z)
            resolution: Output resolution as (width, height). Default 1280x720
            image_format: Output format - 'jpeg', 'png', or 'webp'. Default 'jpeg'
            jpeg_quality: Quality for JPEG/WebP (1-100). Default 85
            render_every_n_frames: Render every Nth frame for faster preview. Default 1 (all frames)
        """
        if Usd is None:
            raise ImportError("usd-core package is required. Install with: pip install usd-core")

        super().__init__()

        self.usd_path = str(Path(output_dir).resolve() / "newton_stage.usd")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.up_axis = up_axis
        self.num_frames = num_frames

        # USD Stage setup
        self.stage = Usd.Stage.CreateNew(self.usd_path)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z if up_axis == "Z" else UsdGeom.Tokens.y)

        # Track USD prims
        self._meshes = {}
        self._instancers = {}

        # Frame tracking
        self._frame_index = 0
        self._frame_count = 0
        self._initial_time = None

        # SensorRTX settings
        self.host = host
        self.port = port
        self.camera_position = camera_position
        self.camera_target = camera_target
        self.resolution = resolution
        self.image_format = image_format.lower()
        self.jpeg_quality = jpeg_quality
        self.render_every_n_frames = render_every_n_frames

        # SensorRTX state
        self.channel = None
        self.write_client = None
        self.render_client = None
        self.runtime_stage = None
        self.view = None
        self._srtx_initialized = False
        self._update_buffer = {}  # Buffers prim updates for the current frame

        # Create camera and lights in the USD stage
        self.camera_path = "/Camera"
        self._create_camera()
        self._create_lights()

        # Async runner thread for non-blocking communication with SRTX
        self._loop = None
        self._thread = None
        self._initialize_async_runner()

        print(f"ViewerSRTX: Will render to {self.output_dir}")
        print(f"ViewerSRTX: USD stage will be saved to {self.usd_path}")

    def begin_frame(self, time):
        super().begin_frame(time)

        if self._initial_time is None:
            self._initial_time = time
            self._sequential_frame = 0
        else:
            # Use sequential frame counter to avoid skips/repeats
            self._sequential_frame += 1

        # Store both the time-based and sequential frame indices
        self._time_based_frame_index = int((time - self._initial_time) * self.fps)
        self._frame_index = self._sequential_frame

        # Only show frame info when it changes or for debugging
        if self._frame_index % 10 == 0:  # Show every 10th frame
            print(f"Frame {self._frame_index} (sim_time={time:.3f}s)")

        if not self._srtx_initialized:
            self.stage.SetStartTimeCode(self._frame_index)

        self.stage.SetEndTimeCode(self._frame_index)
        self._frame_count += 1
        # Don't clear the buffer here - it will be cleared after processing in end_frame

    def end_frame(self):
        """
        Finalizes the frame. On the first frame, it initializes the SRTX stage.
        On subsequent frames, it sends incremental updates and triggers a render.
        """
        if not self.is_running():
            return

        # Track which frames we've already processed to avoid duplicates
        if not hasattr(self, "_processed_frames"):
            self._processed_frames = set()

        if self._frame_index in self._processed_frames and self._srtx_initialized:
            print(f"WARNING: Frame {self._frame_index} already processed, skipping")
            return

        self._processed_frames.add(self._frame_index)

        # Skip rendering based on render_every_n_frames setting
        should_render = self._frame_index % self.render_every_n_frames == 0
        if not should_render and self._srtx_initialized:
            # Still update the USD stage but skip rendering
            return

        try:
            if not self._srtx_initialized:
                # First frame: build and upload the whole stage
                self._run_async(self._initialize_and_render_first_frame())
                # Clear buffer after processing
                self._update_buffer.clear()
            else:
                # Subsequent frames: send updates and render
                if self._update_buffer:
                    self._run_async(self._update_and_render_frame())
                    # Clear buffer after processing
                    self._update_buffer.clear()
                else:
                    print(f"Frame {self._frame_index}: No updates to render.")

        except Exception as e:
            print(f"Failed to process frame {self._frame_index}: {e}")
            # Clean up runtime stage on error
            if self.runtime_stage:
                print("Cleaning up runtime stage due to error...")
                try:
                    self._run_async(self._cleanup_async())
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
            # Stop the simulation if rendering fails critically
            self._frame_count = self.num_frames if self.num_frames is not None else self._frame_count

        super().end_frame()

    # --- Async SRTX Methods ---

    async def _initialize_and_render_first_frame(self):
        """(Frame 0) Saves USD, creates stage, WRITES initial state, creates view, and renders."""
        print("First frame: Initializing SensorRTX stage...")

        try:
            # 1. Write the initial state (buffered during log_instances) into the USD stage object
            # This ensures the saved file contains the correct frame 0 transforms
            for prim_path, attributes in self._update_buffer.items():
                instancer = self._instancers[prim_path]
                positions = Vt.Vec3fArray.FromNumpy(attributes["positions"]["values"][0])
                # Convert quatf back to quath for USD
                orientations = attributes["orientations"]["values"][0]
                # orientations = Vt.QuathArray.FromNumpy(orientations_np)
                scales = (
                    Vt.Vec3fArray.FromNumpy(attributes.get("scales", {}).get("values", [None])[0])
                    if "scales" in attributes
                    else None
                )

                instancer.GetPositionsAttr().Set(positions)
                # Set half-precision quaternions as quath[]
                instancer.GetOrientationsAttr().Set(orientations)
                if scales is not None:
                    instancer.GetScalesAttr().Set(scales)

            # 2. Save the populated USD stage to disk
            self.stage.GetRootLayer().Save()
            print(f"Initial USD stage saved to {self.usd_path}")

            # 3. Connect to SRTX
            await self._connect_to_srtx()

            # 4. Create Runtime Stage
            usd_url = f"file://{self.usd_path}"
            self.runtime_stage = await self.write_client.create_runtime_stage(
                f"newton-sim-{id(self)}", source=write.StorageSource(usd_url=usd_url)
            )
            print(f"Created runtime stage from {usd_url}")

            # 5. COMMIT THE INITIAL STATE (CRITICAL FIX)
            print(f"Writing initial state for frame {self._frame_index}...")
            print(f"Number of prims to update: {len(self._update_buffer)}")
            write_transaction = self.write_client.write(
                change_number=ChangeNumber(self._frame_index),
                stage_name=self.runtime_stage.name,
            )
            for prim_path, attributes in self._update_buffer.items():
                print(f"  Updating prim: {prim_path}, attributes: {list(attributes.keys())}")
                key_column = ("usd-path", [prim_path])

                attribute_columns = []
                for attr_name, data in attributes.items():
                    # Convert numpy arrays to appropriate format for SensorRTX
                    value_data = data["values"][0]
                    if hasattr(value_data, "tolist"):
                        value_data = value_data.tolist()

                    # Create appropriate ValueColumn based on attribute type
                    # For PointInstancer attributes, we're updating a single array-valued attribute
                    # The column should have ONE entry containing the entire array
                    attr_type = data.get("type", "")
                    if attr_type in ["point3f", "float3"]:
                        # For array attributes of 3D vectors, use Float32x3ArrayColumn
                        # We wrap value_data in a list because we're sending ONE array value
                        tuples_data = [tuple(v) for v in value_data] if value_data else []
                        formatted_data = ValueColumn.of_float32x3_array([tuples_data])
                    elif attr_type == "quatf" or attr_type == "quath":
                        # SensorRTX expects half4[] for quaternions; use xyzw order at runtime
                        # tuples_data = (
                        #     [(float(v[1]), float(v[2]), float(v[3]), float(v[0])) for v in value_data]
                        #     if value_data
                        #     else []
                        # )
                        out = []
                        for q in value_data:
                            w = float(q.GetReal())
                            v = q.GetImaginary()  # Gf.Vec3h (x, y, z)
                            out.append((float(v[0]), float(v[1]), float(v[2]), w))

                        # print(out)

                        # [data["values"][0]]
                        # print(value_data)
                        formatted_data = ValueColumn.of_float16x4_array([out])
                    else:
                        # Let ValueColumn.from_value handle other types
                        formatted_data = (
                            ValueColumn.from_value([value_data]) if value_data else ValueColumn.of_string([])
                        )

                    attribute_columns.append((attr_name, formatted_data))

                if attribute_columns:
                    print(f"    Sending {len(attribute_columns)} attributes for {prim_path}")
                    write_transaction.update(key_column, *attribute_columns)
                else:
                    print(f"    WARNING: No attributes to send for {prim_path}")

            print(f"Starting commit for frame {self._frame_index}...")
            try:
                await asyncio.wait_for(write_transaction.commit(), timeout=30.0)
                print(f"Committed initial state for frame {self._frame_index}.")
            except asyncio.TimeoutError:
                print(f"ERROR: Write transaction commit timed out after 30 seconds")
                print(f"This might indicate:")
                print(f"  - SensorRTX server is not responding")
                print(f"  - Data format issues")
                print(f"  - Network connectivity problems")
                print(f"Try checking if SensorRTX server is running properly")
                raise

            # 6. Create the Render View
            render_product_path = getattr(self, "render_product_path", "/Render/RGBDCam")
            self.view = await self.render_client.create_view(
                prim_path=render.PrimPath(render_product_path),
                view_id="newton-render-view",
                parent=StageName(self.runtime_stage.name),
            )
            print(f"Created view for RenderProduct at {render_product_path}")

            # 7. Create stream BEFORE triggering render (important for SensorRTX)
            print(f"Creating stream view for frame {self._frame_index}...")
            stream_view = await self.render_client.stream_view(self.view, outputs=["LdrColor"])
            print(f"Stream view created for frame {self._frame_index}")

            # 8. Now trigger the render
            print(f"Triggering render for frame {self._frame_index}...")
            await asyncio.wait_for(
                self.render_client.render_view(self.view, change_number=ChangeNumber(self._frame_index)), timeout=10.0
            )
            print(f"Render triggered successfully for frame {self._frame_index}")

            # 9. Process the output from the stream
            print("Processing render stream...")
            await asyncio.wait_for(self._process_render_stream_with_existing(stream_view), timeout=30.0)
            print("Render stream processed successfully")

            # 10. Mark as initialized
            self._srtx_initialized = True
            print("SensorRTX initialization complete.")

        except asyncio.TimeoutError:
            print(f"ERROR: Operation timed out during initialization")
            print(f"Check the last operation in the logs above to see where it hung")
            # Clean up runtime stage before re-raising
            if self.runtime_stage:
                print("Cleaning up runtime stage after timeout...")
                try:
                    await self.write_client.delete_runtime_stage(self.runtime_stage)
                    print("Cleaned up runtime stage.")
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
                finally:
                    self.runtime_stage = None
            raise
        except Exception as e:
            print(f"ERROR: Failed during initialization: {e}")
            import traceback

            traceback.print_exc()
            # Clean up runtime stage before re-raising
            if self.runtime_stage:
                print("Cleaning up runtime stage after error...")
                try:
                    await self.write_client.delete_runtime_stage(self.runtime_stage)
                    print("Cleaned up runtime stage.")
                except Exception as cleanup_e:
                    print(f"Error during cleanup: {cleanup_e}")
                finally:
                    self.runtime_stage = None
            raise

    async def _update_and_render_frame(self):
        """(Frame 1+) Sends buffered updates to SRTX and triggers a render."""
        if not self._update_buffer:
            print(f"No updates for frame {self._frame_index}, skipping")
            return

        # 1. Start a write transaction for the current frame
        write_transaction = self.write_client.write(
            change_number=ChangeNumber(self._frame_index),
            stage_name=self.runtime_stage.name,
        )
        for prim_path, attributes in self._update_buffer.items():
            key_column = ("usd-path", [prim_path])

            attribute_columns = []
            for attr_name, data in attributes.items():
                # Convert numpy arrays to appropriate format for SensorRTX
                value_data = data["values"][0]
                if hasattr(value_data, "tolist"):
                    value_data = value_data.tolist()

                # Skip empty data
                data_len = len(value_data) if isinstance(value_data, (list, np.ndarray)) else 0
                if data_len == 0:
                    continue

                # Create appropriate ValueColumn based on attribute type
                # For PointInstancer attributes, we're updating a single array-valued attribute
                # The column should have ONE entry containing the entire array
                attr_type = data.get("type", "")
                if attr_type in ["point3f", "float3"]:
                    # For array attributes of 3D vectors, use Float32x3ArrayColumn
                    # We wrap value_data in a list because we're sending ONE array value
                    tuples_data = [tuple(v) for v in value_data] if value_data else []
                    formatted_data = ValueColumn.of_float32x3_array([tuples_data])
                elif attr_type == "quatf" or attr_type == "quath":
                    # Send as xyzw float16x4.
                    # tuples_data = (
                    #     [(float(v[1]), float(v[2]), float(v[3]), float(v[0])) for v in value_data] if value_data else []
                    # )
                    # if attr_name == "orientations" and prim_path.endswith("shape_4") and self._frame_index % 20 == 0:
                    #     if tuples_data:
                    #         print(f"  First orientation (xyzw): {tuples_data[0]}")
                    # formatted_data = ValueColumn.of_float16x4_array([tuples_data])
                    out = []
                    for q in value_data:
                        w = float(q.GetReal())
                        v = q.GetImaginary()  # Gf.Vec3h (x, y, z)
                        out.append((float(v[0]), float(v[1]), float(v[2]), w))

                    # print(out)

                    # [data["values"][0]]
                    # print(value_data)
                    formatted_data = ValueColumn.of_float16x4_array([out])
                else:
                    # Let ValueColumn.from_value handle other types
                    formatted_data = ValueColumn.from_value([value_data]) if value_data else ValueColumn.of_string([])

                attribute_columns.append((attr_name, formatted_data))

            if attribute_columns:
                write_transaction.update(key_column, *attribute_columns)

        try:
            # Add timeout to detect hanging
            await asyncio.wait_for(write_transaction.commit(), timeout=30.0)
            print(f"Committed updates for frame {self._frame_index}.")
        except asyncio.TimeoutError:
            print(f"ERROR: Write transaction commit timed out after 30 seconds for frame {self._frame_index}")
            print(f"This might indicate a problem with the SensorRTX server or data format")
            raise
        except Exception as e:
            print(f"ERROR: Write transaction commit failed: {e}")
            raise

        # 4. Create stream BEFORE triggering render (important for SensorRTX)
        stream_view = await self.render_client.stream_view(self.view, outputs=["LdrColor"])

        # 5. Now trigger the render
        await self.render_client.render_view(self.view, change_number=ChangeNumber(self._frame_index))

        # 6. Process the output from the stream
        await asyncio.wait_for(self._process_render_stream_with_existing(stream_view), timeout=30.0)

    async def _process_render_stream_with_existing(self, stream_view):
        """Processes an existing render stream, handling multi-chunk responses."""
        async with contextlib.aclosing(stream_view) as stream:
            try:
                print(f"Waiting for render response for frame {self._frame_index}...")
                stream_iter = aiter(stream)

                chunks = []
                chunk_count = 0
                metadata = None

                # Keep reading until we have all chunks
                while True:
                    response = await asyncio.wait_for(anext(stream_iter), timeout=20.0)

                    if isinstance(response, render.RenderError):
                        print(f"Render error for frame {self._frame_index}: {response.message}")
                        return

                    if hasattr(response, "chunk"):
                        # Store metadata from first response
                        if metadata is None and hasattr(response, "metadata"):
                            metadata = response.metadata
                            # print(f"Render metadata: {metadata.width}x{metadata.height}")

                        # First chunk tells us total count
                        if response.chunk.chunk_count:
                            if chunk_count == 0:
                                chunk_count = response.chunk.chunk_count
                                # print(f"Expecting {chunk_count} chunks for frame {self._frame_index}")

                        # Collect data chunks
                        if response.chunk.data:
                            chunks.append(response.chunk.data)
                            # chunk_id = getattr(response.chunk, "chunk_id", len(chunks))
                            # print(f"Received chunk {chunk_id}/{chunk_count}, size: {len(response.chunk.data)} bytes")

                            # Check if we have all chunks
                            if len(chunks) == chunk_count and chunk_count > 0:
                                # Combine all chunks
                                raw_data = b"".join(chunks)
                                # print(f"Total image data: {len(raw_data)} bytes for {metadata.width}x{metadata.height}")

                                # Create image from combined data
                                image = Image.frombytes("RGBA", (metadata.width, metadata.height), raw_data)

                                # Convert and save based on format settings
                                if self.image_format == "jpeg":
                                    # Convert RGBA to RGB for JPEG (no transparency)
                                    image = image.convert("RGB")
                                    output_path = self.output_dir / f"frame_{self._frame_index:06d}.jpg"
                                    image.save(output_path, "JPEG", quality=self.jpeg_quality, optimize=True)
                                elif self.image_format == "webp":
                                    # WebP supports transparency and good compression
                                    output_path = self.output_dir / f"frame_{self._frame_index:06d}.webp"
                                    image.save(output_path, "WebP", quality=self.jpeg_quality, lossless=False)
                                else:  # PNG (default)
                                    output_path = self.output_dir / f"frame_{self._frame_index:06d}.png"
                                    image.save(output_path, "PNG", optimize=True)

                                if self._frame_index % 10 == 0:  # Only print every 10th frame
                                    print(f"Saved frame {self._frame_index} to {output_path}")
                                return  # Successfully processed all chunks

                    # Check for other error conditions
                    if hasattr(response, "error"):
                        print(f"Render error: {response.error}")
                        return

            except asyncio.TimeoutError:
                print(f"ERROR: Timeout waiting for render response for frame {self._frame_index}")
                print("This might indicate the render is taking too long or the stream is not working")
                raise
            except StopAsyncIteration:
                print(f"Stream ended for frame {self._frame_index}")
                # Treat as normal termination (no exception) if the stream ended cleanly
                return
            except Exception as e:
                print(f"Error processing render stream for frame {self._frame_index}: {e}")
                raise

    async def _process_render_stream(self):
        """Creates a new stream and processes it. This method is no longer used but kept for compatibility."""
        print(f"Creating stream view for frame {self._frame_index}...")
        stream_view = await self.render_client.stream_view(self.view, outputs=["LdrColor"])
        print(f"Stream view created for frame {self._frame_index}")
        await self._process_render_stream_with_existing(stream_view)

    # --- Overridden Logging Methods ---

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
        # On Frame 0, define the mesh topology and initial points.
        if not self._srtx_initialized:
            if name in self._meshes:
                mesh_prim = self._meshes[name]
            else:
                # Always put prototypes under /Prototypes to keep them separate
                prototype_path = f"/Prototypes{name}" if not name.startswith("/Prototypes") else name
                self._ensure_scopes_for_path(self.stage, prototype_path)

                # Hide the /Prototypes scope so prototypes aren't rendered directly
                prototypes_scope = self.stage.GetPrimAtPath("/Prototypes")
                if prototypes_scope and not prototypes_scope.HasAttribute("visibility"):
                    UsdGeom.Imageable(prototypes_scope).CreateVisibilityAttr().Set("invisible")
                elif prototypes_scope:
                    prototypes_scope.GetAttribute("visibility").Set("invisible")

                mesh_prim = UsdGeom.Mesh.Define(self.stage, prototype_path)
                self._meshes[name] = mesh_prim

                # Set topology only once
                indices_np = indices.numpy().astype(np.uint32)
                face_vertex_counts = [3] * (len(indices_np) // 3)
                mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
                mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)

            # Set initial points
            points_np = points.numpy().astype(np.float32)
            mesh_prim.GetPointsAttr().Set(points_np)
            return

        # On subsequent frames, if points are provided, buffer them for update.
        if points is not None:
            prim_path = self._meshes[name].GetPath().pathString
            points_np = points.numpy().astype(np.float32)
            self._update_buffer[prim_path] = {"points": {"values": [points_np], "type": "point3f"}}

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        # On Frame 0, define the instancer and its static properties

        if not self._srtx_initialized:
            print("LOG INSTANCES IS HAPPENING")

            if mesh not in self._meshes:
                raise RuntimeError(f"Mesh prototype '{mesh}' not found. Call log_mesh() first.")

            self._ensure_scopes_for_path(self.stage, name)
            instancer = UsdGeom.PointInstancer.Define(self.stage, name)
            self._instancers[name] = instancer

            num_instances = len(xforms)
            prototype_path = self._meshes[mesh].GetPath()

            instancer.GetPrototypesRel().AddTarget(prototype_path)
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            # Set initial colors (can't be updated efficiently per-frame yet)
            self._set_instancer_colors(instancer, colors, num_instances)

        # For ALL frames (including 0), buffer the dynamic transform data for writing
        updates = {}
        if xforms is not None:
            # Get clean numpy arrays directly from the new helper
            positions_np, orientations_np = self._convert_xforms_to_pos_and_quat(xforms)

            xforms_np = xforms.numpy()

            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(len(xforms)):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Store the raw numpy arrays in the buffer
            updates["positions"] = {"values": [positions_np], "type": "point3f"}
            updates["orientations"] = {"values": [orientations], "type": "quath"}

        if scales is not None:
            scales_np = scales.numpy().astype(np.float32)
            updates["scales"] = {"values": [scales_np], "type": "float3"}

        if updates:
            self._update_buffer[name] = updates

    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        num_lines = len(starts) if starts is not None else 0

        if not self._srtx_initialized:
            # --- FRAME 0: DEFINE INSTANCER AND CAPSULE PROTOTYPE ---
            self._ensure_scopes_for_path(self.stage, name)
            instancer = UsdGeom.PointInstancer.Define(self.stage, name)
            self._instancers[name] = instancer

            capsule_path = instancer.GetPath().AppendChild("capsule")
            capsule = UsdGeom.Capsule.Define(self.stage, capsule_path)
            capsule.GetRadiusAttr().Set(width)
            instancer.GetPrototypesRel().SetTargets([capsule.GetPath()])
            instancer.CreateIdsAttr().Set(list(range(num_lines)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_lines)

            if num_lines > 0:
                positions, orientations, scales = self._compute_line_transforms(starts, ends)
                self._set_instancer_transforms(instancer, positions, orientations, scales)
                self._set_instancer_colors(instancer, colors, num_lines)
        else:
            # --- FRAME 1+: BUFFER TRANSFORM UPDATES ---
            if num_lines > 0:
                positions, orientations, scales = self._compute_line_transforms(starts, ends)
                # Convert Vt arrays to numpy arrays for storage
                self._update_buffer[name] = {
                    "positions": {"values": [np.array(positions)], "type": "point3f"},
                    "orientations": {"values": [np.array(orientations)], "type": "quath"},
                    "scales": {"values": [np.array(scales)], "type": "float3"},
                }
            else:  # Ensure lines disappear if there are none
                self._update_buffer[name] = {"positions": {"values": [np.array([])], "type": "point3f"}}

    # --- Helper Methods ---

    def _update_instancer_attributes(self, instancer, xforms, scales, colors):
        """Helper to set all attributes for an instancer, used on Frame 0."""
        num_instances = len(xforms)
        positions, orientations = self._convert_xforms_to_pos_and_quat(xforms)

        self._set_instancer_transforms(instancer, positions, orientations, scales)
        self._set_instancer_colors(instancer, colors, num_instances)

    def _set_instancer_transforms(self, instancer, positions, orientations, scales):
        """Sets transform attributes on a UsdGeom.PointInstancer prim."""
        instancer.GetPositionsAttr().Set(positions)
        instancer.GetOrientationsAttr().Set(orientations)
        if scales is not None:
            scales_np = scales.numpy().astype(np.float32)
            instancer.GetScalesAttr().Set(scales_np)

    def _set_instancer_colors(self, instancer, colors, num_instances):
        """Sets displayColor primvar on a UsdGeom.PointInstancer prim."""
        if colors is not None:
            colors_np = self._promote_colors_to_array(colors, num_instances)
            color_primvar = UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
            )
            color_primvar.Set(colors_np)

    def _compute_line_transforms(self, starts, ends):
        """Computes positions, orientations, and scales for line segment capsules."""
        num_lines = len(starts)
        starts_np = starts.numpy()
        ends_np = ends.numpy()

        positions, orientations, scales = [], [], []
        for i in range(num_lines):
            pos0 = Gf.Vec3f(*starts_np[i])
            pos1 = Gf.Vec3f(*ends_np[i])
            pos, rot, scale = _compute_segment_xform(pos0, pos1)
            positions.append(pos)
            orientations.append(rot)
            scales.append(scale)
        return Vt.Vec3fArray(positions), Vt.QuathArray(orientations), Vt.Vec3fArray(scales)

    @staticmethod
    def _convert_xforms_to_pos_and_quat(xforms: wp.array):
        """
        Converts a warp transform array into separate position and quaternion numpy arrays.
        This version returns numpy arrays directly to avoid complex Vt <-> numpy conversions.
        """
        xforms_np = xforms.numpy()
        positions_np = xforms_np[:, :3].astype(np.float32)

        # Warp transform stores quaternion as (x, y, z, w) in indices [3:7]
        # USD expects quaternion as (w, x, y, z) for both file storage and APIs
        quat_xyzw = xforms_np[:, 3:7].astype(np.float32)
        quat_wxyz_np = np.empty_like(quat_xyzw)
        quat_wxyz_np[:, 0] = quat_xyzw[:, 3]  # w
        quat_wxyz_np[:, 1] = quat_xyzw[:, 0]  # x
        quat_wxyz_np[:, 2] = quat_xyzw[:, 1]  # y
        quat_wxyz_np[:, 3] = quat_xyzw[:, 2]  # z

        return positions_np, quat_wxyz_np

    def is_running(self):
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    def close(self):
        """Cleans up SensorRTX resources and stops the async thread."""
        if self._srtx_initialized:
            try:
                self._run_async(self._cleanup_async())
            except Exception as e:
                print(f"Error during SRTX cleanup: {e}")

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=2.0)

        # Final save of the stage might be useful for inspection
        if self.stage:
            self.stage.GetRootLayer().Save()
            print(f"Final USD output saved in: {self.usd_path}")

        print(f"ViewerSRTX: Rendered {self._frame_count} frames to {self.output_dir}")

    # --- Boilerplate Async and USD Setup ---

    def _initialize_async_runner(self):
        if self._loop is None and self._thread is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="SRTXAsyncRunner", daemon=True)
            self._thread.start()

    def _run_async(self, coro):
        if not self._thread or not self._thread.is_alive():
            raise RuntimeError("Async runner thread is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=60.0)
        except Exception as e:
            print(f"ERROR in _run_async: {e}")
            import traceback

            traceback.print_exc()
            raise

    async def _connect_to_srtx(self):
        if self.channel and hasattr(self.channel, "healthy") and self.channel.healthy:
            return
        try:
            options = ChannelOptions(host=self.host, port=self.port, ssl=False)
            self.channel = Channel(options)
            self.write_client = write.UsdWriteClient(self.channel)
            self.render_client = render.UsdRenderClient(self.channel)
            await self.write_client.wait_for_healthy(max_wait_time=15)
            print(f"Connected to SensorRTX at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to SensorRTX at {self.host}:{self.port}: {e}")
            raise

    async def _cleanup_async(self):
        if self.runtime_stage:
            try:
                await self.write_client.delete_runtime_stage(self.runtime_stage)
                print("Cleaned up runtime stage.")
                self.runtime_stage = None  # Clear the reference after cleanup
            except Exception as e:
                print(f"Error deleting runtime stage: {e}")
        if self.channel:
            try:
                # SensorRTX Channel doesn't have a close method, but might have __aexit__
                if hasattr(self.channel, "__aexit__"):
                    await self.channel.__aexit__(None, None, None)
                    print("Closed SensorRTX channel.")
            except Exception as e:
                print(f"Error closing channel: {e}")

    def _create_camera(self):
        camera = UsdGeom.Camera.Define(self.stage, self.camera_path)
        camera.GetFocusDistanceAttr().Set(10.0)
        camera.GetFocalLengthAttr().Set(24.0)
        aspect_ratio = self.resolution[0] / self.resolution[1]
        camera.GetHorizontalApertureAttr().Set(36.0)
        camera.GetVerticalApertureAttr().Set(36.0 / aspect_ratio)

        eye, center = Gf.Vec3d(*self.camera_position), Gf.Vec3d(*self.camera_target)
        world_up = Gf.Vec3d(0, 0, 1) if self.up_axis == "Z" else Gf.Vec3d(0, 1, 0)
        view_dir = (center - eye).GetNormalized()
        right = Gf.Cross(view_dir, world_up).GetNormalized()
        up = Gf.Cross(right, view_dir).GetNormalized()

        mat = Gf.Matrix4d(1.0)
        mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
        mat.SetRow(1, Gf.Vec4d(up[0], up[1], up[2], 0))
        mat.SetRow(2, Gf.Vec4d(-view_dir[0], -view_dir[1], -view_dir[2], 0))
        mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
        UsdGeom.Xformable(camera).AddTransformOp().Set(mat)

        render_scope = UsdGeom.Scope.Define(self.stage, "/Render")
        self.render_product_path = "/Render/RGBDCam"
        render_product = UsdRender.Product.Define(self.stage, self.render_product_path)
        render_product.GetCameraRel().SetTargets([camera.GetPath()])
        render_product.GetResolutionAttr().Set(Gf.Vec2i(self.resolution[0], self.resolution[1]))
        ldr_color_var = UsdRender.Var.Define(self.stage, "/Render/Vars/ldrColor")
        ldr_color_var.GetSourceNameAttr().Set("LdrColor")
        render_product.GetOrderedVarsRel().AddTarget(ldr_color_var.GetPath())
        print(f"Created Camera at {self.camera_path} and RenderProduct at {self.render_product_path}")

    def _create_lights(self):
        dome_light = UsdLux.DomeLight.Define(self.stage, "/Render/DomeLight")
        dome_light.GetIntensityAttr().Set(500.0)
        distant_light = UsdLux.DistantLight.Define(self.stage, "/Render/DistantLight")
        distant_light.GetIntensityAttr().Set(1500.0)
        distant_light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 0.95))
        UsdGeom.Xformable(distant_light).AddRotateXOp().Set(-60.0)
        print("Created DomeLight and DistantLight.")

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        parent_path = Sdf.Path(prim_path_str).GetParentPath()
        for path in parent_path.GetPrefixes():
            if path != Sdf.Path.absoluteRootPath and not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")

    def _promote_colors_to_array(self, colors, num_items):
        if colors is None:
            return np.tile([0.8, 0.8, 0.8], (num_items, 1))
        if isinstance(colors, wp.array):
            return colors.numpy()
        if isinstance(colors, (list, tuple)):
            return np.tile(colors, (num_items, 1))
        return np.array(colors)

    # --- Unimplemented abstract methods from ViewerBase ---
    def log_points(self, name, points, radii, colors, hidden=False):
        pass

    def log_array(self, name, array):
        pass

    def log_scalar(self, name, value):
        pass
