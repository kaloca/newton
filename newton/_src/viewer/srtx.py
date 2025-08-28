#!/usr/bin/env python3
"""
Minimal SensorRTX client - this is ALL you need!
"""

import asyncio
import contextlib
import sys
from PIL import Image
from sensor_rtx.usd import render, write
from sensor_rtx import ChangeNumber, Channel, StageName


async def minimal_srtx_example():
    """Minimal example of connecting to SensorRTX and rendering."""

    # Configuration
    host = "localhost"
    port = 8081  # or 50051 depending on your server

    # USD file and camera path - the camera should already exist in the USD file
    # Common camera paths in example USD files:
    # - Astronaut: /Render/RGBDLongView, /Render/RGBDCloseUp
    # - Ragnarok: /Render/RGBDmain_cam
    usd_url = "https://omniverse-coreapi-assets-dev.s3.us-west-2.amazonaws.com/Release_Test_Samples/july2024_milestone/Ragnarok/Koenigsegg_Ragnarok.usd"
    camera_path = "/Render/RGBDmain_cam"  # This camera exists in the Ragnarok.usd file

    # Connect to SensorRTX
    channel = Channel(host, port, ssl=False)
    write_client = write.UsdWriteClient(channel)
    render_client = render.UsdRenderClient(channel)

    async with channel:
        # Wait for service to be healthy
        await write_client.wait_for_healthy(max_wait_time=10)
        print("Connected to SensorRTX")

        # Create runtime stage from USD file
        stage = await write_client.create_runtime_stage("my-stage", source=write.StorageSource(usd_url=usd_url))
        print(f"Created runtime stage from {usd_url}")

        # Create a view (camera) - using an existing camera path in the USD file
        view = await render_client.create_view(
            prim_path=render.PrimPath(camera_path),
            view_id="main-camera",
            parent=StageName(stage.name),
        )
        print(f"Created view for camera at {camera_path}")

        # IMPORTANT: Create stream BEFORE submitting render
        # The stream must be ready to receive data when render completes
        stream_view = await render_client.stream_view(
            view,
            outputs=["LdrColor"],  # RGB output
        )
        print("Created stream for output")

        # Now submit the render request
        await render_client.render_view(view, change_number=ChangeNumber(0))
        print("Submitted render request")

        # Read from stream - following the exact pattern from the actual client
        async with contextlib.aclosing(stream_view) as stream:
            stream = aiter(stream)  # Convert to async iterator

            chunks = []
            chunk_count = 0
            metadata = None  # Store metadata from the first response

            # Keep reading until we have all chunks for the output
            while True:
                try:
                    # Get next response from stream
                    response = await anext(stream)

                    # Debug: print response type
                    if hasattr(response, "__class__"):
                        print(f"Received response type: {response.__class__.__name__}")

                    # Check for render response with chunk data
                    if hasattr(response, "chunk"):
                        # Store metadata from first response
                        if metadata is None and hasattr(response, "metadata"):
                            metadata = response.metadata

                        # First chunk tells us the total count
                        if response.chunk.chunk_count:
                            if chunk_count == 0:
                                chunk_count = response.chunk.chunk_count
                                print(f"Expecting {chunk_count} total chunks")

                        # Collect the actual data
                        if response.chunk.data:
                            chunks.append(response.chunk.data)
                            chunk_id = getattr(response.chunk, "chunk_id", len(chunks))
                            print(f"Received chunk {chunk_id}/{chunk_count}, size: {len(response.chunk.data)} bytes")

                            # Check if we have all chunks
                            if len(chunks) == chunk_count and chunk_count > 0:
                                # Combine all chunks
                                raw_data = b"".join(chunks)

                                # Convert raw RGBA data to PNG using PIL
                                if metadata and hasattr(metadata, "width"):
                                    # Use metadata to properly reconstruct the image
                                    image = Image.frombytes(
                                        mode="RGBA",
                                        size=(metadata.width, metadata.height),
                                        data=raw_data,
                                    )
                                    image.save("output.png", "PNG")
                                    print(
                                        f"Saved rendered image to output.png "
                                        f"({metadata.width}x{metadata.height}, {len(raw_data)} bytes raw data)"
                                    )
                                else:
                                    # Fallback: try to guess dimensions from data size
                                    # 8294400 bytes = 1920*1080*4 (RGBA)
                                    print("Warning: No metadata found, trying to guess image dimensions")
                                    width, height = 1920, 1080  # Common resolution
                                    if len(raw_data) == width * height * 4:
                                        image = Image.frombytes(
                                            mode="RGBA",
                                            size=(width, height),
                                            data=raw_data,
                                        )
                                        image.save("output.png", "PNG")
                                        print(f"Saved rendered image to output.png (guessed {width}x{height})")
                                    else:
                                        # Can't determine dimensions, save raw data
                                        with open("output.raw", "wb") as f:
                                            f.write(raw_data)
                                        print(f"Could not determine image format, saved raw data to output.raw")
                                break

                    # Check for errors
                    if hasattr(response, "error"):
                        print(f"Render error: {response.error}")
                        break

                except StopAsyncIteration:
                    print("Stream ended unexpectedly")
                    if chunks:
                        print(f"Partial data received: {len(chunks)} chunks")
                    break
                except Exception as e:
                    print(f"Error reading stream: {e}")
                    import traceback

                    traceback.print_exc()
                    break

        # Cleanup
        await write_client.delete_runtime_stage(stage)
        print("Cleaned up runtime stage")


if __name__ == "__main__":
    # That's it! Just run it
    asyncio.run(minimal_srtx_example())
