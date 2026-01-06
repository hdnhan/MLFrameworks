# https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
import typing as T
from pathlib import Path
import logging

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

ROOT_DIR = Path(__file__).parents[1].resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEBUG = False

# Standard GStreamer initialization
Gst.init(None)


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        logger.error("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        logger.warning("Warning: %s: %s", err, debug)
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        logger.error("Error: %s: %s", err, debug)
        loop.quit()
    return True


def input_probe_callback(pad, info):
    """Probe callback for nvinferserver input"""
    buffer = info.get_buffer()
    if buffer:
        logger.info(
            "NVINFERSERVER INPUT - PTS: %s, Duration: %s, Size: %d bytes",
            buffer.pts,
            buffer.duration,
            buffer.get_size(),
        )
    else:
        logger.warning("No buffer received in input probe")
    return Gst.PadProbeReturn.OK


def output_probe_callback(pad, info):
    """Probe callback for nvinferserver output"""
    buffer = info.get_buffer()
    if buffer:
        logger.info(
            "NVINFERSERVER OUTPUT - PTS: %s, Duration: %s, Size: %d bytes",
            buffer.pts,
            buffer.duration,
            buffer.get_size(),
        )
    else:
        logger.warning("No buffer received in output probe")
    return Gst.PadProbeReturn.OK


def generate_source_elements(sources: T.List[str]) -> str:
    source_elements = []
    for i, source in enumerate(sources):
        if source.endswith(".mp4"):
            source_element = f"""
                filesrc location={source} !
                qtdemux ! h264parse ! nvv4l2decoder ! queue ! mux.sink_{i}
            """
        else:
            raise ValueError(f"Unsupported source format: {source}")
        source_elements.append(source_element)
    return "\n".join(source_elements)


if __name__ == "__main__":
    sources = [f"{ROOT_DIR}/Assets/video.mp4"]
    source_elements_str = generate_source_elements(sources)

    sink = f"""
        nvvideoconvert !
        nvmultistreamtiler rows=1 columns=1 width=1920 height=1080 !
        nvdsosd !
        nvvideoconvert !
        nvv4l2h264enc bitrate=2000000 !
        h264parse !
        qtmux !
        filesink location={ROOT_DIR}/Results/deepstream_python.mp4 sync=0
    """

    pipeline_description = f"""
        {source_elements_str}
        nvstreammux name=mux batch-size={len(sources)} width=1920 height=1080 batched-push-timeout=40000 live-source=0 !
        nvinferserver name=primary-gie config-file-path=config_infer.txt !
        {sink}
    """
    logger.info("Pipeline Description:\n%s", pipeline_description)
    pipeline = Gst.parse_launch(pipeline_description)
    if not pipeline:
        logger.error("Failed to create pipeline")
        exit(1)

    if DEBUG:
        # Add probe to nvinferserver output to monitor frames
        nvinferserver = pipeline.get_by_name("primary-gie")
        if nvinferserver:
            # Add sink pad probe (input)
            sinkpad = nvinferserver.get_static_pad("sink")
            if sinkpad:
                sinkpad.add_probe(Gst.PadProbeType.BUFFER, input_probe_callback)
                logger.info("Added input probe to nvinferserver")
            else:
                logger.warning("Could not get sink pad from nvinferserver")

            # Add source pad probe (output)
            srcpad = nvinferserver.get_static_pad("src")
            if srcpad:
                srcpad.add_probe(Gst.PadProbeType.BUFFER, output_probe_callback)
                logger.info("Added output probe to nvinferserver")
            else:
                logger.warning("Could not get src pad from nvinferserver")
        else:
            logger.warning("Could not find nvinferserver element in pipeline")

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    logger.info("Setting pipeline to PLAYING state")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        logger.info("Running main loop...")
        loop.run()
        logger.info("Main loop finished")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        pass

    logger.info("Setting pipeline to NULL state")
    pipeline.set_state(Gst.State.NULL)
