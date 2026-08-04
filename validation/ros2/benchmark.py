#!/usr/bin/env python3
"""Separate-process ROS 2 PointCloud2, Image, and TF throughput benchmark."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

TOPICS = ("pointcloud2", "image", "tf")


def _tag(topic: str, sequence: int, sent_ns: int) -> str:
    return f"radcounter|{topic}|{sequence}|{sent_ns}"


def _parse_tag(value: str) -> tuple[str, int, int]:
    prefix, topic, sequence, sent_ns = value.split("|")
    if prefix != "radcounter":
        raise ValueError("unexpected benchmark frame")
    return topic, int(sequence), int(sent_ns)


def _qos(reliability: str) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=50,
        reliability=(
            ReliabilityPolicy.RELIABLE
            if reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        ),
    )


class BenchmarkPublisher(Node):
    def __init__(
        self,
        points: int,
        image_width: int,
        image_height: int,
        reliability: str,
    ) -> None:
        super().__init__("radcounter_bandwidth_publisher")
        qos = _qos(reliability)
        self.point_publisher = self.create_publisher(
            PointCloud2, "/radcounter/points", qos
        )
        self.image_publisher = self.create_publisher(
            Image, "/radcounter/image_raw", qos
        )
        self.tf_publisher = self.create_publisher(TFMessage, "/tf", qos)
        self.points = points
        self.image_width = image_width
        self.image_height = image_height
        self.point_data = bytes(points * 16)
        self.image_data = bytes(image_width * image_height * 3)
        self.sent = defaultdict(int)
        self.sent_bytes = defaultdict(int)

    def publish_pointcloud(self, sequence: int, sent_ns: int) -> None:
        message = PointCloud2()
        message.header.frame_id = _tag("pointcloud2", sequence, sent_ns)
        message.height = 1
        message.width = self.points
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = len(self.point_data)
        message.data = self.point_data
        message.is_dense = True
        self.point_publisher.publish(message)
        self.sent["pointcloud2"] += 1
        self.sent_bytes["pointcloud2"] += len(self.point_data)

    def publish_image(self, sequence: int, sent_ns: int) -> None:
        message = Image()
        message.header.frame_id = _tag("image", sequence, sent_ns)
        message.height = self.image_height
        message.width = self.image_width
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = self.image_width * 3
        message.data = self.image_data
        self.image_publisher.publish(message)
        self.sent["image"] += 1
        self.sent_bytes["image"] += len(self.image_data)

    def publish_tf(self, sequence: int, sent_ns: int, transform_count: int) -> None:
        transforms = []
        for index in range(transform_count):
            transform = TransformStamped()
            transform.header.frame_id = _tag("tf", sequence, sent_ns)
            transform.child_frame_id = f"robot_{index}/sensor"
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self.tf_publisher.publish(TFMessage(transforms=transforms))
        self.sent["tf"] += 1
        self.sent_bytes["tf"] += transform_count * 96


class BenchmarkSubscriber(Node):
    def __init__(self, reliability: str) -> None:
        super().__init__("radcounter_bandwidth_subscriber")
        qos = _qos(reliability)
        self.received = defaultdict(int)
        self.received_bytes = defaultdict(int)
        self.sequences: dict[str, set[int]] = defaultdict(set)
        self.latencies_ms: dict[str, list[float]] = defaultdict(list)
        self.create_subscription(
            PointCloud2, "/radcounter/points", self._point_callback, qos
        )
        self.create_subscription(
            Image, "/radcounter/image_raw", self._image_callback, qos
        )
        self.create_subscription(TFMessage, "/tf", self._tf_callback, qos)

    def _record(self, tag: str, payload_bytes: int) -> None:
        topic, sequence, sent_ns = _parse_tag(tag)
        self.received[topic] += 1
        self.received_bytes[topic] += payload_bytes
        self.sequences[topic].add(sequence)
        self.latencies_ms[topic].append(max(0.0, (time.time_ns() - sent_ns) / 1.0e6))

    def _point_callback(self, message: PointCloud2) -> None:
        self._record(message.header.frame_id, len(message.data))

    def _image_callback(self, message: Image) -> None:
        self._record(message.header.frame_id, len(message.data))

    def _tf_callback(self, message: TFMessage) -> None:
        if message.transforms:
            self._record(message.transforms[0].header.frame_id, len(message.transforms) * 96)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def run_publisher(args: argparse.Namespace) -> int:
    rclpy.init()
    node = BenchmarkPublisher(
        args.points,
        args.image_width,
        args.image_height,
        args.reliability,
    )
    discovery_deadline = time.monotonic() + 10.0
    while time.monotonic() < discovery_deadline:
        if all(
            publisher.get_subscription_count() >= 1
            for publisher in (
                node.point_publisher,
                node.image_publisher,
                node.tf_publisher,
            )
        ):
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    else:
        raise RuntimeError("ROS 2 subscribers were not discovered within 10 seconds")
    schedules = {
        "pointcloud2": [0, 1.0 / args.point_rate_hz],
        "image": [0, 1.0 / args.image_rate_hz],
        "tf": [0, 1.0 / args.tf_rate_hz],
    }
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.duration_s:
            elapsed = time.monotonic() - start
            sent_ns = time.time_ns()
            for topic, schedule in schedules.items():
                if elapsed < schedule[0]:
                    continue
                sequence = node.sent[topic]
                if topic == "pointcloud2":
                    node.publish_pointcloud(sequence, sent_ns)
                elif topic == "image":
                    node.publish_image(sequence, sent_ns)
                else:
                    node.publish_tf(sequence, sent_ns, args.transforms)
                schedule[0] += schedule[1]
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.0005)
    finally:
        payload = {
            "duration_s": time.monotonic() - start,
            "sent": dict(node.sent),
            "sent_bytes": dict(node.sent_bytes),
        }
        Path(args.publisher_output).write_text(json.dumps(payload), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()
    return 0


def run_subscriber(args: argparse.Namespace) -> int:
    rclpy.init()
    node = BenchmarkSubscriber(args.reliability)
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.duration_s + 5.0:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        payload = {
            "duration_s": time.monotonic() - start,
            "received": dict(node.received),
            "received_bytes": dict(node.received_bytes),
            "unique_sequences": {
                topic: len(sequences) for topic, sequences in node.sequences.items()
            },
            "latency_p50_ms": {
                topic: _percentile(values, 0.50)
                for topic, values in node.latencies_ms.items()
            },
            "latency_p99_ms": {
                topic: _percentile(values, 0.99)
                for topic, values in node.latencies_ms.items()
            },
        }
        Path(args.subscriber_output).write_text(json.dumps(payload), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()
    return 0


def run_coordinator(args: argparse.Namespace) -> int:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    publisher_output = root / "publisher.json"
    subscriber_output = root / "subscriber.json"
    common = [
        "--duration-s",
        str(args.duration_s),
        "--points",
        str(args.points),
        "--image-width",
        str(args.image_width),
        "--image-height",
        str(args.image_height),
        "--point-rate-hz",
        str(args.point_rate_hz),
        "--image-rate-hz",
        str(args.image_rate_hz),
        "--tf-rate-hz",
        str(args.tf_rate_hz),
        "--transforms",
        str(args.transforms),
        "--publisher-output",
        str(publisher_output),
        "--subscriber-output",
        str(subscriber_output),
    ]
    subscriber = subprocess.Popen([sys.executable, __file__, "subscriber", *common])
    time.sleep(2.0)
    publisher = subprocess.run([sys.executable, __file__, "publisher", *common], check=False)
    subscriber_code = subscriber.wait(timeout=args.duration_s + 15.0)
    if publisher.returncode != 0 or subscriber_code != 0:
        return 2
    published = json.loads(publisher_output.read_text(encoding="utf-8"))
    received = json.loads(subscriber_output.read_text(encoding="utf-8"))
    topic_metrics = {}
    passed = True
    for topic in TOPICS:
        sent = int(published["sent"].get(topic, 0))
        delivered = int(received["received"].get(topic, 0))
        ratio = delivered / sent if sent else 0.0
        p99 = received["latency_p99_ms"].get(topic)
        topic_pass = (
            ratio >= args.minimum_delivery_ratio
            and p99 is not None
            and p99 <= args.maximum_p99_latency_ms
        )
        passed = passed and topic_pass
        topic_metrics[topic] = {
            "sent": sent,
            "received": delivered,
            "delivery_ratio": ratio,
            "received_megabytes_per_s": (
                float(received["received_bytes"].get(topic, 0))
                / max(float(published["duration_s"]), 1.0e-9)
                / 1.0e6
            ),
            "latency_p50_ms": received["latency_p50_ms"].get(topic),
            "latency_p99_ms": p99,
            "passed": topic_pass,
        }
    result = {
        "passed": passed,
        "evidence_kind": "ros2_dds_separate_process_loopback",
        "reliability": args.reliability,
        "duration_s": published["duration_s"],
        "topics": topic_metrics,
    }
    (root / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("coordinator", "publisher", "subscriber"))
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--points", type=int, default=100_000)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--point-rate-hz", type=float, default=10.0)
    parser.add_argument("--image-rate-hz", type=float, default=15.0)
    parser.add_argument("--tf-rate-hz", type=float, default=50.0)
    parser.add_argument("--transforms", type=int, default=24)
    parser.add_argument(
        "--reliability",
        choices=("reliable", "best_effort"),
        default="reliable",
    )
    parser.add_argument("--minimum-delivery-ratio", type=float, default=0.95)
    parser.add_argument("--maximum-p99-latency-ms", type=float, default=250.0)
    parser.add_argument("--output-dir", default="/output")
    parser.add_argument("--publisher-output", default="/output/publisher.json")
    parser.add_argument("--subscriber-output", default="/output/subscriber.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "publisher":
        raise SystemExit(run_publisher(arguments))
    if arguments.mode == "subscriber":
        raise SystemExit(run_subscriber(arguments))
    raise SystemExit(run_coordinator(arguments))
