#!/usr/bin/env python3
"""
Mock service responder for PackeeVisionDetectProductsInCart.

This node exposes a service that accepts robot/order/product identifiers and
returns a configurable boolean + message pair (plus lightweight detection data).
It is useful when testing nodes that consume the Packee vision service without
requiring the real perception stack.
"""

import argparse
import sys
import time
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from shopee_interfaces.msg import BBox, DetectedProduct, Pose6D
from shopee_interfaces.srv import PackeeVisionDetectProductsInCart


class VisionDetectMock(Node):
    """Provides a simple mock response to PackeeVisionDetectProductsInCart requests."""

    def __init__(self, *, service_name: str, delay: float, valid_ids: Optional[List[int]],
                 success_message: str, failure_message: str, strict: bool) -> None:
        super().__init__('vision_detect_mock')
        self._delay = delay
        self._valid_ids = set(valid_ids or [])
        self._success_message = success_message
        self._failure_message = failure_message
        self._strict = strict
        self._default_confidence = 1.0
        self._default_arm_side = 'left'
        # Update these coordinates as needed; the mock cycles through them per request.
        self._pose_sequence = [   
            {'x': 0.081, 'y': -0.186, 'z': 0.035, 'rx': 0.0, 'ry': 1.5707, 'rz': 0.0},  # 사과
            {'x': 0.081, 'y': -0.1181, 'z': 0.03, 'rx': 0.0, 'ry': 1.5707, 'rz': 0.0},  #이클립스
            {'x': 0.114, 'y': -0.067, 'z': 0.03, 'rx': 0.0, 'ry': 1.5707, 'rz': 0.0},  # 와사비
            {'x': 0.155, 'y': -0.15, 'z': 0.045, 'rx': 0.0, 'ry': 1.5707, 'rz': -1.5707},  #두유
        ]
        self._pose_index = 0

        self._service = self.create_service(
            PackeeVisionDetectProductsInCart,
            service_name,
            self._handle_request,
        )

        if self._valid_ids:
            self.get_logger().info(
                f"Vision mock ready on '{service_name}'. "
                f"Valid product_ids={sorted(self._valid_ids)} (strict={self._strict}).")
        else:
            self.get_logger().info(
                f"Vision mock ready on '{service_name}'. "
                f"Accepting any product_id (strict={self._strict}).")

    def _handle_request(self, request: PackeeVisionDetectProductsInCart.Request,
                        response: PackeeVisionDetectProductsInCart.Response
                        ) -> PackeeVisionDetectProductsInCart.Response:
        self.get_logger().info(
            f"Request received: robot_id={request.robot_id} "
            f"order_id={request.order_id} expected_product_id={request.expected_product_id}")

        if self._delay > 0:
            time.sleep(self._delay)

        success, message = self._evaluate_request(request.expected_product_id)

        response.success = success
        response.message = message
        if success:
            response.total_detected = 1
            response.products = [self._make_detected_product(request.expected_product_id)]
        else:
            response.total_detected = 0
            response.products = []

        self.get_logger().info(
            f"Responding success={response.success} total_detected={response.total_detected} "
            f"message='{response.message}'")
        return response

    def _evaluate_request(self, product_id: int) -> Tuple[bool, str]:
        if not self._valid_ids:
            return True, self._success_message.format(product_id=product_id)

        is_valid = product_id in self._valid_ids
        if is_valid:
            return True, self._success_message.format(product_id=product_id)

        if self._strict:
            return False, self._failure_message.format(product_id=product_id)

        return False, self._failure_message.format(product_id=product_id)

    def _make_detected_product(self, product_id: int) -> DetectedProduct:
        pose = self._next_pose()
        product = DetectedProduct()
        product.product_id = product_id
        product.confidence = self._default_confidence
        product.bbox = BBox()
        product.bbox_number = 0
        product.pose = pose
        product.arm_side = self._default_arm_side
        return product

    def _next_pose(self) -> Pose6D:
        pose = Pose6D()
        if not self._pose_sequence:
            return pose

        coordinates = self._pose_sequence[self._pose_index]
        pose.x = coordinates['x']
        pose.y = coordinates['y']
        pose.z = coordinates['z']
        pose.rx = coordinates['rx']
        pose.ry = coordinates['ry']
        pose.rz = coordinates['rz']

        # Advance until we reach the last configured pose; no wrap-around.
        if self._pose_index < len(self._pose_sequence) - 1:
            self._pose_index += 1

        return pose


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock PackeeVisionDetectProductsInCart service responder.")
    parser.add_argument(
        '--service-name', type=str, default='packee_vision_detect_products_in_cart',
        help='Service name to advertise.')
    parser.add_argument(
        '--delay', type=float, default=0.0, help='Artificial processing delay (seconds).')
    parser.add_argument(
        '--valid-ids', type=int, nargs='*', default=None,
        help='Optional list of product_ids that should return success.')
    parser.add_argument(
        '--success-message', type=str, default='Product {product_id} verified.',
        help='Message template for successful responses.')
    parser.add_argument(
        '--failure-message', type=str, default='Product {product_id} not recognised.',
        help='Message template for failed responses.')
    parser.add_argument(
        '--strict', action='store_true',
        help='When set, product_ids not in --valid-ids always fail. '
             'Otherwise they return success=False but include failure message.')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    rclpy.init(args=argv)
    node = VisionDetectMock(
        service_name=args.service_name,
        delay=args.delay,
        valid_ids=args.valid_ids,
        success_message=args.success_message,
        failure_message=args.failure_message,
        strict=args.strict,
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
