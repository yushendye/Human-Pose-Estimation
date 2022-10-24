import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

sys.path.append('common/python')

print(str(Path(__file__).resolve().parents[2] / 'common/python'))

from openvino.model_zoo.model_api.models import ImageModel, OutputTransform
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage


default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))


ARCHITECTURES = {
    'ae': 'HPE-assosiative-embedding',
    'higherhrnet': 'HPE-assosiative-embedding',
    'openpose': 'openpose'
}


def draw_poses(img, poses, point_score_threshold, output_transform, skeleton=default_skeleton, draw_ellipses=False):
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


def main():
    cap = open_images_capture('demo.mp4', False)
    next_frame_id = 1
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()

    plugin_config = get_user_config("CPU", '', '')
    model_adapter = OpenvinoAdapter(
        create_core(),
        'intel\human-pose-estimation-0005\FP32\human-pose-estimation-0005.xml', 
        device='CPU', 
        plugin_config=plugin_config,
        max_num_requests=0, 
        model_parameters = {'input_layouts': None}
    )

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    config = {
        'target_size': None,
        'aspect_ratio': frame.shape[1] / frame.shape[0],
        'confidence_threshold': 0.1,
        'padding_mode': None, # the 'higherhrnet' and 'ae' specific
        'delta': None, # the 'higherhrnet' and 'ae' specific
    }
    model = ImageModel.create_model(ARCHITECTURES['ae'], model_adapter, config)
    model.log_layers_info()

    hpe_pipeline = AsyncPipeline(model)
    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})

    output_transform = OutputTransform(frame.shape[:2], None)
    output_resolution = (frame.shape[1], frame.shape[0])
    presenter = monitors.Presenter('', 55,
                                   (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))

    while True:
        if hpe_pipeline.callback_exceptions:
            raise hpe_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_poses(frame, poses, 0.1, output_transform)
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)
            if video_writer.isOpened() and (next_frame_id_to_show <= 1000 - 1):
                video_writer.write(frame)
            next_frame_id_to_show += 1
            
            cv2.imshow('Pose estimation results', frame)
            key = cv2.waitKey(1)

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            presenter.handleKey(key)
            continue

        if hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            hpe_pipeline.await_any()

    hpe_pipeline.await_all()
    if hpe_pipeline.callback_exceptions:
        raise hpe_pipeline.callback_exceptions[0]
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        (poses, scores), frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        frame = draw_poses(frame, poses, 0.1, output_transform)
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)
        if video_writer.isOpened() and (next_frame_id_to_show <= 1000 - 1):
            video_writer.write(frame)
        
        cv2.imshow('Pose estimation results', frame)
        key = cv2.waitKey(1)

        ESC_KEY = 27
        # Quit.
        if key in {ord('q'), ord('Q'), ESC_KEY}:
            break
        presenter.handleKey(key)

    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          hpe_pipeline.preprocess_metrics.get_latency(),
                          hpe_pipeline.inference_metrics.get_latency(),
                          hpe_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)