#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped

from astar_planner import a_star_search, COORDS, GRAPH

ROBOT_NAME = "bear"
CMD_TOPIC = f"/{ROBOT_NAME}/wheels_driver_node/wheels_cmd"
CAMERA_TOPIC = f"/{ROBOT_NAME}/camera_node/image/compressed"
VIZ_PUB_TOPIC = f"/{ROBOT_NAME}/astar_nav_viz/compressed"

MARKER_SIZE_METERS = 0.065
REACH_THRESHOLD = 0.30

K = np.array([
    [270.456,   0.0,     314.181],
    [0.0,       269.295, 218.886],
    [0.0,       0.0,     1.0]
], dtype=np.float32)

D = np.array([-0.191, 0.026, 0.005, 0.0006, 0.0], dtype=np.float32)

MAP_W = 700
MAP_H = 700
MAP_SCALE = 150.0
MAP_ORIGIN = (100, 600)

def world_to_map_px(x, y):
    px = int(MAP_ORIGIN[0] + x * MAP_SCALE)
    py = int(MAP_ORIGIN[1] - y * MAP_SCALE)
    return px, py

class AutonomousNavigator:
    def __init__(self):
        rospy.init_node("astar_navigator_node")

        self.start_node = 0
        self.goal_node = 15
        rospy.loginfo("A* Algoritması rotayı hesaplıyor...")
        
        self.path, self.total_cost = a_star_search(self.start_node, self.goal_node)
        
        if not self.path:
            rospy.logerr("HATA: Hedefe giden yol bulunamadı!")
            rospy.signal_shutdown("No path")
            return
            
        formatted_path = " -> ".join([f"N{n}" for n in self.path])
        rospy.loginfo(f"Takip Edilecek Rota: {formatted_path}")

        self.path_index = 0
        self.current_node = self.start_node
        self.state = "STARTING"

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.new_api = False
        
        self.last_frame = None
        self.visible_tags = {}

        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist2DStamped, queue_size=1)
        self.viz_pub = rospy.Publisher(VIZ_PUB_TOPIC, CompressedImage, queue_size=1)
        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, self.image_cb, queue_size=1, buff_size=2**24)

    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def stop_robot(self):
        self.drive(0.0, 0.0)

    def drive(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.cmd_pub.publish(msg)

    def detect_markers(self, gray):
        if self.new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def process_detections(self, frame, corners, ids):
        self.visible_tags = {}

        if ids is None or len(ids) == 0:
            return

        obj_pts = np.array([
            [-MARKER_SIZE_METERS / 2,  MARKER_SIZE_METERS / 2, 0],
            [ MARKER_SIZE_METERS / 2,  MARKER_SIZE_METERS / 2, 0],
            [ MARKER_SIZE_METERS / 2, -MARKER_SIZE_METERS / 2, 0],
            [-MARKER_SIZE_METERS / 2, -MARKER_SIZE_METERS / 2, 0]
        ], dtype=np.float32)

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            image_pts = corners[i].reshape((4, 2)).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, image_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)

            if success:
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, MARKER_SIZE_METERS * 0.5)
                node_name = f"N{int(marker_id)}"
                dist = float(tvec[2][0])
                
                self.visible_tags[node_name] = {
                    "x_err": float(tvec[0][0]),
                    "dist": dist
                }
                
                c = image_pts.mean(axis=0).astype(int)
                cv2.putText(frame, f"{node_name} d:{dist:.2f}", (c[0] - 40, c[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    def publish_visualization(self, frame):

        canvas = np.ones((MAP_H, MAP_W, 3), dtype=np.uint8) * 255
        
        for n1, nbrs in GRAPH.items():
            x1, y1 = COORDS[n1]
            p1 = world_to_map_px(x1, y1)

            for n2, cost in nbrs.items():
                x2, y2 = COORDS[n2]
                p2 = world_to_map_px(x2, y2)

                if n1 < n2: 
                    cv2.line(canvas, p1, p2, (180, 180, 180), 2)
                    mx = (p1[0] + p2[0]) // 2
                    my = (p1[1] + p2[1]) // 2
                    cv2.putText(canvas, str(cost), (mx - 10, my + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        for node_id, (x, y) in COORDS.items():
            px, py = world_to_map_px(x, y)
            
            if node_id in self.path:
                color = (0, 165, 255)
            else:
                color = (255, 0, 0)
                
            cv2.circle(canvas, (px, py), 14, color, -1)
            cv2.putText(canvas, f"N{node_id}", (px - 15, py - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        rx, ry = world_to_map_px(COORDS[self.current_node][0], COORDS[self.current_node][1])
        cv2.circle(canvas, (rx, ry), 10, (0, 255, 0), -1)
        cv2.putText(canvas, "ROBOT", (rx + 15, ry + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2, cv2.LINE_AA)

        cv2.putText(canvas, "Path: " + " -> ".join([f"N{n}" for n in self.path]), 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(canvas, f"Current node: N{self.current_node}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        
        if self.path_index < len(self.path) - 1:
            cv2.putText(canvas, f"Next target: N{self.path[self.path_index + 1]}", 
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        h, w = frame.shape[:2]
        new_w = int((MAP_H / h) * w)
        frame_resized = cv2.resize(frame, (new_w, MAP_H))
        combined = np.hstack([frame_resized, canvas])

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", combined)[1]).tobytes()
        self.viz_pub.publish(msg)

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.path_index >= len(self.path) - 1:
                self.stop_robot()
                rospy.loginfo("GOAL REACHED! (Hedefe Başarıyla Ulaşıldı)")
                break

            if self.last_frame is None:
                rate.sleep()
                continue

            frame = cv2.undistort(self.last_frame.copy(), K, D)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids = self.detect_markers(gray)
            self.process_detections(frame, corners, ids)

            next_node_id = self.path[self.path_index + 1]
            next_node_str = f"N{next_node_id}"

            if next_node_str in self.visible_tags:
                self.state = "TRACKING"
                tag_info = self.visible_tags[next_node_str]
                x_err = tag_info["x_err"]
                dist = tag_info["dist"]

                if dist < REACH_THRESHOLD:
                    self.stop_robot()
                    self.current_node = next_node_id
                    self.path_index += 1
                    rospy.loginfo(f"--> Ulaşıldı: {next_node_str}. Sıradaki hedefe geçiliyor...")
                    rospy.sleep(1.0)
                    continue

                Kp_omega = 2.5
                omega = -Kp_omega * x_err 
                omega = max(-2.0, min(2.0, omega))

                v = 0.25
                if abs(x_err) > 0.15:
                    v = 0.05

                self.drive(v, omega)

            else:
                self.state = "SEARCHING"
                self.drive(0.0, 0.4) 

            self.publish_visualization(frame)
            rospy.loginfo_throttle(0.5, f"Durum: {self.state} | Hedef: {next_node_str}")
            
            rate.sleep()

if __name__ == "__main__":
    try:
        nav = AutonomousNavigator()
        nav.run()
    except rospy.ROSInterruptException:
        pass