#!/usr/bin/env python3
import math
import numpy as np
import rospy
import cv2

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


# =========================
# CONFIG
# =========================
ROBOT_NAME = "bear"

CMD_TOPIC = f"/{ROBOT_NAME}/wheels_driver_node/wheels_cmd"
CAMERA_TOPIC = f"/{ROBOT_NAME}/camera_node/image/compressed"
VIZ_PUB_TOPIC = f"/{ROBOT_NAME}/astar_nav_viz/compressed"

MARKER_SIZE_METERS = 0.065
REACH_THRESHOLD = 0.30

# Kamera kalibrasyonu (senin önceki koddakiyle aynı)
K = np.array([
    [270.4563591302591,   0.0,               314.1813567017415],
    [0.0,                 269.2951665378049, 218.88618596346137],
    [0.0,                 0.0,               1.0]
], dtype=np.float32)

D = np.array([
    -0.19162991260105328,
     0.026384790215657535,
     0.005682129590129115,
     0.0006647376545041703,
     0.0
], dtype=np.float32)

# Kamera -> robot offset
CAMERA_TO_ROBOT_X = 0.0
CAMERA_TO_ROBOT_Y = 0.0
CAMERA_TO_ROBOT_YAW = 0.0

# Görselleştirme
MAP_W = 700
MAP_H = 700
MAP_SCALE = 150.0
MAP_ORIGIN = (100, 600)


# =========================
# HELPER FUNCTIONS
# =========================
def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotation_2d(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        [c, -s],
        [s,  c]
    ], dtype=np.float32)


def pose_to_T(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    T = np.eye(3, dtype=np.float32)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 2] = x
    T[1, 2] = y
    return T


def T_to_pose(T):
    x = float(T[0, 2])
    y = float(T[1, 2])
    yaw = math.atan2(T[1, 0], T[0, 0])
    return x, y, yaw


def world_to_map_px(x, y):
    px = int(MAP_ORIGIN[0] + x * MAP_SCALE)
    py = int(MAP_ORIGIN[1] - y * MAP_SCALE)
    return px, py


def camera_pose_from_marker(marker_world, rvec, tvec):
    R_cm, _ = cv2.Rodrigues(rvec)
    t_cm = tvec.reshape(3, 1)

    R_mc = R_cm.T
    t_mc = -R_mc @ t_cm

    cam_x_m = float(t_mc[0, 0])
    cam_z_m = float(t_mc[2, 0])

    marker_x = marker_world["x"]
    marker_y = marker_world["y"]
    marker_yaw = marker_world["yaw"]

    p_local = np.array([cam_z_m, cam_x_m], dtype=np.float32)
    R_wm_2d = rotation_2d(marker_yaw)
    p_world = R_wm_2d @ p_local + np.array([marker_x, marker_y], dtype=np.float32)

    forward_cam_in_marker = R_mc[:, 2]
    fx = float(forward_cam_in_marker[0])
    fz = float(forward_cam_in_marker[2])
    cam_yaw_local = math.atan2(fz, fx)
    cam_yaw_world = wrap_angle(marker_yaw + cam_yaw_local - math.pi / 2)

    return float(p_world[0]), float(p_world[1]), cam_yaw_world


def camera_to_robot_pose(cam_x, cam_y, cam_yaw):
    T_wc = pose_to_T(cam_x, cam_y, cam_yaw)
    T_cr = pose_to_T(CAMERA_TO_ROBOT_X, CAMERA_TO_ROBOT_Y, CAMERA_TO_ROBOT_YAW)
    T_wr = T_wc @ T_cr
    return T_to_pose(T_wr)


# =========================
# NODE
# =========================
class Assignment3Node:
    def __init__(self):
        rospy.init_node("assignment3_astar_navigation_node")

        # 4x4 düğüm koordinatları
        self.coords = {
            "N0": (0, 0), "N1": (1, 0), "N2": (2, 0), "N3": (3, 0),
            "N4": (0, 1), "N5": (1, 1), "N6": (2, 1), "N7": (3, 1),
            "N8": (0, 2), "N9": (1, 2), "N10": (2, 2), "N11": (3, 2),
            "N12": (0, 3), "N13": (1, 3), "N14": (2, 3), "N15": (3, 3)
        }

        # Ödevdeki edge-cost grafiği
        self.graph = {
            "N0":  {"N1": 1.5, "N4": 2.0},
            "N1":  {"N0": 1.5, "N2": 1.0, "N5": 2.0},
            "N2":  {"N1": 1.0, "N3": 1.0, "N6": 1.5},
            "N3":  {"N2": 1.0},

            "N4":  {"N0": 2.0, "N8": 1.5},
            "N5":  {"N1": 2.0, "N6": 1.0, "N9": 2.0},
            "N6":  {"N2": 1.5, "N5": 1.0, "N7": 0.5, "N10": 4.0},
            "N7":  {"N6": 0.5, "N11": 1.5},

            "N8":  {"N4": 1.5, "N9": 1.5, "N12": 2.0},
            "N9":  {"N5": 2.0, "N8": 1.5, "N10": 2.0},
            "N10": {"N6": 4.0, "N9": 2.0, "N11": 1.0, "N14": 1.5},
            "N11": {"N7": 1.5, "N10": 1.0},

            "N12": {"N8": 2.0, "N13": 1.5},
            "N13": {"N12": 1.5, "N14": 2.0},
            "N14": {"N13": 2.0, "N10": 1.5, "N15": 1.0},
            "N15": {"N14": 1.0}
        }

        # ARTag ID -> node
        self.tag_to_node = {i: f"N{i}" for i in range(16)}

        # Marker world map
        # Bu ödevde her ARTag ilgili node koordinatında.
        # yaw değerlerini laboratuvardaki gerçek yerleşime göre düzeltmen gerekebilir.
        self.marker_map = {
            i: {"x": self.coords[f"N{i}"][0], "y": self.coords[f"N{i}"][1], "yaw": math.pi}
            for i in range(16)
        }

        # Başlangıç ve hedef
        self.start_node = "N0"
        self.goal_node = "N15"

        # A* sonucu
        self.path, self.total_cost = self.a_star(self.start_node, self.goal_node)
        rospy.loginfo("Computed path: %s", " -> ".join(self.path))
        rospy.loginfo("Total cost: %.2f", self.total_cost)

        # En kısa yol beklenen: N0 -> N1 -> N2 -> N6 -> N7 -> N11 -> N15
        self.path_index = 0

        # Robot pose
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.current_node = "N0"

        # Kamera frame
        self.last_frame = None
        self.last_seen_time = rospy.Time.now().to_sec()

        # Görülen marker bilgileri
        self.visible_nodes = {}   # node -> {"x_err":..., "distance":..., "robot_pose":(...)}

        # AprilTag detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.new_api = False

        # ROS I/O
        self.cmd_pub = rospy.Publisher(CMD_TOPIC, Twist2DStamped, queue_size=1)
        self.viz_pub = rospy.Publisher(VIZ_PUB_TOPIC, CompressedImage, queue_size=1)

        rospy.Subscriber(CAMERA_TOPIC, CompressedImage, self.image_cb, queue_size=1, buff_size=2**24)

        rospy.loginfo("Assignment 3 A* navigation node started.")

    # -------------------------
    # A*
    # -------------------------
    def heuristic(self, node, goal):
        x1, y1 = self.coords[node]
        x2, y2 = self.coords[goal]
        return abs(x1 - x2) + abs(y1 - y2)   # Manhattan

    def reconstruct_path(self, parent, current):
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)
        path.reverse()
        return path

    def a_star(self, start, goal):
        open_list = [start]
        closed_list = []

        g = {start: 0.0}
        h = {start: self.heuristic(start, goal)}
        f = {start: g[start] + h[start]}
        parent = {}

        while open_list:
            current = min(
                open_list,
                key=lambda n: (f.get(n, float("inf")), h.get(n, float("inf")))
            )

            if current == goal:
                return self.reconstruct_path(parent, current), g[current]

            open_list.remove(current)
            closed_list.append(current)

            for neighbor, cost in self.graph[current].items():
                if neighbor in closed_list:
                    continue

                tentative_g = g[current] + cost

                if neighbor not in open_list:
                    open_list.append(neighbor)
                elif tentative_g >= g.get(neighbor, float("inf")):
                    continue

                parent[neighbor] = current
                g[neighbor] = tentative_g
                h[neighbor] = self.heuristic(neighbor, goal)
                f[neighbor] = g[neighbor] + h[neighbor]

        return [], float("inf")

    # -------------------------
    # ROS callbacks
    # -------------------------
    def image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.last_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # -------------------------
    # Detection
    # -------------------------
    def detect_markers(self, gray):
        if self.new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids

    def process_detections(self, frame, corners, ids):
        self.visible_nodes = {}

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
            marker_id = int(marker_id)

            if marker_id not in self.marker_map:
                continue

            image_pts = corners[i].reshape((4, 2)).astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                obj_pts,
                image_pts,
                K,
                D,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            cv2.drawFrameAxes(frame, K, D, rvec, tvec, MARKER_SIZE_METERS * 0.5)

            node_name = self.tag_to_node[marker_id]
           
            x_err = float(tvec[0][0])      # sağ-sol hata
            dist = float(tvec[2][0])       # ileri mesafe

            self.visible_nodes[node_name] = {
                "x_err": x_err,
                "distance": dist
                
            }

            c = image_pts.mean(axis=0).astype(int)
            txt = f"{node_name} d:{dist:.2f}"
            cv2.putText(frame, txt, (c[0] - 40, c[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        if self.visible_nodes:
            self.last_seen_time = rospy.Time.now().to_sec()

    # -------------------------
    # Motion
    # -------------------------
    def publish_cmd(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)

    # -------------------------
    # Viz
    # -------------------------
    def draw_map(self):
        canvas = np.ones((MAP_H, MAP_W, 3), dtype=np.uint8) * 255

        # edge'ler
        for n1, nbrs in self.graph.items():
            x1, y1 = self.coords[n1]
            p1 = world_to_map_px(x1, y1)

            for n2, cost in nbrs.items():
                x2, y2 = self.coords[n2]
                p2 = world_to_map_px(x2, y2)

                if n1 < n2:
                    cv2.line(canvas, p1, p2, (180, 180, 180), 2)
                    mx = (p1[0] + p2[0]) // 2
                    my = (p1[1] + p2[1]) // 2
                    cv2.putText(canvas, f"{cost}", (mx, my),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # node'lar
        for node, (x, y) in self.coords.items():
            px, py = world_to_map_px(x, y)
            color = (255, 0, 0)
            if node in self.path:
                color = (0, 180, 255)

            cv2.circle(canvas, (px, py), 14, color, -1)
            cv2.putText(canvas, node, (px - 18, py - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # robot
        cx, cy = self.coords[self.current_node]
        rx, ry = world_to_map_px(cx, cy)
        cv2.circle(canvas, (rx, ry), 10, (0, 180, 0), -1)
        cv2.putText(canvas, "ROBOT", (rx + 10, ry + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)

        cv2.putText(canvas, "Path: " + " -> ".join(self.path), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(canvas, f"Current node: {self.current_node}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        if self.path_index < len(self.path) - 1:
            cv2.putText(canvas, f"Next target: {self.path[self.path_index + 1]}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        return canvas

    def publish_visualization(self, frame):
        map_img = self.draw_map()

        h, w = frame.shape[:2]
        new_w = int((MAP_H / h) * w)
        frame_resized = cv2.resize(frame, (new_w, MAP_H))
        combined = np.hstack([frame_resized, map_img])

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode(".jpg", combined)[1]).tobytes()
        self.viz_pub.publish(msg)

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.path_index >= len(self.path) - 1:
                self.stop_robot()
                rospy.loginfo("Goal Reached")
                break

            if self.last_frame is None:
                self.stop_robot()
                rate.sleep()
                continue

            frame = cv2.undistort(self.last_frame.copy(), K, D)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            corners, ids = self.detect_markers(gray)
            self.process_detections(frame, corners, ids)

            next_node = self.path[self.path_index + 1]

            if next_node in self.visible_nodes:
                info = self.visible_nodes[next_node]

                x_err = info["x_err"]
                dist = info["distance"]

                if dist < 0.05 and abs(x_err) < 0.05:
                    self.current_node = next_node
                    self.path_index += 1
                    self.stop_robot()
                    rospy.loginfo("Reached %s", next_node)
                    rospy.sleep(0.5)
                else:
                    omega = -2.0 * x_err
                    omega = max(-3.0, min(3.0, omega))

                    v = 0.20
                    if abs(x_err) > 0.08:
                        v = 0.08

                    self.publish_cmd(v, omega)
            else:
                lost_time = rospy.Time.now().to_sec() - self.last_seen_time

                if lost_time > 1.0:
                    self.publish_cmd(0.0, 0.25)
                else:
                    self.stop_robot()

            self.publish_visualization(frame)

            rospy.loginfo_throttle(
                0.5,
                f"Current:{self.current_node} | Next:{next_node} | PathIndex:{self.path_index}"
            )

            rate.sleep()


if __name__ == "__main__":
    try:
        Assignment3Node().run()
    except rospy.ROSInterruptException:
        pass