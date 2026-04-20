#!/usr/bin/env python3
import math
import rospy
import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped


# =========================
# CONFIG
# =========================
ROBOT_NAME = "bear"

CMD_TOPIC = f"/{ROBOT_NAME}/car_cmd_switch_node/cmd"
CAMERA_TOPIC = f"/{ROBOT_NAME}/camera_node/image/compressed"
VIZ_PUB_TOPIC = f"/{ROBOT_NAME}/astar_nav_viz/compressed"

MARKER_SIZE_METERS = 0.05

# Reached kararı için daha güvenli eşikler
REACH_DISTANCE = 0.095
REACH_XERR = 0.15
REQUIRED_REACH_COUNT = 2

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


# =========================
# MAP + A*
# =========================
COORDS = {
    0: (0, 0),   1: (1, 0),   2: (2, 0),   3: (3, 0),
    4: (0, 1),   5: (1, 1),   6: (2, 1),   7: (3, 1),
    8: (0, 2),   9: (1, 2),  10: (2, 2),  11: (3, 2),
    12: (0, 3), 13: (1, 3),  14: (2, 3),  15: (3, 3)
}

# PDF’ye göre düzeltilmiş graph
GRAPH = {
    0:  {1: 1.5, 4: 2.0},
    1:  {0: 1.5, 2: 1.0, 5: 2.0},
    2:  {1: 1.0, 3: 1.0, 6: 1.5},
    3:  {2: 1.0},

    4:  {0: 2.0, 8: 1.5},
    5:  {1: 2.0, 6: 1.0, 9: 2.0},
    6:  {2: 1.5, 5: 1.0, 7: 0.5, 10: 4.0},
    7:  {6: 0.5, 11: 1.5},

    8:  {4: 1.5, 9: 1.5, 12: 2.0},
    9:  {5: 2.0, 8: 1.5, 10: 2.0},
    10: {6: 4.0, 9: 2.0, 11: 1.0, 14: 1.5},
    11: {7: 1.5, 10: 1.0},

    12: {8: 2.0, 13: 1.5},
    13: {12: 1.5, 14: 2.0},
    14: {10: 1.5, 13: 2.0, 15: 1.0},
    15: {14: 1.0}
}


def heuristic(node, goal, method="manhattan"):
    x1, y1 = COORDS[node]
    x2, y2 = COORDS[goal]

    if method == "manhattan":
        return abs(x1 - x2) + abs(y1 - y2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def reconstruct_path(parent, current):
    total_path = [current]
    while current in parent and parent[current] is not None:
        current = parent[current]
        total_path.insert(0, current)
    return total_path


def a_star_search(start, goal):
    if start not in COORDS or goal not in COORDS:
        return [], float("inf")

    if start == goal:
        return [start], 0.0

    # (f, h, node)
    open_list = [(heuristic(start, goal), heuristic(start, goal), start)]
    closed_set = set()

    g_costs = {node: float("inf") for node in COORDS}
    g_costs[start] = 0.0
    parent = {start: None}

    while open_list:
        open_list.sort(key=lambda x: (x[0], x[1]))
        _, _, current = open_list.pop(0)

        if current == goal:
            return reconstruct_path(parent, current), g_costs[goal]

        closed_set.add(current)

        for neighbor, move_cost in GRAPH[current].items():
            if neighbor in closed_set:
                continue

            tentative_g = g_costs[current] + move_cost

            if tentative_g < g_costs.get(neighbor, float("inf")):
                parent[neighbor] = current
                g_costs[neighbor] = tentative_g
                h_val = heuristic(neighbor, goal, method="manhattan")
                f_val = tentative_g + h_val

                open_list = [item for item in open_list if item[2] != neighbor]
                open_list.append((f_val, h_val, neighbor))

    return [], float("inf")


def world_to_map_px(x, y):
    px = int(MAP_ORIGIN[0] + x * MAP_SCALE)
    py = int(MAP_ORIGIN[1] - y * MAP_SCALE)
    return px, py


# =========================
# MAIN NODE
# =========================
class AutonomousNavigator:
    def __init__(self):
        rospy.init_node("astar_navigator_node")

        self.search_turn_dir = 1.0
        self.turn_mode = False

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
        rospy.loginfo(f"Toplam Maliyet: {self.total_cost:.2f}")

        self.path_index = 0
        self.current_node = self.start_node
        self.state = "STARTING"

        self.reach_counter = 0
        self.last_seen_time = rospy.Time.now().to_sec()

        # Eğer sahada AprilTag kullanıyorsanız bu doğru seçim
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

    def drive(self, v, omega):
        msg = Twist2DStamped()
        msg.header.stamp = rospy.Time.now()
        msg.v = v
        msg.omega = omega
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.drive(0.0, 0.0)

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
            marker_id = int(marker_id)
            if marker_id not in COORDS:
                continue

            image_pts = corners[i].reshape((4, 2)).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, image_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if not success:
                continue

            cv2.drawFrameAxes(frame, K, D, rvec, tvec, MARKER_SIZE_METERS * 0.5)

            node_name = f"N{marker_id}"
            x_err = float(tvec[0][0])
            dist = float(tvec[2][0])

            self.visible_tags[node_name] = {
                "x_err": x_err,
                "dist": dist
            }

            c = image_pts.mean(axis=0).astype(int)
            cv2.putText(
                frame,
                f"{node_name} x:{x_err:.2f} d:{dist:.2f}",
                (c[0] - 55, c[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        if self.visible_tags:
            self.last_seen_time = rospy.Time.now().to_sec()

    # dönüş yönü fonksiyonu
    def get_turn_direction(self):
        # path başındaysa önceki yön bilinmiyor; varsayılan sağa dön
        if self.path_index == 0:
            return 1.0

        prev_node = self.path[self.path_index - 1]
        curr_node = self.path[self.path_index]
        next_node = self.path[self.path_index + 1]

        x0, y0 = COORDS[prev_node]
        x1, y1 = COORDS[curr_node]
        x2, y2 = COORDS[next_node]

        v1 = (x1 - x0, y1 - y0)
        v2 = (x2 - x1, y2 - y1)

        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]

        # cross işaretine göre dönüş yönü seç
        if cross > 0:
            return 1.0
        elif cross < 0:
            return -1.0
        else:
            # aynı doğrultuysa düz; geri dönüşse sabit bir yön seç
            if dot >= 0:
                return 0.0
            else:
                return 1.0

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
                    cv2.putText(
                        canvas, str(cost), (mx - 10, my + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
                    )

        for node_id, (x, y) in COORDS.items():
            px, py = world_to_map_px(x, y)
            color = (0, 165, 255) if node_id in self.path else (255, 0, 0)
            cv2.circle(canvas, (px, py), 14, color, -1)
            cv2.putText(
                canvas, f"N{node_id}", (px - 15, py - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )

        rx, ry = world_to_map_px(COORDS[self.current_node][0], COORDS[self.current_node][1])
        cv2.circle(canvas, (rx, ry), 10, (0, 255, 0), -1)
        cv2.putText(
            canvas, "ROBOT", (rx + 15, ry + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2, cv2.LINE_AA
        )

        cv2.putText(
            canvas, "Path: " + " -> ".join([f"N{n}" for n in self.path]),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
        )
        cv2.putText(
            canvas, f"Current node: N{self.current_node}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
        )

        if self.path_index < len(self.path) - 1:
            cv2.putText(
                canvas, f"Next target: N{self.path[self.path_index + 1]}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
            )

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

            next_node_id = self.path[self.path_index + 1]
            next_node_str = f"N{next_node_id}"

            if next_node_str in self.visible_tags:
                self.turn_mode = False
                self.state = "TRACKING"
                tag_info = self.visible_tags[next_node_str]
                x_err = tag_info["x_err"]
                dist = tag_info["dist"]

                rospy.loginfo_throttle(
                    0.2,
                    f"Durum:{self.state} | Hedef:{next_node_str} | x_err:{x_err:.3f} | dist:{dist:.3f} | cnt:{self.reach_counter}"
                )

                # Daha güvenli reached koşulu
                if dist < REACH_DISTANCE and abs(x_err) < REACH_XERR:
                    self.reach_counter += 1
                else:
                    self.reach_counter = 0

                if self.reach_counter >= REQUIRED_REACH_COUNT:
                    self.stop_robot()
                    self.current_node = next_node_id
                    self.path_index += 1
                    self.reach_counter = 0

                    if self.path_index < len(self.path) - 1:
                        self.search_turn_dir = self.get_turn_direction()
                        self.turn_mode = True

                    rospy.loginfo(f"--> Ulaşıldı: {next_node_str}")
                    rospy.sleep(0.5)
                    self.publish_visualization(frame)
                    rate.sleep()
                    continue

                # Basit P kontrol
                kp_omega = 2.5
                omega = -kp_omega * x_err
                omega = max(-2.0, min(2.0, omega))

                v = 0.18
                if abs(x_err) > 0.15:
                    v = 0.06

                self.drive(v, omega)

            else:
                self.state = "SEARCHING"
                self.reach_counter = 0

                if self.turn_mode:
                    if self.search_turn_dir == 0.0:
                        # düz devam etmesi gerekiyorsa çok hafif ileri git
                        self.drive(0.08, 0.0)
                    else:
                        self.drive(0.0, 0.35 * self.search_turn_dir)
                else:
                    self.drive(0.0, 0.25)

            self.publish_visualization(frame)
            rospy.loginfo_throttle(0.5, f"Durum: {self.state} | Hedef: {next_node_str}")
            rate.sleep()


if __name__ == "__main__":
    try:
        nav = AutonomousNavigator()
        nav.run()
    except rospy.ROSInterruptException:
        pass