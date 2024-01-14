import base64
import json
import math
import os
import random
import threading
import time

import airsim
import cv2
import numpy as np
import openai
import requests
from google.cloud import vision

objects_dict = {
    "turbine1": "BP_Wind_Turbines_C_1",
    "turbine2": "StaticMeshActor_2",
    "solarpanels": "StaticMeshActor_146",
    "crowd": "StaticMeshActor_6",
    "car": "StaticMeshActor_10",
    "tower1": "SM_Electric_trellis_179",
    "tower2": "SM_Electric_trellis_7",
    "tower3": "SM_Electric_trellis_8",
}


class AirSimWrapper:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.stop_thread = False
        self.flutter_thread = None

    def takeoff(self):
        self.client.takeoffAsync().join()

    def land(self):
        self.client.landAsync().join()

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        if point[2] > 0:
            self.client.moveToPositionAsync(point[0], point[1], -point[2], 5).join()
        else:
            self.client.moveToPositionAsync(point[0], point[1], point[2], 5).join()

    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(
            airsim_points,
            5,
            120,
            airsim.DrivetrainType.ForwardOnly,
            airsim.YawMode(False, 0),
            20,
            1,
        ).join()

    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    def get_position(self, object_name):
        query_string = objects_dict[object_name] + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        if object_name == "crowd": # tweak to get good cam view of crowd, or change pitch/yaw in settings.json
            return [pose.position.x_val+2, pose.position.y_val, pose.position.z_val]
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    @staticmethod
    def is_within_boundary(start_pos, current_pos, limit_radius):
        """Check if the drone is within the spherical boundary"""
        distance = math.sqrt(
            (current_pos.x_val - start_pos.x_val) ** 2
            + (current_pos.y_val - start_pos.y_val) ** 2
            + (current_pos.z_val - start_pos.z_val) ** 2
        )
        return distance <= limit_radius

    def flutter(self, speed=5, change_interval=1, limit_radius=10):
        """Simulate Brownian motion /fluttering with the drone"""
        # Takeoff and get initial position
        self.client.takeoffAsync().join()
        start_position = self.client.simGetVehiclePose().position

        while not self.stop_thread:
            pitch = random.uniform(-1, 1) 
            roll = random.uniform(-1, 1)
            yaw = random.uniform(-1, 1)

            self.client.moveByRollPitchYawrateThrottleAsync(
                roll, pitch, yaw, 0.5, change_interval
            ).join()

            current_position = self.client.simGetVehiclePose().position

            if not self.is_within_boundary(
                start_position, current_position, limit_radius
            ):
                self.client.moveToPositionAsync(
                    start_position.x_val,
                    start_position.y_val,
                    start_position.z_val,
                    speed,
                ).join()

            time.sleep(change_interval)

    def start_fluttering(self, speed=5, change_interval=1, limit_radius=10):
        self.stop_thread = False
        self.flutter_thread = threading.Thread(
            target=self.flutter, args=(speed, change_interval, limit_radius)
        )
        self.flutter_thread.start()

    def stop_fluttering(self):
        self.stop_thread = True
        if self.flutter_thread is not None:
            self.flutter_thread.join()

    def generate_circular_path(center, radius, height, segments=12):
        path = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = height
            path.append(x, y, z)
        return path

    def take_photo(self, filename="image.png"):
        responses = self.client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
        )
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        filename = os.path.normpath(filename + ".png")
        cv2.imwrite(filename, img_rgb)
        with open(filename, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_image

    def analyze_with_vision_model(self, image_data):
        # Google Vision API: https://cloud.google.com/vision/docs/object-localizer
        client = vision.ImageAnnotatorClient()
        content = base64.b64decode(image_data)
        image = vision.Image(content=content)
        objects = client.object_localization(image=image).localized_object_annotations
        return objects

    def query_language_model(self, prompt):
        """ Query inner LM to interpret json from vision model (outer LM interprets human language instructions) """
        with open("config.json", "r") as f:
            config = json.load(f)
        openai.api_key = config["OPENAI_API_KEY"]
        chat_history = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=chat_history, temperature=0
        )
        return completion.choices[0].message.content

    # Complex commands
    def count(self, object_name):
        """ Count the number of instances of target object """
        image_data = self.take_photo()
        vision_outputs = self.analyze_with_vision_model(image_data)
        # Naive: converts vision model json output to string, append to count prompt
        prompt = "\n\n Based on this json output, count the number of instances of " + object_name + " in the scene. Return a single number"
        response = self.query_language_model(str(vision_outputs) + prompt)
        print(response)
        return response
        

    def search(self, object_name, radius):
        self.fly_to(self.get_position(object_name))
        circular_path = self.generate_circular_path(
            self.get_position(object_name)[:2],
            radius,
            self.get_position(object_name)[2],
        )
        vision_outputs = ""
        for point in circular_path:
            self.fly_to(point)
            image_data = self.take_photo(str(point))
            vision_output = self.analyze_with_vision_model(image_data)
            vision_outputs += str(vision_output)
        prompt = "\n Based on these json outputs, is " + object_name + "present in the scene? Return TRUE or FALSE."
        return self.query_language_model(str(vision_outputs) + prompt)

    def get_latitude_longitude(self, object_name):
        self.fly_to(self.get_position(object_name))
        return (
            self.get_position(object_name)[0],
            self.get_drone_position(object_name)[1],
        )
