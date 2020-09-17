#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
import math
import numpy

try:
    # sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    #     sys.version_info.major,
    #     sys.version_info.minor,
    #     'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('/home/hsyoon/software/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import logging
import math
import datetime
import random
import re
import weakref
import torch

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from reference_path import reference_path
from calculate_distance import lat_lng_orientation

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def mycar_grid(x, y):
    if 25.0 <= x <= 40.0 and -25 <= y <= 0:
        return np.array([1, 0, 0, 0, 0])

    elif 0 <= x <= 25.0 and -25 <= y <= 0:
        return np.array([0, 1, 0, 0, 0])

    elif 0 <= x <= 25.0 and -42 <= y <= -25:
        return np.array([0, 0, 1, 0, 0])

    elif 0 <= x <= 25.0 and y <= -42:
        return np.array([0, 0, 0, 1, 0])

    else:
        return np.array([0, 0, 0, 0, 1])


def othercar_grid(x, y):
    if 0 <= x <= 25 and -25 <= y <= 0:
        return np.array([1, 0, 0, 0, 0])

    elif -25 <= x <= 0 and -25 <= y <= 0:
        return np.array([0, 1, 0, 0, 0])

    elif -25 <= x <= 0 and 0 <= y <= 25:
        return np.array([0, 0, 1, 0, 0])

    elif 0 <= x <= 25 and 0 <= y <= 25:
        return np.array([0, 0, 0, 1, 0])

    else:
        return np.array([0, 0, 0, 0, 1])


def othercar_rad_grid(x, y):
    return (math.atan2(y, x) + math.pi) / (2 * math.pi)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.actor_role_name = args.rolename
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.car1 = None
        self.car2 = None
        self.car3 = None
        self.car4 = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.observation_space_dim = 10
        self.action_space_dim = 3

        self.blueprint_mycar = None
        self.blueprint_car1 = None
        self.blueprint_car2 = None
        self.blueprint_car3 = None
        self.blueprint_car4 = None

        self.transformation_mycar = None
        self.transformation_car1 = None
        self.transformation_car2 = None
        self.transformation_car3 = None
        self.transformation_car4 = None

        self.location_mycar = None
        self.location_car1 = None
        self.location_car2 = None
        self.location_car3 = None
        self.location_car4 = None
        self.get_blueprint_library = self.world.get_blueprint_library()
        self.camera_sensor = None
        self.camera_img = None



    def restart(self):
        # Keep same camera config if the camera manager exists.
        weather = carla.WeatherParameters(cloudyness=1.0, precipitation=10.0, sun_altitude_angle=70.0)
        self.world.set_weather(weather)

        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        # blueprint = self.world.get_blueprint_library().filter('vehicle.audi.a2')

        self.blueprint_mycar = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.tt'))
        self.blueprint_car1 = random.choice(self.world.get_blueprint_library().filter('vehicle.tesla.model3'))
        self.blueprint_car2 = random.choice(self.world.get_blueprint_library().filter('vehicle.bmw.grandtourer'))
        self.blueprint_car3 = random.choice(self.world.get_blueprint_library().filter('vehicle.chevrolet.impala'))
        self.blueprint_car4 = random.choice(self.world.get_blueprint_library().filter('vehicle.ford.mustang'))


        # Spawn the player.
        # And in this place, spawn other players with given spawn point and autopilot=True
        if self.player is not None:
            self.destroy()
            self.player = None

        """
        # Intersection 4
        mycar_transform = carla.Transform(carla.Location(x=-38.211617, y=135.013290, z=2), carla.Rotation(0, 0, 0))
        car1_transformation = carla.Transform(carla.Location(x=2.0, y=154.3, z=2), carla.Rotation(0, -90, 0))
        car2_transformation = carla.Transform(carla.Location(x=-6.1, y=104.5, z=2), carla.Rotation(0, 90, 0))
        """

        # Roundabout
        self.transformation_mycar = carla.Transform(carla.Location(38.9, -4.4, 2), carla.Rotation(0, 180, 0))
        self.transformation_car1 = carla.Transform(carla.Location(40.7, -7.9, 2), carla.Rotation(0, 180, 0))
        self.transformation_car2 = carla.Transform(carla.Location(22.471565, 4.197405, 2), carla.Rotation(0.089967, -79.148270, -0.010895))
        self.transformation_car3 = carla.Transform(carla.Location(13.526010, 14.151862, 1), carla.Rotation(0.206647, -49.756748, 0.130238))
        self.transformation_car4 = carla.Transform(carla.Location(-1.023689, 20.312210, 1), carla.Rotation(0.134186, 2.310838, 0.005359))

        self.location_mycar = carla.Location(38.9, -4.4, 2)
        self.location_car1 = carla.Location(40.7, -7.9, 2)
        self.location_car2 = carla.Location(22.471565, 4.197405, 2)
        self.location_car3 = carla.Location(13.526010, 14.151862, 1)
        self.location_car4 = carla.Location(-1.023689, 20.312210, 1)

        try:
            self.car1 = self.world.try_spawn_actor(self.blueprint_car1, self.transformation_car1)
            self.car1.set_autopilot(True)
        except AttributeError as ae:
            self.car1 = None

        try:
            self.car2 = self.world.try_spawn_actor(self.blueprint_car2, self.transformation_car2)
            self.car2.set_autopilot(True)
        except AttributeError as ie:
            self.car2 = None

        # try:
        #     self.car3 = self.world.try_spawn_actor(self.blueprint_car3, self.transformation_car3)
        #     self.car3.set_autopilot(True)
        # except AttributeError as ie:
        #     self.car3 = None
        #
        # try:
        #     self.car4 = self.world.try_spawn_actor(self.blueprint_car4, self.transformation_car4)
        #     self.car4.set_autopilot(True)
        # except AttributeError as ie:
        #     self.car4 = None
        #

        while self.player is None:
            self.player = self.world.try_spawn_actor(self.blueprint_mycar, self.transformation_mycar)

        # blueprint_tmp = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.etron'))
        # tmp_transform = carla.Transform(carla.Location(69.489059, -204.821091, 0.1), carla.Rotation(0.061110, 2.803965, -0.020020))
        # car_tmp = None
        # while car_tmp is None:
        #     try:
        #         car_tmp = self.world.try_spawn_actor(blueprint_tmp, tmp_transform)
        #     except AttributeError as ie:
        #         car_tmp = None

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.sem_bp.set_attribute("image_size_x", str(192))
        self.sem_bp.set_attribute("image_size_y", str(192))
        self.sem_bp.set_attribute("fov", str(105))
        sem_location = carla.Location(0, 0, 1000)
        sem_rotation = carla.Rotation(0, 270, 0)
        sem_transform = carla.Transform(sem_location, sem_rotation)

        self.sem_cam = self.world.spawn_actor(self.sem_bp, sem_transform, attach_to=self.player, attachment_type=carla.AttachmentType.Rigid)

        self.sem_cam.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def tick_without_clock(self):
        self.world.tick()

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player, self.car1, self.car2, self.car3, self.car4]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def get_snapshot(self):
        return self.world.get_snapshot()

    def get_actors(self):
        return self.world.get_actors()

    def get_actor(self, id):
        return self.world.get_actor(id)

    def get_settings(self):
        return self.world.get_settings()

    def apply_settings(self, settings):
        self.world.apply_settings(settings)

    def try_spawn_actor(self, car_blueprint, car_transform):
        return self.world.try_spawn_actor(car_blueprint, car_transform)
    
    
    
# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 3.333 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=int)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    pygame.font.init()
    episode = 1
    success_episode = 0
    total_obs = None
    total_action = None
    total_reward = None
    total_len = None

    # Just get carla client and make world
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)

    display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    hud = HUD(args.width, args.height)
    world = World(client.get_world(), hud, args)
    world.restart()

    # --------------
    # Add a new semantic segmentation camera to my ego
    # --------------


    # This time, a color converter is applied to the image, to get the semantic segmentation view

    if 'mini' in args.filename:
        world.observation_space_dim = 3

    while True:
        obs = np.array([])
        action = np.array([])
        reward = np.array([])
        len = np.array([])
        termination = False
        save = False
        count = 0
        episode_step = 0
        # print("save", save)
        # print("termination", termination)
        # world.tick_without_clock()


        while termination is False:
            episode_step = episode_step + 1
            # print(carla.Waypoint.lane_id)

            if episode_step > 600:
                save = False
                termination = True
                episode = episode + 1
                print("TOO LONG STEP!")
                world.restart()
                break

            clock = pygame.time.Clock()

            """
            # Intersection 4
            destination_location = carla.Location(x=1.869288, y=111.384148, z=-0.004142)
            """

            # Roundabout
            destination_location = carla.Location(4.1, -46.9, 0)
            location_intersection = carla.Location(x=26.206285, y=-4.993992, z=-0.002554)

            """
            termination = True if 
            1. Car gets to the destination point (save = True) 
            2. Go left at the intersection (save = False)
            
            3. Episode step becomes larger than 600 
            """

            clock.tick_busy_loop(60)
            controller = KeyboardControl(world, True)
            if controller.parse_events(client, world, clock):
                return

            world_snapshot = world.get_snapshot()

            # Find my car and check if i am
            # 1. going right left direction
            # 2. reaching to my destination point
            actor_snapshot_mycar = None
            tmp_obs = np.zeros(world.observation_space_dim)

            for actor_snapshot in world_snapshot:
                actual_actor = world.get_actor(actor_snapshot.id)

                # ===================================================================================================================
                # Temporary Observation

                # 0~1 : x, y of the location of the intersection
                # 2~3 : x, y of the location of mycar
                # 4~9 : properties of car1 (x,y value of location, velocity, acceleration)
                # 10~15 : properties of car2 (x,y value of location, velocity, acceleration)
                # 16~21 : properties of car3 (x,y value of location, velocity, acceleration)
                # 22~27 : properties of car4 (x,y value of location, velocity, acceleration)
                # ===================================================================================================================

                # tmp_obs[0] = location_intersection.x
                # tmp_obs[1] = location_intersection.y

                if 'vehicle.audi.tt' in actual_actor.type_id:
                    actor_snapshot_mycar = actor_snapshot

                    lat, lng, ori = lat_lng_orientation(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y, actor_snapshot.get_transform().rotation.yaw, reference_path)

                    tmp_obs[0] = actor_snapshot.get_transform().location.x
                    tmp_obs[1] = actor_snapshot.get_transform().location.y

                    tmp_obs[2] = lat
                    tmp_obs[3] = lng
                    tmp_obs[4] = ori
                    tmp_obs[5] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2)) / 10
                    # tmp_obs[2] = actor_snapshot.get_transform().location.x
                    # tmp_obs[3] = actor_snapshot.get_transform().location.y

                    # tmp_obs[0] = get_grid_location(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                    # tmp_obs[0:5] = mycar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)
                    # tmp_obs[5] = actor_snapshot.get_transform().rotation.yaw  # orientation
                    # tmp_obs[5] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2)) / 10  # speed, not velocity
                    # tmp_obs[6] = 1.0 - float(abs(actor_snapshot.get_transform().location.x - destination_location.x) + abs(actor_snapshot.get_transform().location.y - destination_location.y)) / float(abs(world.location_mycar.x - destination_location.x) + abs(world.location_mycar.y - destination_location.y))


                    # destination_distance = math.sqrt(math.pow(actor_snapshot.get_transform().location.x - destination_location.x, 2) + math.pow(actor_snapshot.get_transform().location.y - destination_location.y, 2) + math.pow(actor_snapshot.get_transform().location.z - destination_location.z, 2))
                    destination_distance = actor_snapshot.get_transform().location.y - destination_location.y

                    if count % 500 is 0:
                        print("=================================================================================")
                        print("                               EXPERT DEMO LOG")
                        print("episode :", episode)
                        print("success episode :", success_episode)
                        print("step :", count)
                        print("destination distance :", destination_distance)
                        print("current steer :", actual_actor.get_control().steer)
                        print("current throttle :", actual_actor.get_control().throttle)
                        print("current brake :", actual_actor.get_control().brake)
                        print("=================================================================================")
                    """
                    # Intersection 4
                    if -18.0 < actor_snapshot.get_transform().location.x < -14.0 and actual_actor.get_control().steer >= 0:
                    """

                    if actual_actor.get_control().brake == 0:
                        tmp_brake = 0
                    else:
                        tmp_brake = 1

                    # action = np.append(action, [actual_actor.get_control().steer, actual_actor.get_control().throttle, actual_actor.get_control().brake])
                    action = np.append(action, [actual_actor.get_control().steer, actual_actor.get_control().throttle, tmp_brake])
                    reward = np.append(reward, [0.5])

                    # Roundabout
                    if actor_snapshot.get_transform().location.x < 10 and actual_actor.get_control().steer < 0 or actor_snapshot.get_transform().location.x < -0.5:
                        save = False
                        termination = True
                        episode = episode + 1

                        print("\n\nWrong way with x value {} and steering angle {}\n\n".format(actor_snapshot.get_transform().location.x, actual_actor.get_control().steer))
                        break

                    # if destination_distance <= 5.0:
                    #     print("episode_step", episode_step)

                    if destination_distance <= 1.0 and episode_step > 200:
                        save = True
                        termination = True
                        episode = episode + 1
                        success_episode = success_episode + 1

                        print("\n\nSuccess!!")
                        break

            # ===================================================================================================================
            # Record Observation Space

            # Assign the indexes of the observation values for own actor car
            # ===================================================================================================================

            for actor_snapshot in world_snapshot:
                actual_actor = world.get_actor(actor_snapshot.id)
                # FOR TABULAR GRID
                # if 'vehicle.tesla.model3' in actual_actor.type_id:
                #
                #     tmp_obs[8:13] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                #     tmp_obs[13] = actor_snapshot.get_transform().rotation.yaw  # orientation
                #     tmp_obs[14] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
                #
                # if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
                #     tmp_obs[15:20] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                #     tmp_obs[20] = actor_snapshot.get_transform().rotation.yaw  # orientation
                #     tmp_obs[21] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
                #
                # if 'vehicle.chevrolet.impala' in actual_actor.type_id:
                #     tmp_obs[22:27] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                #     tmp_obs[27] = actor_snapshot.get_transform().rotation.yaw  # orientation
                #     tmp_obs[28] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity
                #
                # if 'vehicle.ford.mustang' in actual_actor.type_id:
                #     tmp_obs[29:34] = othercar_grid(actor_snapshot.get_transform().location.x, actor_snapshot.get_transform().location.y)  # grid location
                #     tmp_obs[34] = actor_snapshot.get_transform().rotation.yaw  # orientation
                #     tmp_obs[35] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2))  # speed, not velocity

                # FOR RADIAN GRID
                if 'vehicle.tesla.model3' in actual_actor.type_id:
                    if math.sqrt(math.pow(actor_snapshot.get_transform().location.x, 2) + math.pow(actor_snapshot.get_transform().location.y, 2)) <= 25.0:
                        tmp_obs[6] = othercar_rad_grid(actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location.x, actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location.y)  # grid location
                    else:
                        tmp_obs[6] = 2.0
                    # tmp_obs[9] = actor_snapshot.get_transform().rotation.yaw  # orientation
                    tmp_obs[7] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2)) / 10  # speed, not velocity

                if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
                    if math.sqrt(math.pow(actor_snapshot.get_transform().location.x, 2) + math.pow(actor_snapshot.get_transform().location.y, 2)) <= 25.0:
                        tmp_obs[8] = othercar_rad_grid(actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location.x, actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location.y)  # grid location
                    else:
                        tmp_obs[8] = 2.0
                    # tmp_obs[12] = actor_snapshot.get_transform().rotation.yaw  # orientation
                    tmp_obs[9] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2)) / 10  # speed, not velocity

                # if 'vehicle.chevrolet.impala' in actual_actor.type_id:
                #     if math.sqrt(math.pow(actor_snapshot.get_transform().location.x, 2) + math.pow(actor_snapshot.get_transform().location.y, 2)) <= 25.0:
                #         tmp_obs[11] = othercar_rad_grid(actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location().x, actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location().y)  # grid location
                #     else:
                #         tmp_obs[11] = 2.0
                #     # tmp_obs[15] = actor_snapshot.get_transform().rotation.yaw  # orientation
                #     tmp_obs[12] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2)) / 10  # speed, not velocity
                #
                # if 'vehicle.ford.mustang' in actual_actor.type_id:
                #     if math.sqrt(math.pow(actor_snapshot.get_transform().location.x, 2) + math.pow(actor_snapshot.get_transform().location.y, 2)) <= 25.0:
                #         tmp_obs[13] = othercar_rad_grid(actor_snapshot.get_transform().location.x - actor_snapshot_mycar.get_transform().location().x, actor_snapshot.get_transform().location.y - actor_snapshot_mycar.get_transform().location().y)  # grid location
                #     else:
                #         tmp_obs[13] = 2.0
                #     # tmp_obs[18] = actor_snapshot.get_transform().rotation.yaw  # orientation
                #     tmp_obs[14] = math.sqrt(math.pow(actor_snapshot.get_velocity().x, 2) + math.pow(actor_snapshot.get_velocity().y, 2)) / 10  # speed, not velocity

            # ===================================================================================================================
            # Actor Boundary Check

            # Check if there are the cars which are out of boundary.
            # For actor which is out of boundary, respawn the actor to initial point for its own point.
            # ===================================================================================================================

            for actor_snapshot in world_snapshot:
                actual_actor = world.get_actor(actor_snapshot.id)
                if 'vehicle.tesla.model3' in actual_actor.type_id:
                    if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:

                        actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_transform(world.transformation_car1)

                if 'vehicle.bmw.grandtourer' in actual_actor.type_id:
                    if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:

                        actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                        actual_actor.set_transform(world.transformation_car2)

                # if 'vehicle.chevrolet.impala' in actual_actor.type_id:
                #     if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                #         actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                #         actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                #         actual_actor.set_transform(world.transformation_car3)
                #
                # if 'vehicle.ford.mustang' in actual_actor.type_id:
                #     if actor_snapshot.get_transform().location.x < -2.3 or actor_snapshot.get_transform().location.x > 45.0 or actor_snapshot.get_transform().location.y < -46.9 or actor_snapshot.get_transform().location.y > 25.0:
                #         actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                #         actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                #         actual_actor.set_transform(world.transformation_car4)

            # print("tmp_obs_mycar", tmp_obs[0:4])
            # print("tmp_obs_othercar1", tmp_obs[4:6])
            # print("tmp_obs_othercar2", tmp_obs[6:8])

            obs = np.append(obs, tmp_obs)

            if save is True:
                break

            count = count + 1
            world.render(display)
            pygame.display.flip()

        # ===================================================================================================================
        # Save Expert Demonstrations.

        # Save the recorded expert demonstrations.
        # 100 demos are recorded in one .pt file.
        # ===================================================================================================================

        if save is True:
            max_time_step = 600
            max_episode = 100

            # print(obs.shape)
            # print(action.shape)
            # print(reward.shape)

            if obs.shape[0] < max_time_step * world.observation_space_dim:
                # Get current time step (0~99). Fill the empty spaces of expert demo.
                # Example below
                # If episode ends with 520 stesp, other 80 steps are filled with the end values.
                for i in range(max_time_step - int(obs.shape[0]/world.observation_space_dim)):
                    obs = np.append(obs, obs[-(world.observation_space_dim+1):-1])
                    action = np.append(action, action[-(world.action_space_dim+1):-1])
                    reward = np.append(reward, reward[-1])

            obs = np.reshape(obs, (obs.shape[0], 1))
            action = np.reshape(action, (action.shape[0], 1))
            reward = np.reshape(reward, (reward.shape[0], 1))

            # print("obs_raw", obs.shape)
            # print("action_raw", action.shape)
            # print("reward_raw", reward.shape)
            # print("len_raw", len.shape)

            obs = np.array([np.reshape(obs, (int(obs.shape[0]/world.observation_space_dim), world.observation_space_dim))])
            action = np.array([np.reshape(action, (int(action.shape[0]/world.action_space_dim), world.action_space_dim))])
            reward = np.array([np.reshape(reward, (int(reward.shape[0]), 1))])

            print("obs", obs.shape)
            print("action", action.shape)
            print("reward", reward.shape)

            if total_obs is None:
                total_obs = obs
                total_action = action
                total_reward = reward

            else:
                total_obs = np.append(total_obs, obs, axis=0)
                total_action = np.append(total_action, action, axis=0)
                total_reward = np.append(total_reward, reward, axis=0)

            if total_obs.shape[0] is max_episode:

                tmp_len = np.array([])
                for i in range(max_episode):
                    tmp_len = np.append(tmp_len, np.array([max_time_step]))

                states = torch.from_numpy(total_obs).float()
                actions = torch.from_numpy(total_action).float()
                rewards = torch.from_numpy(total_reward).float()
                lens = torch.from_numpy(tmp_len).long()

                data = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'lengths': lens
                }

                # now = datetime.now()
                # now_date = str(now.year)[-2:] + str(now.month).zfill(2) + str(now.day).zfill(2)
                # now_time = str(now.hour).zfill(2) + str(now.minute).zfill(2)

                # torch.save(data, '/home/hsyoon/job/vilab-pytorch-rl_exploration/pytorch-a2c-ppo-acktr-gail-master/gail_experts/data-%s-%s.pt' % (args.filename, success_episode))
                torch.save(data, './expert_demo/data-%s-%s.pt' % (args.filename, success_episode))
                print("=================================================================================")
                print(" DATA SAVED COMPLETED ")
                print("states:", total_obs.shape)
                print("actions:", total_action.shape)
                print("rewards:", total_reward.shape)
                print("tmp_len", tmp_len)
                print("=================================================================================")

                total_obs = None
                total_action = None
                total_reward = None
                total_len = None

        # Respawn the mycar actor.
        for actor_snapshot in world_snapshot:
            actual_actor = world.get_actor(actor_snapshot.id)
            if 'vehicle.audi.tt' in actual_actor.type_id:

                actual_actor.set_transform(world.transformation_mycar)
                # actual_actor.set_velocity(carla.Vector3D(x=0, y=0, z=0))
                # actual_actor.set_angular_velocity(carla.Vector3D(x=0, y=0, z=0))
                actual_actor.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False))

                # settings = world.get_settings()
                # settings.synchronous_mode = True
                # settings.fixed_delta_seconds = 0.5
                # world.apply_settings(settings)

                # actual_actor.destroy()
                # world.mycar = None
                #
                # while world.mycar is None:
                #     try:
                #         world.mycar = world.try_spawn_actor(world.blueprint_mycar, world.transformation_mycar)
                #         world.mycar.set_autopilot(True)
                #     except AttributeError as ae:
                #         world.mycar = None

                # world.tick_without_clock()
                # world.tick(clock)
                print("MYCAR RESPAWNED!!")

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--filename',
        default='20200101',
        help='Saving file name')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
