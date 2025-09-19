# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import USD_AVAILABLE, get_test_devices

devices = get_test_devices()


class TestUsdLoops(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_loop(self):
        builder = newton.ModelBuilder()

        usd_path = os.path.join(os.path.dirname(__file__), "assets", "_4_bar/_4_bar.usd")
        usd_path = os.path.join(os.path.dirname(__file__), "assets", "digit.usda")

        # usd_path = "/home/gnoya/Newton/newton/newton/examples/assets/g1_o.usd"
        usd_path = "/home/gnoya/Newton/newton/newton/examples/assets/humanoid.usda"

        # usd_path = "/home/gnoya/Newton/newton/test.usda"
        # usd_path = "/home/gnoya/Downloads/cassie_newton_repro.usda"
        # usd_path = "/home/gnoya/Downloads/digit_v4.usda"
        # usd_path = "/home/gnoya/Newton/newton/g1.usd"

        results = builder.add_usd(
            usd_path,
            # collapse_fixed_joints=True,
        )

        # UNCOMMENT TO ADD GROUND PLANE
        builder.add_ground_plane()

        start_rot = wp.quat_identity()

        # Ensure base is high enough to clear the ground
        builder.joint_q[:7] = [0.0, 0.0, 1.5, *start_rot]

        model = builder.finalize()

        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver="newton",
            integrator="euler",
            nefc_per_env=500,
            ncon_per_env=150,
            cone="elliptic",
            impratio=100,
            iterations=100,
            ls_iterations=50,
        )

        sim_time = 0.0
        frame_dt = 1 / 100
        num_frames = 3000000
        sim_substeps = 10
        sim_dt = frame_dt / sim_substeps

        renderer = newton.viewer.ViewerGL()

        renderer.set_model(model)

        control = model.control()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        for _ in range(num_frames):
            contacts = model.collide(state_0)

            for _ in range(sim_substeps):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0

            renderer.begin_frame(sim_time)  # renderer
            renderer.log_state(state_0)
            renderer.end_frame()

            sim_time += frame_dt


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
