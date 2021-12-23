#!/usr/bin/env python3

# Copyright (2021-) Shahruk Hossain <shahruk10@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================



import numpy as np


class RefTdnnNarrow():

    config = """
input-node name=input dim=3
component-node name=tdnn1.affine component=tdnn1.affine input=Append(Offset(input, -2), Offset(input, -1), input, Offset(input, 1), Offset(input, 2))
component-node name=tdnn1.relu component=tdnn1.relu input=tdnn1.affine
component-node name=tdnn1.batchnorm component=tdnn1.batchnorm input=tdnn1.relu
component-node name=tdnn2.affine component=tdnn2.affine input=Append(Offset(tdnn1.batchnorm, -2), tdnn1.batchnorm, Offset(tdnn1.batchnorm, 2))
component-node name=tdnn2.relu component=tdnn2.relu input=tdnn2.affine
component-node name=tdnn2.batchnorm component=tdnn2.batchnorm input=tdnn2.relu
component-node name=tdnn3.affine component=tdnn3.affine input=Append(Offset(tdnn2.batchnorm, -3), tdnn2.batchnorm, Offset(tdnn2.batchnorm, 3))
component-node name=tdnn3.relu component=tdnn3.relu input=tdnn3.affine
component-node name=tdnn3.batchnorm component=tdnn3.batchnorm input=tdnn3.relu
component-node name=tdnn4.affine component=tdnn4.affine input=tdnn3.batchnorm
component-node name=tdnn4.relu component=tdnn4.relu input=tdnn4.affine
component-node name=tdnn4.batchnorm component=tdnn4.batchnorm input=tdnn4.relu
component-node name=tdnn5.affine component=tdnn5.affine input=tdnn4.batchnorm
component-node name=tdnn5.relu component=tdnn5.relu input=tdnn5.affine
component-node name=tdnn5.batchnorm component=tdnn5.batchnorm input=tdnn5.relu
component-node name=output.affine component=output.affine input=tdnn5.batchnorm
output-node name=output input=output.affine objective=linear
""".split("\n")[1:19]

    components = [
        {
            "name": "tdnn1.affine",
            "type": "<NaturalGradientAffineComponent>",

            "params": np.float32([
                0.5341127, 0.2938498, -0.3873977, -0.3356587, 0.1053552, -0.01093684, 0.4139098, -0.1083158, 0.09591173, -0.01700557, -0.2182646, 0.4705898, -0.1530701, -0.1900474, -0.1892251,
                -0.2840329, 0.2710086, 0.06283575, -0.5326806, -0.2057315, -0.3018429, -0.2249597, 0.0657663, -0.09736361, -0.1753622, 0.02000471, -0.4229214, 0.1025034, -0.164837, -0.1338404,
                -0.06682256, -0.06576736, 0.1323554, 0.3581094, 0.1110945, 0.3795628, -0.4286604, 0.1879511, -0.2402663, 0.002876687, -0.3988708, -0.6713861, 0.4500691, 0.05344684, 0.06198638,
                0.2828276, 0.1666822, 0.4836042, -0.007175575, 0.3461935, -0.3897989, -0.09620879, 0.4940447, 0.0460803, -0.2847865, 0.1924698, -0.3911248, -0.1854927, -0.7229586, -0.04206589,
                -0.1903168, -0.1914652, 0.2096246, 0.2375592, 0.09258451, -0.02263471, -0.03167005, -0.1684487, -0.2598038, 0.1596587, 0.2579161, -0.3230085, -0.1323228, -0.02893267, -0.3110466,
            ]).reshape(5, 15),

            "bias": np.float32([
                -1.266862, 0.213418, 1.202663, -1.261553, 1.110903,
            ]),
        },
        {
            "name": "tdnn1.relu",
            "type": "<RectifiedLinearComponent>",
            "dim": 5,
            "value-avg": np.array([], dtype=np.float32),
            "deriv-avg": np.array([], dtype=np.float32),
            "count": 10.0,
            "oderiv-rms": np.array([], dtype=np.float32),
            "oderiv-count": 0,
        },
        {
            "name": "tdnn1.batchnorm",
            "type": "<BatchNormComponent>",
            "dim": 5,
            "block-dim": 5,
            "epsilon": 0.001,
            "target-rms": 1.0,
            "test-mode": True,
            "count": 10.0,
            "stats-mean": np.float32([5.525478, 5.038039, 2.2788742, 5.2100024, 7.274048]),
            "stats-var": np.float32([3.806511, 3.4790154, 0.86363363, 3.8433418, 5.611824]),
        },
        {
            "name": "tdnn2.affine",
            "type": "<NaturalGradientAffineComponent>",

            "params": np.float32([
                -0.04512721, 0.00186575, 0.1797781, 0.5000758, -0.2908441, 0.1668068, -0.1956375, -0.001371455, -0.1249213, 0.008259594, 0.218361, 0.1446064, -0.05833546, 0.2178984, -0.5882257,
                0.06041425, -0.436099, 0.4928613, -0.08216596, -0.2784598, 0.3155676, -0.08583137, -0.02987163, 0.0693356, -0.1280177, -0.1131059, 0.3724145, -0.05390729, 0.154342, 0.2367409,
                0.06135295, 0.03583226, 0.06458891, -0.01182718, -0.1791641, -0.1608031, -0.1978755, -0.4350127, 0.1136643, 0.4339, 0.06960054, 0.1252459, 0.1771367, -0.03833843, -0.02378256,
                -0.1620382, 0.06846633, 0.286598, 0.3530979, -0.6612704, 0.07164574, -0.1036536, -0.4043652, 0.2362045, -0.2503228, 0.408674, 0.06512548, 0.521822, 0.2629726, -0.126479,
                0.1077092, -0.3126723, -0.07071986, -0.1102077, -0.01302463, -0.3115547, 0.06287698, 0.1327096, 0.09239054, 0.01316129, -0.02240477, 0.01688844, 0.1337554, -0.1060426, 0.03006054,
                0.1458119, 0.1539762, 0.3467998, -0.1950718, -0.1490429, -0.1981185, -0.2046856, 0.2799147, 0.1948479, -0.2885794, 0.1823813, 0.04512719, 0.1333404, 0.1980345, -0.2066861,
                -0.137983, 0.2632469, -0.4779148, -0.08819931, 0.1778037, -0.1104125, 0.1281125, 0.4009565, -0.2983454, 0.03359888, 0.1922089, -0.1038533, 0.1042123, 0.04303175, 0.05270234,
                -0.02542716, 0.1486124, -0.1354046, 0.272972, -0.04695081, -0.2644137, -0.2361274, 0.1111716, 0.6023751, 0.2019961, 0.04472407, 0.4372472, -0.06977846, 0.3173658, 0.0653147,
            ]).reshape(8, 15),

            "bias": np.float32([
                2.351039, 0.8741217, -1.005901, -0.1507833, 0.07875402, 0.6556886, 0.05149465, -0.5317125,
            ]),
        },
        {
            "name": "tdnn2.relu",
            "type": "<RectifiedLinearComponent>",
            "dim": 8,
            "value-avg": np.array([], dtype=np.float32),
            "deriv-avg": np.array([], dtype=np.float32),
            "count": 10.0,
            "oderiv-rms": np.array([], dtype=np.float32),
            "oderiv-count": 0,
        },
        {
            "name": "tdnn2.batchnorm",
            "type": "<BatchNormComponent>",
            "dim": 8,
            "block-dim": 8,
            "epsilon": 0.001,
            "target-rms": 1.0,
            "test-mode": True,
            "count": 10.0,
            "stats-mean": np.float32([5.525478, 5.038039, 2.2788742, 5.2100024, 7.274048, 2.927882062, 3.962218457, 4.214613874]),
            "stats-var": np.float32([3.806511, 3.4790154, 0.86363363, 3.8433418, 5.611824, 1.3066893, 2.3252077, 2.600586]),
        },
        {
            "name": "tdnn3.affine",
            "type": "<NaturalGradientAffineComponent>",

            "params": np.float32([
                0.1735435, 0.3577363, -0.06200178, 0.3333988, 0.2439802, 0.08663209, 0.2970491, 0.1092892, -0.0743224, 0.1115896, 0.2253682, 0.08286656, -0.1470744, 0.4329704, 0.04559321, -0.2484298, -0.09268007, 0.1388071, 0.04322106, -0.04351111, 0.2321791, 0.1778221, -0.08887652, -0.2046162,
                -0.2962161, -0.2179276, 0.2042319, 0.2550953, 0.2307595, -0.02627961, -0.1662541, -0.004100056, -0.5273287, 0.5314235, -0.4168359, 0.1953757, -0.1672027, 0.3499477, 0.1359487, -0.0849737, -0.08090358, 0.1795989, -0.1566454, -0.03196386, 0.213928, -0.09282452, -0.1204706, -0.07235739,
                0.1866009, 0.05259007, -0.3594273, -0.0658913, -0.09493823, -0.09701601, -0.05536911, 0.3116052, -0.02151296, -0.05254015, 0.07858268, -0.1035843, 0.006203727, -0.2423346, 0.1067014, -0.1194547, -0.3473672, 0.06638719, -0.24035, 0.3418832, 0.0430114, -0.1756357, -0.2396338, 0.1261413,
                -0.01289022, 0.5263477, -0.08688384, 0.1579813, 0.2194913, -0.2097877, -0.05193809, 0.2983095, 0.2296223, 0.19999, -0.06277496, -0.09233396, -0.1603582, 0.09478632, 0.04954312, 0.02609112, 0.1506261, -0.1138912, 0.03036199, -0.171111, -0.2366532, -0.1494045, 0.269658, -0.3766864,
                0.04809963, -0.2384379, 0.07576179, -0.1725348, 0.4285876, -0.1100774, 0.2053145, 0.08831132, 0.1187498, -0.1761742, -0.1940557, -0.1598636, 0.2397125, 0.01898255, -0.1136835, -0.3612231, -0.2584361, -0.2451524, 0.01448271, -0.06063742, -0.02414056, -0.1361191, -0.08148662, 0.2277277,
                -0.1072426, -0.1892205, 0.06173211, -0.1168213, 0.04704026, 0.04231368, 0.2045222, 0.06752628, 0.02690721, 0.02691013, 0.1903378, 0.1797938, 0.2818761, -0.4407568, 0.04322208, -0.1110673, -0.06015273, 0.3038835, 0.1499033, 0.02966169, -0.05127065, -0.2294173, 0.02468928, 0.08550644,
                0.237895, 0.2569878, 0.7536578, -0.1890832, -0.06858391, 0.2049391, -0.1184761, -0.2692471, 0.2224183, -0.08390709, -0.1879581, 0.1169342, -0.2626758, 0.05974394, -0.2006411, -0.09207049, 0.1195247, -0.2613073, 0.1480596, 0.04464027, 0.1843224, -0.08147421, 0.1649689, 0.1719963,
                -0.004668477, 0.1403263, 0.004933849, -0.1786258, 0.1928146, -0.2157856, -0.1334273, -0.1549241, -0.03428869, -0.1360482, -0.1401113, -0.5551727, -0.09718414, 0.06290051, -0.2224454, 0.09440713, 0.07533172, 0.1245146, -0.1179698, 0.04672097, -0.01817632, 0.2305191, 0.1114058, 0.0002221979
            ]).reshape(8, 24),

            "bias": np.float32([
                -1.244074, 0.9767587, 0.4723652, 0.368234, 0.003425274, -0.2573405, -0.2746394, -0.6463977,
            ]),
        },
        {
            "name": "tdnn3.relu",
            "type": "<RectifiedLinearComponent>",
            "dim": 8,
            "value-avg": np.array([], dtype=np.float32),
            "deriv-avg": np.array([], dtype=np.float32),
            "count": 10.0,
            "oderiv-rms": np.array([], dtype=np.float32),
            "oderiv-count": 0,
        },
        {
            "name": "tdnn3.batchnorm",
            "type": "<BatchNormComponent>",
            "dim": 8,
            "block-dim": 8,
            "epsilon": 0.001,
            "target-rms": 1.0,
            "test-mode": True,
            "count": 10.0,
            "stats-mean": np.float32([5.525478, 5.038039, 2.2788742, 5.2100024, 7.274048, 2.927882062, 3.962218457, 4.214613874]),
            "stats-var": np.float32([3.806511, 3.4790154, 0.86363363, 3.8433418, 5.611824, 1.3066893, 2.3252077, 2.600586]),
        },
        {
            "name": "tdnn4.affine",
            "type": "<NaturalGradientAffineComponent>",

            "params": np.float32([
                0.1887785, 0.7816643, -0.2720307, 0.4856474, -0.1846417, -0.4450786, 0.1717921, 0.3962565,
                -0.2021513, -0.1605111, -0.1356536, -0.1265672, -0.2699598, -0.1566673, 0.07661749, -0.55981,
                -0.04144005, -0.1566437, -0.1053632, -0.2119815, 0.5083517, 0.1250263, 0.1245799, -0.1480865,
                -0.1889259, 0.09798139, -0.4255897, -0.09988786, 0.1606722, 0.3841815, 0.5042814, -0.4705917,
                0.08592279, 0.09578219, 0.5721941, -0.3505367, -0.06354748, 0.03511445, -0.4785058, 0.1021285,
                0.01250202, -0.4788841, 0.01555135, -0.4705239, -0.7931228, 0.2703354, 0.453099, 0.07863578,
                0.2795593, -0.7947801, 0.2545326, -0.534194, -0.3375342, 0.4088709, -0.27137, 0.7428765,
                -0.6175874, 0.1528881, 0.05184618, 0.3815357, -0.1606709, -0.4082896, 0.4514395, -0.1779884,
            ]).reshape(8, 8),

            "bias": np.float32([
                -1.648215, 0.9167002, 0.1469106, -2.331616, 0.5643279, -1.672481, 0.6631289, -1.37326,
            ]),
        },
        {
            "name": "tdnn4.relu",
            "type": "<RectifiedLinearComponent>",
            "dim": 8,
            "value-avg": np.array([], dtype=np.float32),
            "deriv-avg": np.array([], dtype=np.float32),
            "count": 10.0,
            "oderiv-rms": np.array([], dtype=np.float32),
            "oderiv-count": 0,
        },
        {
            "name": "tdnn4.batchnorm",
            "type": "<BatchNormComponent>",
            "dim": 8,
            "block-dim": 8,
            "epsilon": 0.001,
            "target-rms": 1.0,
            "test-mode": True,
            "count": 10.0,
            "stats-mean": np.float32([5.525478, 5.038039, 2.2788742, 5.2100024, 7.274048, 2.927882062, 3.962218457, 4.214613874]),
            "stats-var": np.float32([3.806511, 3.4790154, 0.86363363, 3.8433418, 5.611824, 1.3066893, 2.3252077, 2.600586]),
        },
        {
            "name": "tdnn5.affine",
            "type": "<NaturalGradientAffineComponent>",

            "params": np.float32([
                -0.2769823, -0.134262, 0.409532, 0.1670311, -0.8215401, -0.3783705, 0.4978216, 0.1224703,
                -0.2013802, -0.4612775, 0.26199, 0.5464359, -0.269195, -0.5417937, 0.9573851, 0.2480077,
                -0.1859604, -0.1845462, 0.3674807, 0.7308291, 0.4376447, -0.3486033, -0.3534056, 0.2793548,
                -0.1895945, 0.4526098, -0.2292254, 0.02418093, 0.2487966, 0.1441187, 0.4238513, -0.2426005,
                0.4365571, -0.3693313, 0.398768, 0.03210105, -0.04330832, -0.5015105, 0.4480547, 0.2110117,
                0.8164227, 0.08367506, 0.7391468, -0.1286156, -0.06986938, 0.4504189, 0.04167435, -0.5613189,
                0.2165364, 0.767347, -0.1081066, -0.2189954, -0.247826, 0.2619543, 0.09545019, 0.1621316,
                0.3819256, 0.3603185, -0.1729193, 0.2073097, -0.04167856, 1.164522, -0.4983679, -0.09316859,
            ]).reshape(8, 8),

            "bias": np.float32([
                -0.3988283, 0.005634499, 2.268849, -0.4222976, 0.1610796, -0.6409795, 0.4560543, 1.776013,
            ]),
        },
        {
            "name": "tdnn5.relu",
            "type": "<RectifiedLinearComponent>",
            "dim": 8,
            "value-avg": np.array([], dtype=np.float32),
            "deriv-avg": np.array([], dtype=np.float32),
            "count": 10.0,
            "oderiv-rms": np.array([], dtype=np.float32),
            "oderiv-count": 0,
        },
        {
            "name": "tdnn5.batchnorm",
            "type": "<BatchNormComponent>",
            "dim": 8,
            "block-dim": 8,
            "epsilon": 0.001,
            "target-rms": 1.0,
            "test-mode": True,
            "count": 10.0,
            "stats-mean": np.float32([5.525478, 5.038039, 2.2788742, 5.2100024, 7.274048, 2.927882062, 3.962218457, 4.214613874]),
            "stats-var": np.float32([3.806511, 3.4790154, 0.86363363, 3.8433418, 5.611824, 1.3066893, 2.3252077, 2.600586]),
        },
        {
            "name": "output.affine",
            "type": "<NaturalGradientAffineComponent>",

            "params": np.float32([
                -0.3126723, -0.07071986, -0.1102077, -0.01302463, -0.3115547, 0.06287698, 0.1327096, 0.09239054,
            ]).reshape(1, 8),

            "bias": np.float32([
                0,
            ]),
        },
    ]

    inputs = np.float32([
        0.1, 0.2, 0.3, 
        0.2, 0.4, 0.6, 
        0.3, 0.6, 0.9, 
        0.4, 0.8, 0.12, 
        0.5, 0.10, 0.15, 
        0.6, 0.12, 0.18, 
        0.7, 0.14, 0.21, 
        0.8, 0.16, 0.24, 
    ]).reshape(1, 8, 3)

    outputs = np.float32([
        1.458157, 1.458771, 1.457126, 1.458507, 1.457815, 1.458788, 1.457821, 1.458223,
    ]).reshape(1, 8, 1)

    @classmethod
    def weights(cls, name: str) -> list:
        """
        Returns the list of weights / parameters of the component
        with the given name.

        Parameters
        ----------
        name : str
            Name of the component.

        Returns
        -------
        list
            List of weights.

        Raises
        ------
        ValueError
            If component name does not exist.
        """
        notFound = True
        for c in cls.components:
            if c["name"] == name:
                notFound = False
                break
                
        if notFound:
            raise ValueError(f"no component found with name '{name}'")

        t = c["type"]
        if t == "<NaturalGradientAffineComponent>":
            kernel, bias = c["params"], c["bias"]             
            return [kernel, bias]
        if t == "<BatchNormComponent>":
            gamma, mean, var = c["target-rms"], c["stats-mean"], c["stats-var"]
            return [gamma, mean, var]
        else:
            return []
