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


class RefPldaScores:
    ark = {
        "seg-0000-0150": np.float32([0.3972324, 0.3409896, 0.3823311, 0.3828984, 0.3392047, 0.3992842, 0.3913662, 0.03234625, -0.1581558, -0.1563786, 0.3690189, -0.04268643, -0.3984621, -0.3745084, -0.3277929, 0.3935258, 0.2776859, 0.3991854, 0.3834253, 0.2847214, 0.3690947, -0.08635055, -0.2226363, 0.3967836, 0.2709937, -0.07077265, -0.1706422, -0.1385034, 0.3974413]),
        "seg-0075-0225": np.float32([0.3409896, 0.5036027, 0.2651681, 0.2673337, 0.5044478, 0.397155, 0.4421276, 0.384763, 0.1334489, 0.1366013, 0.2201344, -0.4815654, -0.7859191, -0.7855399, -0.7657872, 0.4347242, 0.5143452, 0.3987044, 0.4614355, 0.5146314, 0.2203677, -0.534921, -0.681999, 0.3373216, 0.5138171, 0.2688367, 0.1107997, 0.16734, 0.4153258]),
        "seg-0150-0300": np.float32([0.3823311, 0.2651681, 0.3855725, 0.3856576, 0.2626998, 0.368676, 0.3457864, -0.08213122, -0.2421254, -0.2408698, 0.3819052, 0.06942306, -0.2822847, -0.2534513, -0.2020939, 0.350624, 0.1832996, 0.3681048, 0.3302881, 0.191959, 0.3819329, 0.03039246, -0.09653093, 0.3828287, 0.1751394, -0.1747299, -0.2508059, -0.2279654, 0.3611224]),
        "seg-0225-0375": np.float32([0.3828984, 0.2673337, 0.3856576, 0.3857555, 0.2648832, 0.3696589, 0.3471633, -0.07902693, -0.2398938, -0.2386238, 0.3817329, 0.0665491, -0.2853633, -0.2566489, -0.205398, 0.3519306, 0.1859414, 0.3691001, 0.3318632, 0.1945596, 0.3817619, 0.02738663, -0.09981624, 0.383371, 0.1778184, -0.1719387, -0.2486794, -0.2255815, 0.3622558]),
        "seg-0300-0450": np.float32([0.3392047, 0.5044478, 0.2626998, 0.2648832, 0.5053264, 0.3959939, 0.4415919, 0.3884715, 0.1369847, 0.140148, 0.2173184, -0.4871059, -0.7900862, -0.7900594, -0.7707484, 0.4340741, 0.5161641, 0.3975626, 0.4612283, 0.5163587, 0.2175534, -0.5404983, -0.6874447, 0.3355001, 0.5157202, 0.2726771, 0.1142535, 0.1709841, 0.4143995]),
        "seg-0375-0525": np.float32([0.3992842, 0.397155, 0.368676, 0.3696589, 0.3959939, 0.4150659, 0.4203608, 0.1277381, -0.08517943, -0.08298674, 0.3470776, -0.1463836, -0.499433, -0.4803942, -0.4387893, 0.4201471, 0.3510475, 0.4153819, 0.4191423, 0.3565527, 0.3471948, -0.1934959, -0.335905, 0.398012, 0.3457448, 0.01767321, -0.1007094, -0.0611756, 0.4182473]),
        "seg-0450-0600": np.float32([0.3913662, 0.4421276, 0.3457864, 0.3471633, 0.4415919, 0.4203608, 0.4384901, 0.2156006, -0.01486758, -0.01231338, 0.3163505, -0.2510819, -0.5955675, -0.5818723, -0.5461817, 0.4359619, 0.4134847, 0.421078, 0.443851, 0.4174124, 0.3165067, -0.3009431, -0.4472968, 0.3893054, 0.4096195, 0.1009788, -0.03305899, 0.01289864, 0.4284083]),
        "seg-0525-0675": np.float32([0.03234625, 0.384763, -0.08213122, -0.07902693, 0.3884715, 0.1277381, 0.2156006, 0.6001274, 0.4585283, 0.4612132, -0.1449262, -0.8768089, -0.9166782, -0.9568681, -0.9947541, 0.2000965, 0.4842046, 0.130555, 0.2588697, 0.475589, -0.1446089, -0.9154117, -0.9957891, 0.0264893, 0.4919872, 0.5562944, 0.4387519, 0.4864781, 0.1615957]),
        "seg-0600-0750": np.float32([-0.1581558, 0.1334489, -0.2421254, -0.2398938, 0.1369847, -0.08517943, -0.01486758, 0.4585283, 0.4894406, 0.490153, -0.2867672, -0.7260743, -0.5696027, -0.6225188, -0.6857253, -0.02752971, 0.2357671, -0.08297563, 0.02114571, 0.2261068, -0.2865441, -0.7398016, -0.7443227, -0.1625407, 0.2446769, 0.4979513, 0.4836825, 0.4959143, -0.05848283]),
        "seg-0675-0825": np.float32([-0.1563786, 0.1366013, -0.2408698, -0.2386238, 0.140148, -0.08298674, -0.01231338, 0.4612132, 0.490153, 0.4908903, -0.2858064, -0.7291022, -0.5744601, -0.6273273, -0.6903679, -0.02503749, 0.2391452, -0.080771, 0.02386791, 0.2294723, -0.2855816, -0.7431172, -0.7484506, -0.1607896, 0.2480647, 0.4997411, 0.4842167, 0.4968952, -0.05614804]),
        "seg-0750-0900": np.float32([0.3690189, 0.2201344, 0.3819052, 0.3817329, 0.2173184, 0.3470776, 0.3163505, -0.1449262, -0.2867672, -0.2858064, 0.3833962, 0.1256692, -0.2208514, -0.1897578, -0.1364615, 0.3225854, 0.1289408, 0.3462581, 0.2969217, 0.1384033, 0.383398, 0.0893745, -0.03161222, 0.3700178, 0.1200576, -0.2308857, -0.2933029, -0.2757224, 0.3365271]),
        "seg-0825-0975": np.float32([-0.04268643, -0.4815654, 0.06942306, 0.0665491, -0.4871059, -0.1463836, -0.2510819, -0.8768089, -0.7260743, -0.7291022, 0.1256692, 0.5410075, 0.4132958, 0.4493655, 0.4936876, -0.2319237, -0.6389253, -0.1495985, -0.3061866, -0.6244979, 0.1253939, 0.5477742, 0.5403435, -0.03663493, -0.6521072, -0.8372244, -0.7038931, -0.7577638, -0.185633]),
        "seg-0900-1050": np.float32([-0.3984621, -0.7859191, -0.2822847, -0.2853633, -0.7900862, -0.499433, -0.5955675, -0.9166782, -0.5696027, -0.5744601, -0.2208514, 0.4132958, 0.5827605, 0.5871996, 0.5807735, -0.5784062, -0.8941388, -0.5024706, -0.6438426, -0.8852497, -0.2211578, 0.4478103, 0.5364591, -0.3923851, -0.9020284, -0.7722299, -0.5346573, -0.6216604, -0.5361454]),
        "seg-0975-1125": np.float32([-0.3745084, -0.7855399, -0.2534513, -0.2566489, -0.7900594, -0.4803942, -0.5818723, -0.9568681, -0.6225188, -0.6273273, -0.1897578, 0.4493655, 0.5871996, 0.5965658, 0.5971088, -0.5637038, -0.9046406, -0.4835901, -0.6331217, -0.8946628, -0.190075, 0.4821761, 0.5629622, -0.3681563, -0.913546, -0.8209126, -0.5878654, -0.6739399, -0.5190643]),
        "seg-1050-1200": np.float32([-0.3277929, -0.7657872, -0.2020939, -0.205398, -0.7707484, -0.4387893, -0.5461817, -0.9947541, -0.6857253, -0.6903679, -0.1364615, 0.4936876, 0.5807735, 0.5971088, 0.6078923, -0.5268739, -0.8989553, -0.4421556, -0.6008548, -0.8875302, -0.1367874, 0.5230148, 0.5897679, -0.3211659, -0.9092189, -0.8740255, -0.6521802, -0.7352064, -0.4795878]),
        "seg-1125-1275": np.float32([0.3935258, 0.4347242, 0.350624, 0.3519306, 0.4340741, 0.4201471, 0.4359619, 0.2000965, -0.02752971, -0.02503749, 0.3225854, -0.2319237, -0.5784062, -0.5637038, -0.5268739, 0.4338517, 0.4028735, 0.4207921, 0.4401325, 0.4070927, 0.3227346, -0.2813352, -0.4271308, 0.3916064, 0.3987424, 0.08612889, -0.04526375, -0.000406465, 0.4273182]),
        "seg-1200-1350": np.float32([0.2776859, 0.5143452, 0.1832996, 0.1859414, 0.5161641, 0.3510475, 0.4134847, 0.4842046, 0.2357671, 0.2391452, 0.1289408, -0.6389253, -0.8941388, -0.9046406, -0.8989553, 0.4028735, 0.553674, 0.3531312, 0.4420593, 0.5512367, 0.1292198, -0.6922168, -0.8319663, 0.2730157, 0.5556594, 0.3759557, 0.2113665, 0.2718354, 0.3757486]),
        "seg-1275-1425": np.float32([0.3991854, 0.3987044, 0.3681048, 0.3691001, 0.3975626, 0.4153819, 0.421078, 0.130555, -0.08297563, -0.080771, 0.3462581, -0.1495985, -0.5024706, -0.4835901, -0.4421556, 0.4207921, 0.3531312, 0.4157105, 0.4200645, 0.3585885, 0.3463766, -0.1968059, -0.339369, 0.3978884, 0.3478721, 0.02031425, -0.09859335, -0.05884709, 0.4187157]),
        "seg-1350-1500": np.float32([0.3834253, 0.4614355, 0.3302881, 0.3318632, 0.4612283, 0.4191423, 0.443851, 0.2588697, 0.02114571, 0.02386791, 0.2969217, -0.3061866, -0.6438426, -0.6331217, -0.6008548, 0.4401325, 0.4420593, 0.4200645, 0.4526052, 0.4451407, 0.2970974, -0.3572095, -0.5047591, 0.3809649, 0.4389674, 0.1428149, 0.001711681, 0.05064919, 0.4296784]),
        "seg-1425-1575": np.float32([0.2847214, 0.5146314, 0.191959, 0.1945596, 0.5163587, 0.3565527, 0.4174124, 0.475589, 0.2261068, 0.2294723, 0.1384033, -0.6244979, -0.8852497, -0.8946628, -0.8875302, 0.4070927, 0.5512367, 0.3585885, 0.4451407, 0.5490623, 0.1386783, -0.6779068, -0.8186907, 0.2801397, 0.5529787, 0.3662395, 0.2018095, 0.2620647, 0.3806666]),
        "seg-1500-1650": np.float32([0.3690947, 0.2203677, 0.3819329, 0.3817619, 0.2175534, 0.3471948, 0.3165067, -0.1446089, -0.2865441, -0.2855816, 0.383398, 0.1253939, -0.2211578, -0.190075, -0.1367874, 0.3227346, 0.1292198, 0.3463766, 0.2970974, 0.1386783, 0.3834, 0.08908508, -0.03193297, 0.3700912, 0.1203401, -0.2306034, -0.2930906, -0.2754833, 0.3366592]),
        "seg-1575-1725": np.float32([-0.08635055, -0.534921, 0.03039246, 0.02738663, -0.5404983, -0.1934959, -0.3009431, -0.9154117, -0.7398016, -0.7431172, 0.0893745, 0.5477742, 0.4478103, 0.4821761, 0.5230148, -0.2813352, -0.6922168, -0.1968059, -0.3572095, -0.6779068, 0.08908508, 0.5581322, 0.561565, -0.08007417, -0.7052668, -0.8638993, -0.7155747, -0.7746191, -0.2338596]),
        "seg-1650-1800": np.float32([-0.2226363, -0.681999, -0.09653093, -0.09981624, -0.6874447, -0.335905, -0.4472968, -0.9957891, -0.7443227, -0.7484506, -0.03161222, 0.5403435, 0.5364591, 0.5629622, 0.5897679, -0.4271308, -0.8319663, -0.339369, -0.5047591, -0.8186907, -0.03193297, 0.561565, 0.5996416, -0.2159307, -0.8439907, -0.9063313, -0.7143534, -0.7880449, -0.3780047]),
        "seg-1725-1875": np.float32([0.3967836, 0.3373216, 0.3828287, 0.383371, 0.3355001, 0.398012, 0.3893054, 0.0264893, -0.1625407, -0.1607896, 0.3700178, -0.03663493, -0.3923851, -0.3681563, -0.3211659, 0.3916064, 0.2730157, 0.3978884, 0.3809649, 0.2801397, 0.3700912, -0.08007417, -0.2159307, 0.3963842, 0.2662434, -0.07614538, -0.174836, -0.1431627, 0.3958687]),
        "seg-1800-1950": np.float32([0.2709937, 0.5138171, 0.1751394, 0.1778184, 0.5157202, 0.3457448, 0.4096195, 0.4919872, 0.2446769, 0.2480647, 0.1200576, -0.6521072, -0.9020284, -0.913546, -0.9092189, 0.3987424, 0.5556594, 0.3478721, 0.4389674, 0.5529787, 0.1203401, -0.7052668, -0.8439907, 0.2662434, 0.5578703, 0.384832, 0.2201943, 0.2808264, 0.3709803]),
        "seg-1875-2025": np.float32([-0.07077265, 0.2688367, -0.1747299, -0.1719387, 0.2726771, 0.01767321, 0.1009788, 0.5562944, 0.4979513, 0.4997411, -0.2308857, -0.8372244, -0.7722299, -0.8209126, -0.8740255, 0.08612889, 0.3759557, 0.02031425, 0.1428149, 0.3662395, -0.2306034, -0.8638993, -0.9063313, -0.07614538, 0.384832, 0.5540276, 0.4845028, 0.5160952, 0.04954024]),
        "seg-1950-2100": np.float32([-0.1706422, 0.1107997, -0.2508059, -0.2486794, 0.1142535, -0.1007094, -0.03305899, 0.4387519, 0.4836825, 0.4842167, -0.2933029, -0.7038931, -0.5346573, -0.5878654, -0.6521802, -0.04526375, 0.2113665, -0.09859335, 0.001711681, 0.2018095, -0.2930906, -0.7155747, -0.7143534, -0.174836, 0.2201943, 0.4845028, 0.4792017, 0.4882347, -0.07505732]),
        "seg-2025-2175": np.float32([-0.1385034, 0.16734, -0.2279654, -0.2255815, 0.1709841, -0.0611756, 0.01289864, 0.4864781, 0.4959143, 0.4968952, -0.2757224, -0.7577638, -0.6216604, -0.6739399, -0.7352064, -0.000406465, 0.2718354, -0.05884709, 0.05064919, 0.2620647, -0.2754833, -0.7746191, -0.7880449, -0.1431627, 0.2808264, 0.5160952, 0.4882347, 0.5052868, -0.03299678]),
        "seg-2100-2248": np.float32([0.3974413, 0.4153258, 0.3611224, 0.3622558, 0.4143995, 0.4182473, 0.4284083, 0.1615957, -0.05848283, -0.05614804, 0.3365271, -0.185633, -0.5361454, -0.5190643, -0.4795878, 0.4273182, 0.3757486, 0.4187157, 0.4296784, 0.3806666, 0.3366592, -0.2338596, -0.3780047, 0.3958687, 0.3709803, 0.04954024, -0.07505732, -0.03299678, 0.4232754]),
    }

    arkWithoutPCA = {
        "seg-0000-0150": np.float32([47.06849, 23.59407, 20.12508, 26.04021, 22.03778, 21.5614, 20.52911, 11.35007, 7.698378, 5.150657, 10.00016, 0.9732885, -4.150854, 0.2301702, -2.12312, 17.9878, 14.2687, 19.93838, 21.76958, 12.07997, 18.48372, 9.113974, 3.250712, 18.07258, 16.86334, 8.164557, 0.8279532, 2.199572, 13.89641]),
        "seg-0075-0225": np.float32([23.59407, 43.88758, 20.83966, 10.53899, 18.48118, 8.017858, 9.603593, 12.14174, 6.608181, 5.220404, 1.504818, -6.582832, -4.127761, -2.876651, -6.728229, 13.57406, 16.38615, 10.60349, 14.34924, 16.15437, 10.79314, 3.228273, 0.873604, 12.4916, 15.54234, 5.879812, -0.1094564, 6.488595, 8.41398]),
        "seg-0150-0300": np.float32([20.12508, 20.83966, 48.00472, 25.92203, 8.085988, 20.19846, 19.79998, 3.779383, 0.3875459, -3.010092, 11.63015, 1.359593, -4.483882, 0.1777603, -1.068776, 14.57609, 10.29615, 18.3819, 14.80412, 6.815275, 21.25406, 10.12573, 2.993007, 11.25034, 8.388752, 3.486269, -2.731537, 1.508217, 19.31315]),
        "seg-0225-0375": np.float32([26.04021, 10.53899, 25.92203, 46.97931, 30.40008, 24.81441, 23.93623, 8.072527, 9.951218, 1.592131, 13.34461, 3.040929, -4.257887, 3.092336, 2.698942, 19.06305, 14.15128, 24.30472, 26.2732, 12.68752, 22.50616, 8.418992, 1.615925, 21.90845, 16.72528, 12.20815, 6.662308, 5.595982, 19.48761]),
        "seg-0300-0450": np.float32([22.03778, 18.48118, 8.085988, 30.40008, 54.03372, 24.23197, 17.76847, 18.37745, 18.19768, 14.62263, 4.604254, -9.951577, -6.567956, -2.115815, -6.027356, 16.90472, 20.61996, 17.67267, 27.13501, 22.25138, 14.05117, -2.992819, -8.054164, 21.08607, 25.27167, 15.70531, 14.56119, 14.75139, 12.62265]),
        "seg-0375-0525": np.float32([21.5614, 8.017858, 20.19846, 24.81441, 24.23197, 47.30317, 29.13427, 12.39571, 14.53888, 11.70097, 15.99032, 2.668646, -4.017709, 0.6694348, -0.4309683, 13.85442, 13.58422, 24.86831, 22.58697, 12.26987, 21.41729, 5.863125, -3.45818, 16.37848, 14.37982, 10.71177, 16.28684, 15.71847, 20.70144]),
        "seg-0450-0600": np.float32([20.52911, 9.603593, 19.79998, 23.93623, 17.76847, 29.13427, 49.90577, 30.77362, 13.32108, 10.2555, 17.86427, 1.736521, -8.781994, -4.300763, -4.850345, 19.63623, 20.63711, 23.55264, 27.22426, 15.83939, 18.14651, 4.084166, -3.329866, 21.32762, 17.17757, 14.9773, 12.64303, 7.646043, 21.03513]),
        "seg-0525-0675": np.float32([11.35007, 12.14174, 3.779383, 8.072527, 18.37745, 12.39571, 30.77362, 54.72931, 33.58134, 19.43822, 4.777281, -10.68768, -12.16652, -9.633373, -13.71646, 10.15934, 21.33597, 10.8002, 15.76662, 16.3337, 6.273963, -6.206336, -7.954259, 16.28196, 19.74314, 21.56902, 18.44107, 9.600986, 5.247079]),
        "seg-0600-0750": np.float32([7.698378, 6.608181, 0.3875459, 9.951218, 18.19768, 14.53888, 13.32108, 33.58134, 55.44957, 35.8337, 3.089208, -10.83135, -10.21009, -6.834868, -10.14344, 6.838242, 15.87684, 12.89593, 14.10293, 13.83077, 4.715608, -4.38833, -5.168502, 12.14847, 15.47094, 22.68871, 26.31181, 19.18735, 6.52594]),
        "seg-0675-0825": np.float32([5.150657, 5.220404, -3.010092, 1.592131, 14.62263, 11.70097, 10.2555, 19.43822, 35.8337, 51.87394, 15.67828, -12.85254, -12.54571, -9.337814, -10.6601, 6.458554, 12.48237, 9.324861, 13.24786, 15.31549, -0.1171825, -7.843658, -7.780651, 5.069711, 9.734314, 20.37995, 27.67594, 22.17413, 10.92202]),
        "seg-0750-0900": np.float32([10.00016, 1.504818, 11.63015, 13.34461, 4.604254, 15.99032, 17.86427, 4.777281, 3.089208, 15.67828, 45.81387, 18.15894, -1.12582, 2.313715, 6.548772, 15.17908, 9.197289, 15.75735, 13.02566, 5.552843, 14.37276, 10.28347, 5.0632, 12.72446, 4.780853, 9.751192, 5.887, 6.127333, 21.56773]),
        "seg-0825-0975": np.float32([0.9732885, -6.582832, 1.359593, 3.040929, -9.951577, 2.668646, 1.736521, -10.68768, -10.83135, -12.85254, 18.15894, 48.74586, 26.58524, 13.5046, 17.21192, 7.852512, -2.941522, 3.465961, -2.062602, -10.08162, 5.393537, 19.73314, 17.24785, 9.557104, -6.356449, -12.21082, -14.98369, -11.68273, 0.2625484]),
        "seg-0900-1050": np.float32([-4.150854, -4.127761, -4.483882, -4.257887, -6.567956, -4.017709, -8.781994, -12.16652, -10.21009, -12.54571, -1.12582, 26.58524, 48.21519, 30.10303, 13.11853, 2.597962, -1.874078, -3.921988, -7.517746, -6.621152, -0.03715836, 12.85677, 13.94308, 3.133375, -6.163746, -14.53098, -15.38485, -9.598403, -8.586892]),
        "seg-0975-1125": np.float32([0.2301702, -2.876651, 0.1777603, 3.092336, -2.115815, 0.6694348, -4.300763, -9.633373, -6.834868, -9.337814, 2.313715, 13.5046, 30.10303, 48.16653, 27.11709, 4.437562, -1.20532, 0.1018105, -3.580126, -6.192767, 5.04591, 13.3729, 12.25586, 8.406878, -3.240687, -8.60241, -7.915169, -5.362964, -3.119526]),
        "seg-1050-1200": np.float32([-2.12312, -6.728229, -1.068776, 2.698942, -6.027356, -0.4309683, -4.850345, -13.71646, -10.14344, -10.6601, 6.548772, 17.21192, 13.11853, 27.11709, 53.96874, 16.1737, -6.69091, -0.09268691, -4.730345, -9.768769, 0.8740888, 13.28816, 14.87069, 8.062669, -7.657321, -12.67736, -8.711327, -5.459363, 1.015196]),
        "seg-1125-1275": np.float32([17.9878, 13.57406, 14.57609, 19.06305, 16.90472, 13.85442, 19.63623, 10.15934, 6.838242, 6.458554, 15.17908, 7.852512, 2.597962, 4.437562, 16.1737, 49.37877, 32.00689, 17.59833, 22.85701, 17.92584, 13.61373, 9.439522, 5.470896, 20.75849, 17.67146, 9.067924, 1.910408, 6.145671, 18.29778]),
        "seg-1200-1350": np.float32([14.2687, 16.38615, 10.29615, 14.15128, 20.61996, 13.58422, 20.63711, 21.33597, 15.87684, 12.48237, 9.197289, -2.941522, -1.874078, -1.20532, -6.69091, 32.00689, 50.65928, 27.84736, 26.76013, 27.51053, 13.47344, 0.7428427, -3.856814, 18.68457, 26.02597, 17.92646, 8.494267, 9.906923, 13.0773]),
        "seg-1275-1425": np.float32([19.93838, 10.60349, 18.3819, 24.30472, 17.67267, 24.86831, 23.55264, 10.8002, 12.89593, 9.324861, 15.75735, 3.465961, -3.921988, 0.1018105, -0.09268691, 17.59833, 27.84736, 48.49231, 35.36256, 15.3508, 18.40238, 8.304958, -0.353349, 17.36737, 14.61368, 14.17513, 12.21771, 10.42378, 22.85445]),
        "seg-1350-1500": np.float32([21.76958, 14.34924, 14.80412, 26.2732, 27.13501, 22.58697, 27.22426, 15.76662, 14.10293, 13.24786, 13.02566, -2.062602, -7.517746, -3.580126, -4.730345, 22.85701, 26.76013, 35.36256, 51.54482, 33.44912, 17.86165, 1.291159, -4.723728, 21.50542, 21.80781, 18.23394, 12.54363, 13.40983, 20.50827]),
        "seg-1425-1575": np.float32([12.07997, 16.15437, 6.815275, 12.68752, 22.25138, 12.26987, 15.83939, 16.3337, 13.83077, 15.31549, 5.552843, -10.08162, -6.621152, -6.192767, -9.768769, 17.92584, 27.51053, 15.3508, 33.44912, 47.49252, 20.47537, -5.983632, -6.232026, 15.84658, 18.69941, 13.58472, 8.228183, 13.17058, 11.70217]),
        "seg-1500-1650": np.float32([18.48372, 10.79314, 21.25406, 22.50616, 14.05117, 21.41729, 18.14651, 6.273963, 4.715608, -0.1171825, 14.37276, 5.393537, -0.03715836, 5.04591, 0.8740888, 13.61373, 13.47344, 18.40238, 17.86165, 20.47537, 44.85411, 16.58058, 1.131631, 18.05991, 10.30096, 2.502191, 1.172583, 5.755553, 16.265]),
        "seg-1575-1725": np.float32([9.113974, 3.228273, 10.12573, 8.418992, -2.992819, 5.863125, 4.084166, -6.206336, -4.38833, -7.843658, 10.28347, 19.73314, 12.85677, 13.3729, 13.28816, 9.439522, 0.7428427, 8.304958, 1.291159, -5.983632, 16.58058, 47.70871, 34.5579, 8.046446, -4.81701, -7.192653, -8.878605, -7.589932, 3.19087]),
        "seg-1650-1800": np.float32([3.250712, 0.873604, 2.993007, 1.615925, -8.054164, -3.45818, -3.329866, -7.954259, -5.168502, -7.780651, 5.0632, 17.24785, 13.94308, 12.25586, 14.87069, 5.470896, -3.856814, -0.353349, -4.723728, -6.232026, 1.131631, 34.5579, 50.70931, 16.4794, -9.473855, -9.476904, -11.23821, -11.15984, -2.795127]),
        "seg-1725-1875": np.float32([18.07258, 12.4916, 11.25034, 21.90845, 21.08607, 16.37848, 21.32762, 16.28196, 12.14847, 5.069711, 12.72446, 9.557104, 3.133375, 8.406878, 8.062669, 20.75849, 18.68457, 17.36737, 21.50542, 15.84658, 18.05991, 8.046446, 16.4794, 52.19976, 29.15395, 9.984087, 7.921965, 8.694613, 13.35735]),
        "seg-1800-1950": np.float32([16.86334, 15.54234, 8.388752, 16.72528, 25.27167, 14.37982, 17.17757, 19.74314, 15.47094, 9.734314, 4.780853, -6.356449, -6.163746, -3.240687, -7.657321, 17.67146, 26.02597, 14.61368, 21.80781, 18.69941, 10.30096, -4.81701, -9.473855, 29.15395, 52.28794, 30.02391, 8.8108, 7.138403, 6.753395]),
        "seg-1875-2025": np.float32([8.164557, 5.879812, 3.486269, 12.20815, 15.70531, 10.71177, 14.9773, 21.56902, 22.68871, 20.37995, 9.751192, -12.21082, -14.53098, -8.60241, -12.67736, 9.067924, 17.92646, 14.17513, 18.23394, 13.58472, 2.502191, -7.192653, -9.476904, 9.984087, 30.02391, 53.3923, 28.4342, 8.682467, 9.592747]),
        "seg-1950-2100": np.float32([0.8279532, -0.1094564, -2.731537, 6.662308, 14.56119, 16.28684, 12.64303, 18.44107, 26.31181, 27.67594, 5.887, -14.98369, -15.38485, -7.915169, -8.711327, 1.910408, 8.494267, 12.21771, 12.54363, 8.228183, 1.172583, -8.878605, -11.23821, 7.921965, 8.8108, 28.4342, 56.34903, 35.39927, 10.31298]),
        "seg-2025-2175": np.float32([2.199572, 6.488595, 1.508217, 5.595982, 14.75139, 15.71847, 7.646043, 9.600986, 19.18735, 22.17413, 6.127333, -11.68273, -9.598403, -5.362964, -5.459363, 6.145671, 9.906923, 10.42378, 13.40983, 13.17058, 5.755553, -7.589932, -11.15984, 8.694613, 7.138403, 8.682467, 35.39927, 49.79538, 22.52544]),
        "seg-2100-2248": np.float32([13.89641, 8.41398, 19.31315, 19.48761, 12.62265, 20.70144, 21.03513, 5.247079, 6.52594, 10.92202, 21.56773, 0.2625484, -8.586892, -3.119526, 1.015196, 18.29778, 13.0773, 22.85445, 20.50827, 11.70217, 16.265, 3.19087, -2.795127, 13.35735, 6.753395, 9.592747, 10.31298, 22.52544, 46.83135]),
    }

    @classmethod
    def scores(cls, withoutPCA=True) -> np.ndarray:
        """
        Stacks the reference PLDA scores returns them as a single 3D numpy array
        with shape (batch, batch).
        """
        if withoutPCA:
            pldaScores = np.stack(list(cls.arkWithoutPCA.values()), axis=0)
        else:
            pldaScores = np.stack(list(cls.ark.values()), axis=0)

        return pldaScores
