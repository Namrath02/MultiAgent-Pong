import gym, roboschool, sys
import numpy as np

def relu(x):
    return np.maximum(x, 0)

class SmallReactivePolicy:
    "Simple multi-layer perceptron policy, no internal state"
    def __init__(self, ob_space, ac_space):
        assert weights_dense1_w.shape == (ob_space.shape[0], 64)
        assert weights_dense2_w.shape == (64, 32)
        assert weights_final_w.shape  == (32, ac_space.shape[0])

    def act(self, ob):
        x = ob
        x = relu(np.dot(x, weights_dense1_w) + weights_dense1_b)
        x = relu(np.dot(x, weights_dense2_w) + weights_dense2_b)
        x = np.dot(x, weights_final_w) + weights_final_b
        return x

def demo_run():
    env = gym.make("RoboschoolPong-v1")
    if len(sys.argv)==3: env.unwrapped.multiplayer(env, sys.argv[1], player_n=int(sys.argv[2]))

    pi = SmallReactivePolicy(env.observation_space, env.action_space)

    while 1:
        frame = 0
        score = 0
        restart_delay = 0
        obs = env.reset()

        while 1:
            a = pi.act(obs)
            obs, r, done, _ = env.step(a)
            score += r
            frame += 1
            still_open = env.render("human")
            if still_open==False:
                return
            if not done: continue
            if restart_delay==0:
                print("score=%0.2f in %i frames" % (score, frame))
                if still_open!=True:      # not True in multiplayer or non-Roboschool environment
                    break
                restart_delay = 60*2  # 2 sec at 60 fps
            restart_delay -= 1
            if restart_delay==0: break

weights_dense1_w = np.array([
[ +0.1715, +0.3430, -0.1723, +0.6869, -0.4260, +0.3379, -0.4932, -0.2240, -0.1503, -0.2176, -0.5173, +0.4112, +0.0467, -0.1937, -0.1825, -0.3676, -0.1045, +0.0464, +0.3762, -0.3076, +0.1724, +0.0389, -0.1453, +0.2589, +0.2910, +0.2362, -0.0911, +0.4609, -0.1081, -0.3154, +0.2379, -0.2510, -0.4019, -0.1754, -0.4161, -0.2357, +0.4940, +0.3861, -0.2083, -0.1130, -0.2249, -0.5164, -0.1231, +0.1612, -0.0409, +0.1221, -0.0651, -0.1466, +0.2345, +0.7348, +0.2424, -0.0668, -0.0123, +0.1321, +0.1579, -0.1631, -0.1473, +0.0403, +0.5718, +0.3684, -0.2947, +0.0277, +0.0361, -0.0569],
[ -0.3879, +0.4405, -0.2423, -0.1405, -0.3188, -0.5470, +0.0214, -0.4809, +0.0636, +0.4443, -0.1630, +0.1426, -0.8021, +0.0242, -0.1597, -0.2045, +0.1917, -0.1223, -0.2168, +0.1189, -0.0578, -0.2438, +0.1610, +0.5382, -0.1014, -0.5141, +0.3928, +0.3471, -0.0082, +0.4213, -0.0997, +0.2823, +0.5097, +0.2156, -0.0482, +0.8383, -0.3505, -0.0103, +0.0411, +0.1111, -0.2970, +0.1082, +0.4691, -0.1831, +0.3548, +0.0505, +0.0471, +0.3495, +0.4937, +0.1256, -0.6137, -0.3205, -0.1358, +0.0913, -0.5111, +0.0237, +0.2465, -0.0321, +0.1105, +0.3070, +0.4970, -0.3082, +0.4196, -0.0266],
[ -0.4682, +0.8344, +0.7754, -0.0885, +0.3332, -0.1328, -0.7459, +0.1683, -0.1345, -0.2486, -0.2852, +0.2404, -0.3460, +0.4109, +0.8857, +0.7358, +0.1575, +0.7849, -0.4010, +0.1327, +0.1769, +0.5453, -0.5207, +0.1889, +0.3144, +0.5622, -0.7073, +0.3013, -0.4706, -0.4760, -0.2877, -0.2597, -0.3318, +0.8102, +0.2319, -0.2195, -0.3957, -0.6564, -0.6077, -0.3090, +0.8570, -0.3647, +0.0313, -0.3227, -0.2782, -0.9195, -0.0045, +0.3847, +0.0776, -0.7084, -0.4239, +0.7884, +0.8796, +0.3884, +0.4421, -0.2443, +0.1071, +0.6321, +0.5521, -0.0388, +0.2713, -0.3827, +0.5716, -0.3569],
[ +0.0390, +0.1779, +0.3441, -0.2578, +0.0697, -0.1651, +0.1739, -0.1989, -0.5347, +0.7128, -0.0652, +0.4095, -0.7970, +0.0980, +0.1492, -0.4740, -0.0162, -0.2794, +0.2002, -0.3060, -0.5371, +0.3831, +0.1752, +0.4124, +0.2820, -0.4654, -0.2594, +0.7359, -0.4635, +0.2446, -0.4071, +0.1975, +0.0525, +0.1070, +0.3467, +0.2561, +0.4631, -0.1364, +0.4012, +0.4728, +0.4685, +0.1617, +0.1451, -0.0916, +0.0870, -0.2206, -0.1410, -0.3459, -0.0176, -0.1772, +0.2144, +0.0508, +0.2687, +0.0599, -0.0338, -0.2766, -0.4633, -0.2338, -0.1131, +0.4612, -0.5116, -0.0752, -0.4700, -0.0418],
[ +0.1092, -0.0082, -0.0750, +0.4731, +0.1430, -0.0354, -0.4332, -0.0884, +0.2608, -0.1213, +0.3761, -0.4892, +0.1399, +0.1955, +0.0047, -0.0286, +0.2872, +0.1445, -0.9362, -0.2958, -0.1207, +0.4364, +0.3295, -0.0486, +0.3631, +0.1021, +0.0942, +0.0450, +0.2406, -0.2762, -0.1684, +0.4545, +0.0129, -0.2798, +0.3391, +0.2302, -0.1548, +0.0892, -0.0573, -0.0494, -0.1249, +0.2231, +0.1756, -0.4544, +0.2691, -0.1091, +0.1779, +0.1909, -0.4220, +0.0725, +0.1836, +0.0258, -0.2695, +0.0501, +0.1141, -0.2029, +0.1376, +0.1211, -0.0079, -0.0944, -0.0084, -0.3572, +0.2244, +0.0485],
[ -0.1010, +0.0229, -0.2688, +0.1397, -0.5427, -0.1314, +0.2333, +0.0254, +0.1935, -0.3542, -0.0782, -0.3569, -0.0390, +0.3769, +0.2505, -0.4655, -0.3666, +0.2580, +0.2376, -0.8522, +0.1415, -0.5660, -0.2562, +0.0452, +0.1214, -0.0863, +0.2725, -0.0504, -0.4608, -0.0014, +0.3117, -0.1108, -0.2150, +0.1778, +0.5537, -0.1378, -0.2386, -0.1028, -0.4310, -0.0526, +0.1667, -0.0909, -0.8169, +0.9034, -0.2599, +0.6675, -0.8480, +0.2862, -0.4230, -0.0543, -0.5248, +0.0702, -0.3535, -0.3697, +0.1767, +0.3260, +0.1699, +0.0511, +0.2891, +0.2237, -0.0363, -0.0991, +0.4014, +0.4126],
[ -0.2165, -0.1171, +0.0359, +0.1477, -0.7248, -0.5255, -0.4897, +0.1368, +0.5969, -0.4360, +0.3122, +0.1539, +0.4166, +0.6801, -0.1392, -0.3555, -0.4065, -0.2243, -0.0501, +0.7695, +0.0425, -0.0788, -0.0713, +0.4650, +0.2619, -0.1702, -0.7497, -0.3926, -0.4373, -0.1167, +0.1391, -0.0965, -0.4499, +0.1296, +0.0225, -0.0730, +0.0260, -0.1804, +0.3822, -0.4282, -0.2530, -0.0129, -0.1080, +0.2951, -0.1607, +0.2365, -0.1189, -0.2262, -0.1207, -0.1437, -0.1587, +0.5148, -0.0341, +0.0622, +0.2806, +0.5174, -0.5722, -0.2129, -0.1345, -0.1628, -0.1323, -0.5364, +0.3289, +0.4275],
[ -0.2978, +0.0781, +0.3430, +0.3655, -0.3578, +0.4007, +0.8692, +0.1722, -0.1771, -0.4616, +0.1160, +0.0644, +0.1481, -0.2210, -0.0719, -0.1069, -0.3300, +0.0228, -0.0854, +0.1616, +0.1891, +0.2822, +0.1121, +0.3733, +0.5426, -0.2902, -0.1898, +0.3228, +0.7167, +0.1781, -0.0988, -0.1842, +0.0063, +0.6496, -0.0856, +0.0772, -0.1344, -0.3857, +0.1416, -0.3645, -0.2357, +0.2054, -0.0890, -0.2773, +0.1188, +0.1404, +0.5177, -0.3012, +0.0242, +0.2491, +0.2247, -0.3442, -0.1733, -0.5917, -0.0855, -0.0503, +0.5804, -0.0191, +0.1993, -0.2371, -0.0748, -0.3553, +0.2350, -0.6054],
[ +0.0841, -0.4213, -0.4923, +0.3267, -0.3253, -0.1432, -0.2700, -0.8175, -0.5522, -0.5016, -0.3012, +0.5216, +0.2477, -0.8067, -0.2482, +0.4715, -0.0870, +0.5045, +0.1427, +0.9746, -0.5189, +0.1352, -0.1384, -0.6531, +0.1040, +0.6055, -0.0317, +0.6961, +0.3667, +0.6287, -1.1134, +0.3032, -0.0439, +0.5414, +0.9091, -1.1206, -0.5608, +0.1486, +0.4649, -0.9176, -0.0565, +0.4982, +0.3449, -0.3049, -0.8357, -0.3019, -0.4940, +0.9378, -0.8251, +0.0559, -0.1795, +0.1622, -0.0250, -0.2724, -0.1221, +0.1283, +0.0154, -0.2728, +0.0511, -0.2059, +0.0570, +0.8983, -0.1112, -0.6098],
[ -0.5296, -0.2024, -0.0967, -0.1495, +0.1043, -0.1663, -0.0203, -0.3506, +0.1288, -0.1015, -0.0522, +0.2403, -0.2120, +0.1749, -0.0027, +0.1707, -0.5416, -0.3525, +0.0768, +0.0528, -0.0519, -0.1066, -0.2628, -0.0024, +0.0903, -0.2304, -0.1880, -0.0540, -0.0167, -0.0951, +0.0433, -0.3013, +0.5276, -0.0973, +0.1387, -0.4228, -0.1384, -0.2436, -0.3687, +0.0074, -0.2497, +0.4062, +0.3306, +0.0084, +0.1087, -0.4373, +0.0913, -0.2891, -0.5075, +0.1031, -0.2372, -0.3543, -0.3336, +0.1474, -0.4196, -0.0728, -0.4983, +0.2209, +0.2552, +0.4151, -0.1412, -0.0173, +0.0086, -0.2625],
[ +0.3189, -0.7581, -0.7602, -0.6903, +0.1939, -0.3229, -0.2305, -0.8411, +0.4983, +0.3843, +0.4820, -0.1261, -0.2968, -0.2709, -0.8530, -0.2846, +0.5322, -0.4092, +0.0357, +0.3819, -0.7893, -0.1649, +0.3798, +0.1721, -0.4132, -0.2832, +0.5674, +0.4594, -0.3192, -0.0482, +0.2082, -0.3406, +0.1877, +0.2116, +0.1664, +0.1198, +0.6267, -0.8385, +0.4550, +0.7804, -0.3628, -0.8434, +0.1835, +0.0156, -0.6174, +0.4785, +0.3030, -0.7702, +0.1135, -0.4491, -0.5616, -0.0407, -0.9976, -0.2187, -0.8404, +0.7266, +0.1467, -0.6710, +0.3033, +0.2092, -1.1333, -0.7641, -0.7143, -0.1171],
[ +0.6517, +0.1177, +0.3439, -0.3474, +0.3031, -0.1858, -0.1435, -0.5892, +0.2752, -0.4344, +0.1030, -0.3065, +0.0119, -0.0417, -0.4321, +0.2610, -0.0772, -0.0959, +0.2074, -0.0033, -0.5398, -0.4220, +0.7012, +0.6062, +0.3992, +0.1469, -0.3395, +0.4394, -0.0686, -0.6500, +0.3763, -0.2518, +0.1873, -0.0577, -0.1933, -0.4154, -0.2012, -0.5817, +0.2178, +0.1248, +0.1297, -0.8407, +0.2822, -0.1211, +0.2437, +0.2114, +0.0084, -0.3846, +0.1005, +0.6060, +0.1618, -0.2808, +0.1500, -0.6019, -0.3509, +0.6103, -0.0811, +0.6274, -0.4414, -0.1852, -0.0847, -0.2162, -0.0407, -0.1041],
[ -0.5132, +0.0234, -0.0248, -0.2470, +0.2560, -0.4169, +0.1530, +0.0429, -0.1029, -0.4455, -0.6013, -0.3303, +0.1179, -0.0465, -0.2597, -0.1817, -0.2351, -0.4252, +0.1884, +0.0757, +0.3710, +0.0239, -0.6107, +0.1455, -0.3824, +0.3134, +0.1273, -0.0007, +0.3064, +0.7165, -0.6974, +0.3200, +0.0477, +0.3128, +0.0510, -0.3025, +0.5113, +0.3175, -0.3866, +0.2624, +0.1353, -0.1343, -0.3281, -0.1714, +0.3179, +0.0139, +0.2618, -0.1950, -0.0554, +0.4341, -0.0572, -0.0237, -0.0120, -0.0417, +0.3571, -0.0446, +0.0878, -0.1722, -0.5348, -0.1272, -0.4186, -0.2447, -0.2781, +0.1118]
])

weights_dense1_b = np.array([ +0.0956, +0.1585, +0.0679, -0.0669, -0.2784, -0.3509, +0.3316, -0.0575, +0.2816, +0.1454, +0.2677, -0.0568, -0.0427, +0.0555, +0.2666, +0.1150, -0.2540, -0.0923, +0.0168, -0.1438, -0.2684, -0.0911, +0.0437, -0.2436, -0.2069, +0.2031, -0.0137, -0.1365, -0.2034, +0.1394, +0.2452, +0.1841, -0.0623, -0.0679, +0.2172, -0.1698, +0.1305, -0.1792, +0.2689, +0.0251, +0.1834, -0.0285, -0.1062, +0.4480, +0.2345, -0.0761, +0.0011, -0.0906, +0.0445, -0.0401, -0.0945, -0.1218, +0.1120, -0.0728, -0.1278, +0.2343, +0.0229, +0.0364, -0.0436, -0.0779, +0.1035, -0.1843, +0.0757, +0.1377])

weights_dense2_w = np.array([
[ +0.1890, +0.1360, +0.0861, +0.1788, -0.2116, -0.2345, +0.1622, -0.1981, -0.1149, +0.0635, +0.0497, +0.0633, -0.0739, -0.0086, +0.1646, -0.3889, +0.0193, -0.3933, +0.0257, +0.0901, +0.1097, -0.1006, +0.1869, +0.0631, +0.1634, +0.0766, -0.1123, +0.1147, +0.0243, +0.1395, +0.1625, +0.1185],
[ -0.1964, +0.2863, -0.2986, +0.0978, +0.1232, +0.4640, -0.1394, +0.2038, +0.0084, -0.4016, -0.2803, +0.5603, +0.3830, -0.0055, +0.3135, -0.0776, +0.3857, +0.3793, -0.2383, -0.3924, -0.0198, +0.0412, -0.4467, -0.2741, -0.3243, -0.4111, +0.1804, -0.2027, +0.0946, -0.2746, -0.1824, -0.3386],
[ -0.1115, +0.2006, -0.3155, -0.0721, +0.0660, +0.3983, -0.2715, +0.3115, -0.0363, -0.3400, -0.2476, +0.4923, +0.4011, -0.0873, +0.1354, -0.2817, +0.0012, +0.5288, -0.2536, -0.1042, +0.3522, +0.2423, -0.2632, -0.1927, -0.0287, +0.0722, +0.3484, -0.4166, +0.1632, -0.3094, -0.1764, -0.4646],
[ +0.1403, +0.2567, +0.0526, +0.2236, -0.0495, +0.0673, -0.5005, +0.0452, -0.2518, +0.2014, -0.0215, +0.0915, -0.0594, +0.1187, -0.2205, -0.1564, -0.1004, +0.0066, +0.0581, -0.0603, +0.1073, +0.0091, +0.2609, -0.0399, +0.0636, -0.2252, +0.0655, -0.2393, -0.0067, +0.1328, -0.1656, -0.0471],
[ +0.0830, -0.1515, -0.0809, -0.0276, +0.2401, -0.1397, +0.0507, -0.1173, +0.1939, -0.2468, +0.1852, -0.3666, +0.0936, -0.2177, +0.0093, -0.0018, +0.2516, -0.2792, -0.4748, +0.2176, -0.1053, -0.0210, -0.2795, +0.1746, -0.0224, +0.1043, +0.1884, -0.1956, +0.0330, -0.1706, +0.4415, -0.1594],
[ -0.0451, +0.3128, -0.5779, +0.2754, +0.2269, -0.0332, -0.1704, +0.2442, -0.1268, +0.0592, -0.1208, -0.0484, +0.1412, +0.0875, -0.3159, -0.0748, -0.2162, +0.0671, -0.1487, -0.0611, +0.1728, +0.1793, -0.1583, -0.0563, +0.0060, -0.5107, -0.0032, -0.0226, +0.3364, +0.1224, -0.2969, +0.4833],
[ +0.3756, -0.2796, +0.1604, -0.3159, -0.0602, +0.0354, +0.3072, -0.1317, +0.0808, +0.0402, -0.0497, -0.0281, +0.1460, -0.2525, -0.4313, +0.1513, +0.1866, +0.2356, -0.3847, +0.0908, +0.1222, +0.2788, -0.0262, -0.0173, +0.1014, -0.5622, -0.0528, +0.0871, +0.0221, +0.3104, +0.2293, -0.2402],
[ -0.1344, +0.0022, -0.3414, -0.1041, +0.2794, +0.2239, -0.1946, +0.0081, +0.4607, -0.0131, +0.0855, +0.2923, +0.1824, -0.2406, +0.2315, +0.3070, +0.3585, +0.2683, -0.2662, +0.0341, -0.1751, +0.1095, -0.1154, +0.1554, -0.0533, -0.1937, +0.0353, -0.1561, -0.3375, -0.2228, -0.0426, +0.1497],
[ +0.0576, -0.0155, +0.3271, +0.0078, -0.1162, -0.2591, +0.1321, -0.1731, +0.0475, +0.1766, +0.0095, +0.1152, -0.1146, +0.3403, -0.1887, +0.0591, -0.0288, +0.0363, +0.0614, +0.2335, -0.1532, +0.2219, -0.2828, +0.1529, -0.1213, +0.1987, -0.3419, +0.0244, -0.0817, +0.0495, +0.2012, +0.0682],
[ +0.0767, +0.0230, +0.2161, +0.0879, -0.0092, +0.1019, +0.0895, +0.2258, +0.0017, -0.0174, -0.0032, +0.0298, +0.1101, +0.2144, -0.2732, +0.1997, +0.0824, +0.1358, +0.4202, +0.3286, -0.3572, +0.2908, -0.1839, +0.0607, +0.1423, -0.3761, -0.2829, +0.2290, -0.0536, +0.0928, +0.0459, +0.0365],
[ -0.0230, -0.0752, +0.3295, -0.1061, +0.1414, -0.1301, +0.2319, -0.0149, +0.1470, +0.0554, +0.1899, -0.3831, -0.0885, +0.0812, +0.0230, -0.1439, -0.0761, -0.1400, +0.1539, -0.0467, -0.0707, +0.1540, +0.0078, +0.0866, -0.2487, +0.1218, +0.0884, +0.0163, -0.1987, +0.1389, +0.2080, +0.1366],
[ +0.1302, +0.1048, -0.0473, +0.2571, +0.1424, +0.1357, +0.1654, +0.1774, -0.1950, -0.0690, -0.2288, +0.2519, -0.0419, +0.3965, +0.3769, -0.0914, -0.0155, +0.0416, -0.1865, -0.2475, +0.3013, -0.2246, -0.1029, -0.1932, -0.2433, -0.2161, -0.0551, +0.0435, +0.1625, +0.0142, -0.3569, +0.2131],
[ +0.0203, +0.3741, +0.0676, -0.3026, -0.3862, -0.2954, +0.1226, -0.3069, -0.0810, +0.5236, +0.2686, +0.1319, -0.1076, -0.1010, +0.0361, +0.0332, -0.0975, +0.2315, -0.3056, +0.0209, -0.5751, +0.1506, +0.2271, +0.3197, +0.0679, -0.0884, -0.1675, +0.1747, -0.2314, +0.1960, +0.0657, +0.0171],
[ -0.1073, -0.1672, -0.1370, -0.0318, +0.2383, +0.2522, -0.1787, -0.0019, -0.0292, +0.0740, -0.0327, -0.0338, +0.3576, -0.2701, +0.1201, +0.0110, +0.2854, +0.1259, +0.0756, +0.1064, +0.1077, +0.3114, -0.0951, +0.3923, -0.0779, +0.1838, +0.0847, -0.3515, +0.2159, -0.1762, -0.0222, +0.0961],
[ -0.2921, +0.3037, -0.1961, +0.0599, +0.0638, +0.1292, -0.7197, +0.1240, -0.2985, +0.0195, -0.1948, +0.1978, +0.2401, -0.0710, +0.4129, +0.0746, +0.1027, +0.3386, -0.1796, +0.0306, +0.0315, +0.3053, -0.2726, +0.1031, +0.0169, +0.1945, +0.3867, -0.6120, -0.1108, +0.0076, -0.4864, -0.2438],
[ +0.1826, +0.0508, -0.0571, +0.0178, -0.0212, +0.1302, +0.0755, -0.1985, -0.5466, -0.2225, -0.1200, -0.0158, -0.4032, +0.1625, -0.0796, +0.0075, -0.2723, -0.0667, +0.0248, +0.0234, -0.1108, -0.5298, +0.3197, -0.4849, +0.3532, +0.0058, -0.0614, +0.2420, +0.1702, -0.0033, +0.2919, +0.1096],
[ -0.2559, -0.1363, +0.0229, +0.3451, -0.0290, +0.0496, +0.2860, +0.1528, +0.4027, -0.2992, -0.1703, -0.2360, -0.0425, +0.0635, -0.0778, -0.0705, +0.0194, -0.1441, +0.0272, -0.0387, -0.2596, +0.1964, +0.2181, +0.1563, -0.1889, -0.1423, +0.1661, +0.1407, -0.0310, -0.3919, +0.1191, +0.0466],
[ -0.0174, +0.0564, -0.2934, -0.0871, +0.1976, +0.1446, -0.1383, +0.0369, -0.0014, -0.2447, +0.0546, +0.3209, +0.2523, -0.1607, +0.1617, -0.2328, -0.1683, +0.0346, -0.1944, -0.2145, -0.0947, -0.1601, +0.1103, -0.0317, -0.0364, -0.0807, +0.1327, -0.1589, +0.0323, -0.1220, -0.0522, +0.1414],
[ +0.1008, +0.1634, -0.0808, -0.0695, -0.0506, +0.1878, +0.1258, +0.0364, -0.1744, +0.1842, +0.0040, -0.1610, -0.1965, -0.1180, +0.3027, +0.2602, -0.2615, +0.2204, -0.7618, -0.0345, +0.0196, -0.3156, +0.3960, +0.2344, +0.1386, +0.1492, -0.2507, +0.2092, -0.1826, +0.0814, -0.1700, -0.2030],
[ -0.5055, -0.5526, -0.1480, +0.0915, +0.0490, -0.0084, +0.1435, +0.1183, -0.1029, +0.3267, -0.0883, -0.0402, -0.3069, -0.0693, +0.0065, -0.6557, -0.2083, +0.0208, -0.0552, +0.1318, +0.2464, -0.5812, +0.2057, +0.2531, +0.0219, +0.1481, +0.4057, +0.0175, -0.3342, +0.2475, -0.1158, +0.2756],
[ -0.0922, +0.0747, -0.2041, +0.0395, -0.2935, -0.1778, -0.1964, -0.0411, +0.0359, +0.1562, -0.1222, +0.0500, +0.1481, -0.3695, -0.0436, +0.1310, +0.2494, +0.2868, -0.3530, +0.0203, -0.1527, +0.3245, -0.0213, -0.1594, -0.2779, -0.0657, +0.0764, -0.2960, -0.0156, -0.0642, -0.4337, -0.2643],
[ -0.0351, +0.0816, +0.0195, +0.2005, -0.2058, +0.2570, -0.4350, -0.1454, +0.0395, +0.3473, -0.0333, -0.3469, +0.1290, -0.0212, +0.0940, +0.0819, -0.1572, +0.2670, +0.2010, -0.1522, +0.2812, -0.2386, -0.1985, -0.1724, -0.0065, +0.0883, +0.0296, -0.2453, -0.1585, -0.4612, +0.0549, +0.1197],
[ -0.3008, -0.0685, -0.1640, +0.0115, -0.0253, -0.3882, +0.0991, -0.1231, +0.0885, +0.2783, +0.0455, -0.3638, -0.1627, -0.0204, -0.1317, -0.4790, +0.1747, -0.0548, +0.1953, +0.0163, -0.0629, +0.0531, +0.1324, +0.1924, +0.0174, -0.3031, -0.4526, +0.1290, +0.1278, -0.0469, +0.0706, +0.0432],
[ -0.1133, +0.0754, +0.1583, +0.0904, -0.2356, -0.3040, -0.0064, +0.0790, -0.0630, +0.0249, -0.0625, -0.2619, +0.0407, -0.1144, +0.0234, +0.1648, +0.3380, +0.0208, -0.4268, -0.0312, +0.1948, +0.2207, -0.0391, +0.1819, -0.2349, -0.3568, -0.3114, -0.0632, +0.0595, -0.0753, -0.1394, -0.0616],
[ +0.1890, +0.2513, +0.1088, -0.0654, -0.2217, -0.1496, +0.1195, -0.2250, -0.0976, -0.0893, +0.2962, +0.1540, +0.0971, -0.1541, -0.3417, +0.1156, +0.1639, -0.2374, +0.2755, -0.0554, -0.2154, +0.0203, +0.2022, +0.0022, -0.0366, -0.0443, -0.0964, +0.0938, +0.1084, -0.1098, +0.0994, +0.0365],
[ -0.3200, +0.0704, -0.3281, -0.0116, -0.0719, -0.2834, +0.0265, -0.3017, -0.2485, +0.1360, +0.1673, +0.2372, -0.1458, -0.0519, +0.1835, +0.2586, -0.2465, -0.1010, +0.0720, +0.0870, +0.0201, -0.1765, +0.2266, -0.1606, +0.1540, -0.0004, -0.0021, -0.0728, +0.1112, +0.0459, +0.0416, -0.0833],
[ -0.0889, -0.1881, +0.2352, +0.2830, -0.1353, -0.3780, +0.0887, +0.0187, +0.2900, +0.0390, +0.1153, -0.1858, -0.0876, -0.0813, -0.2946, -0.1013, +0.1154, -0.2894, +0.0793, +0.2558, -1.0968, +0.1856, +0.1056, +0.3513, +0.3182, -0.6020, -0.1035, +0.1922, +0.0591, +0.0100, -0.0078, -0.2869],
[ +0.1260, +0.2636, -0.0299, +0.5393, +0.2762, -0.3897, -0.1344, +0.4990, -0.1322, +0.4129, -0.4283, -0.1488, -0.5046, +0.4760, +0.2211, +0.0322, -0.0415, -0.4524, -0.1079, -0.2253, +0.3433, -0.1878, -0.1874, -0.2750, +0.0976, -0.6354, -0.0080, +0.2861, -0.1426, -0.1134, -0.0524, -0.1032],
[ -0.1123, +0.0984, -0.2271, -0.2927, -0.2101, -0.1359, +0.2709, -0.2863, +0.1169, -0.8859, +0.3437, -0.1530, -0.1924, -0.2076, -0.6525, +0.0509, +0.0225, -0.0224, -0.0506, +0.1261, +0.1202, -0.0489, +0.1965, -0.0636, -0.2801, -0.3235, -0.3324, -0.0639, +0.0101, +0.1949, +0.1886, +0.2435],
[ +0.5202, -0.0788, +0.0935, +0.1206, +0.0943, +0.2376, +0.0199, +0.0809, -0.1000, +0.4435, +0.2826, +0.0791, -0.1278, +0.0175, -0.2423, +0.5007, -0.2973, +0.1188, -0.1962, -0.2312, -0.1387, -0.1505, -0.1178, -0.1892, +0.4293, -0.6195, -0.2000, +0.1742, -0.2877, +0.3701, -0.1670, -0.3181],
[ -0.1648, -0.1312, +0.0320, -0.0773, -0.1743, -0.0965, +0.2572, -0.4246, +0.0891, +0.2678, +0.2809, +0.1210, +0.1710, -0.6427, +0.1082, -0.0931, +0.1570, +0.1142, +0.1526, +0.2659, -0.0990, +0.3711, -0.1318, +0.2537, -0.3323, -0.0531, -0.2066, +0.1475, +0.0838, +0.0793, +0.2244, -0.0653],
[ +0.1246, +0.0388, +0.0654, -0.0375, +0.0702, +0.1590, +0.0388, +0.0419, -0.0099, -0.1131, +0.1043, +0.0267, -0.0297, +0.2015, -0.0355, +0.3598, -0.3218, +0.1836, +0.1700, +0.1589, -0.3711, -0.1516, +0.1648, -0.0222, -0.1216, +0.0662, +0.0491, -0.0211, +0.0982, +0.1932, -0.1416, +0.0357],
[ +0.1313, +0.0866, +0.1424, +0.1869, -0.0528, -0.0087, +0.4184, +0.1424, +0.0145, -0.1983, +0.0854, +0.1196, -0.3056, +0.2983, -0.0038, -0.2122, +0.1143, -0.0353, +0.0310, +0.1018, +0.1273, -0.1392, +0.1804, +0.1344, +0.0517, +0.0475, +0.1623, +0.2386, -0.0502, +0.0760, +0.0121, -0.0620],
[ -0.1744, +0.1902, -0.2564, -0.1894, +0.2875, -0.0131, -0.4636, +0.3034, -0.2198, +0.2331, -0.1147, +0.0845, -0.1451, +0.2253, +0.2001, -0.6375, -0.0982, +0.0688, -0.6139, +0.0426, +0.3580, +0.1200, -0.0811, +0.2374, -0.2239, -0.2642, +0.2734, -0.0397, -0.0496, +0.0216, -0.1365, +0.0201],
[ -0.0287, +0.0011, +0.0934, -0.0500, +0.0656, +0.2269, -0.2526, +0.4691, -0.4386, +0.1845, +0.1620, +0.2219, -0.2842, +0.0237, +0.0285, -0.0764, -0.5342, +0.1201, +0.1335, +0.0070, +0.1168, -0.5173, +0.1991, -0.2201, -0.0218, +0.1139, +0.2312, +0.0850, +0.1156, -0.3101, -0.0735, +0.1485],
[ -0.1212, -0.3350, +0.0151, -0.0794, +0.1722, +0.1593, +0.0985, -0.1714, +0.2489, -0.4316, +0.0389, -0.2168, +0.2459, +0.1028, -0.1245, -0.2668, +0.1973, +0.1023, +0.0784, +0.3656, -0.3425, +0.1624, +0.0785, +0.0381, -0.1907, -0.1127, +0.0966, -0.0702, -0.0666, +0.0225, +0.3685, -0.1164],
[ +0.4614, +0.0035, +0.3533, +0.0514, -0.3274, -0.3388, +0.4022, -0.0483, -0.3036, +0.2327, +0.0375, -0.0832, -0.1241, +0.1458, -0.3680, +0.1334, -0.0207, -0.1426, -0.0877, +0.1155, -0.1402, +0.0782, -0.2842, -0.3982, +0.2139, -0.2655, -0.5092, +0.5765, -0.0828, +0.5125, +0.2551, +0.2846],
[ +0.0513, -0.0188, -0.1097, +0.0815, -0.4674, +0.0365, -0.3736, -0.2433, -0.1515, -0.6254, +0.1185, +0.0361, -0.0856, -0.2980, +0.0328, +0.1963, -0.2323, -0.2783, +0.3168, +0.3180, -0.3979, -0.2125, +0.2611, +0.0349, -0.0401, -0.0717, -0.1708, -0.0302, -0.0688, +0.1530, -0.3791, -0.1662],
[ +0.2782, -0.1089, -0.0083, -0.2502, -0.0747, +0.1368, +0.1362, -0.1927, +0.2756, +0.0374, +0.1144, -0.1570, -0.1455, +0.1363, -0.2070, -0.3822, -0.0969, -0.2791, +0.3433, +0.0257, -0.0345, -0.3242, +0.0356, +0.0584, +0.2061, +0.1558, -0.0450, +0.2668, -0.2319, -0.0717, +0.1867, +0.0523],
[ -0.4126, -0.2263, +0.2913, +0.1001, -0.0374, -0.3915, -0.0163, +0.0432, +0.2957, +0.2464, -0.3147, -0.3757, -0.1751, +0.0457, +0.0477, +0.4503, +0.2309, -0.3745, -0.0731, +0.1399, -0.2736, +0.3959, -0.0575, +0.1107, +0.3201, -0.1269, -0.3734, +0.0675, -0.0345, +0.3160, +0.1516, -0.0670],
[ +0.1225, +0.3831, +0.0209, +0.1480, +0.1145, -0.0193, +0.0283, +0.3412, -0.1084, +0.0413, -0.1573, -0.0364, +0.2148, +0.2388, +0.1441, -0.0505, +0.2838, +0.0119, -0.1036, -0.2157, +0.2216, -0.0097, -0.2480, -0.0477, +0.4362, -0.0015, +0.0417, -0.1052, -0.0606, +0.0758, -0.0739, -0.0983],
[ +0.0201, +0.0995, -0.0029, -0.0091, -0.0461, +0.1202, -0.1419, +0.1161, -0.5639, -0.0784, +0.2608, -0.0977, -0.7572, +0.1123, -0.3354, +0.1187, +0.0130, -0.2929, -0.0588, +0.1186, +0.1876, -0.1493, -0.0051, -0.0030, -0.2094, +0.1533, -0.1115, -0.1114, -0.1006, -0.0245, -0.2629, +0.0523],
[ +0.1903, +0.0549, -0.3139, +0.4010, +0.3641, +0.0078, +0.0741, +0.0021, +0.0141, -0.0178, -0.0172, +0.0781, +0.2342, +0.2194, +0.1308, -0.3751, +0.0969, +0.1712, +0.4681, +0.0437, +0.1495, -0.4479, -0.0808, -0.5391, -0.3562, -0.1244, -0.0252, -0.0187, -0.3473, -0.1270, -0.0493, +0.1577],
[ -0.1402, -0.0898, +0.1143, -0.2459, -0.1335, +0.0016, +0.2465, -0.1913, -0.0776, +0.0574, +0.3282, +0.0824, +0.3877, -0.3828, +0.2362, +0.0339, +0.1042, -0.0917, +0.2599, +0.0290, -0.0720, -0.0552, +0.1401, +0.1802, +0.3211, +0.3155, -0.0344, +0.1719, +0.0909, +0.3074, +0.0200, -0.1496],
[ -0.0712, +0.0723, +0.2062, +0.0555, -0.0043, -0.1107, +0.2608, -0.1867, +0.2308, -0.4231, -0.1494, -0.0781, +0.3885, -0.0988, -0.0151, -0.0078, +0.1554, +0.0736, +0.0131, -0.0796, +0.0778, +0.4545, -0.5886, +0.1803, -0.1800, -0.2218, +0.2906, +0.1014, -0.1423, -0.2774, -0.1550, -0.2119],
[ +0.0333, -0.0947, +0.3064, -0.0783, -0.2067, -0.0022, +0.1319, -0.2238, -0.0065, +0.0298, +0.1971, -0.3727, -0.2704, -0.1754, -0.1347, -0.1222, -0.0381, -0.2761, +0.1860, +0.1502, -0.2801, -0.3786, -0.0090, +0.1859, +0.0125, -0.2538, +0.0149, -0.1426, +0.1591, +0.2057, +0.3822, +0.0438],
[ -0.1878, -0.4258, +0.0954, -0.1985, -0.1466, -0.0371, +0.0328, -0.2601, +0.3175, -0.1034, +0.1134, -0.1718, +0.0821, +0.1031, -0.0517, -0.0324, +0.1056, +0.1813, -0.1427, +0.0910, +0.0052, -0.1433, -0.1580, -0.1052, +0.0304, -0.1843, -0.0916, +0.1405, -0.2949, -0.2231, +0.1705, +0.0972],
[ -0.0252, -0.1619, +0.0619, +0.2052, -0.3010, +0.0441, -0.1057, -0.0297, -0.1879, -0.1726, +0.0865, +0.0969, +0.0556, -0.2345, +0.0549, +0.1745, -0.3224, +0.0415, -0.1433, +0.1348, -0.4060, -0.2099, +0.1994, +0.0298, -0.1105, -0.0288, +0.3039, +0.0499, +0.2916, -0.0192, -0.2249, +0.0661],
[ +0.0280, -0.1213, +0.0419, +0.2211, +0.0369, +0.1539, +0.3985, -0.1923, +0.1406, -0.1604, -0.3583, +0.0383, +0.2458, -0.2204, +0.3049, -0.2537, +0.1774, +0.0577, +0.1064, +0.0345, +0.1516, -0.0600, -0.2957, +0.0665, -0.3605, -0.0029, +0.1357, +0.1428, -0.2488, +0.1092, +0.0598, -0.1731],
[ -0.2581, -0.1445, -0.0163, +0.3312, -0.2132, -0.2409, +0.0676, -0.2511, +0.1081, +0.0544, +0.2332, -0.5912, +0.0189, -0.0390, -0.0965, +0.0100, -0.1135, -0.2133, -0.2888, -0.1221, -0.1665, +0.2848, +0.2432, +0.0203, -0.2907, -0.5324, +0.1468, +0.0028, -0.0378, -0.0795, +0.2670, -0.1578],
[ +0.1637, +0.1276, +0.0002, -0.1992, -0.1263, +0.2018, +0.0786, -0.1780, +0.1065, -0.2347, +0.2097, -0.0312, +0.1827, -0.2064, -0.1037, +0.1330, -0.3397, +0.0563, +0.1672, -0.1861, -0.0882, +0.2191, +0.2495, -0.1904, +0.1715, -0.0241, -0.0622, +0.1257, +0.3820, +0.0179, +0.2855, +0.1586],
[ -0.3363, -0.1810, -0.2252, +0.1486, +0.4597, +0.0894, +0.0298, +0.1197, +0.1056, +0.0638, -0.0118, +0.0722, +0.1176, +0.1429, -0.0280, +0.0714, +0.0803, -0.0176, -0.0073, -0.1247, -0.0168, -0.0039, +0.0456, +0.0214, +0.0025, +0.2455, +0.0699, -0.0852, -0.0361, -0.1000, -0.0750, -0.1551],
[ -0.0854, +0.2189, -0.4675, +0.0103, +0.3181, +0.2001, -0.4294, +0.2941, +0.1224, -0.1800, -0.2498, +0.0979, +0.2272, +0.0443, +0.0349, +0.0713, +0.0075, +0.3209, -0.4157, -0.4582, +0.0953, -0.1221, -0.1097, -0.1843, -0.0115, +0.2911, +0.2542, -0.5041, +0.0848, -0.3456, -0.2521, -0.2192],
[ +0.0247, +0.2705, -0.0255, +0.0719, +0.1114, +0.2039, -0.3252, -0.0547, -0.0997, +0.1575, +0.0914, -0.1542, +0.2559, +0.1077, +0.1726, +0.0758, +0.4121, -0.0312, -0.1317, +0.0114, +0.1578, +0.0429, -0.1497, -0.2746, -0.1709, +0.2868, +0.1604, -0.0005, -0.0732, +0.0657, -0.0453, -0.1091],
[ +0.2483, +0.4488, +0.0387, -0.0621, -0.0943, +0.0918, -0.1701, +0.0280, +0.0727, -0.1223, +0.1569, +0.0854, +0.0176, +0.0074, -0.0053, -0.0446, -0.1916, +0.3639, -0.0138, +0.0761, -0.1081, -0.2277, -0.0600, +0.1638, +0.0358, +0.2976, +0.0384, -0.0672, +0.1897, -0.1515, -0.2685, +0.2573],
[ +0.1452, -0.0768, +0.0243, -0.0880, +0.1488, -0.0276, +0.1265, -0.2369, -0.0817, +0.2025, +0.2168, -0.2964, -0.5773, +0.0660, +0.1342, +0.0630, -0.0557, -0.5637, -0.1535, +0.0960, +0.0634, -0.1720, +0.1647, -0.0490, +0.1168, -0.2331, -0.1733, -0.1793, +0.2466, +0.2034, -0.1551, -0.1006],
[ +0.1160, +0.1557, -0.0204, -0.3106, -0.1961, -0.0685, +0.0985, -0.0395, +0.2089, -0.0798, +0.0492, +0.3205, -0.0873, -0.1147, -0.1371, +0.0305, +0.0851, -0.0196, -0.1657, +0.2321, -0.1699, -0.0559, -0.1822, +0.2553, -0.0340, -0.2578, -0.0883, +0.2352, +0.0139, +0.2568, +0.1391, +0.0413],
[ +0.0896, +0.2081, -0.0377, -0.0613, -0.0050, +0.2374, -0.2581, -0.3751, +0.0862, +0.2179, +0.0216, +0.1896, +0.0731, -0.0064, +0.1599, +0.2472, +0.1401, -0.1878, +0.1971, -0.2624, +0.0003, -0.1499, -0.1140, -0.1432, +0.0272, -0.0492, +0.1056, +0.0094, +0.0600, -0.0899, +0.3079, -0.1557],
[ +0.2409, +0.2036, +0.2814, -0.0476, -0.2997, -0.0229, -0.0805, -0.3044, +0.0230, +0.1583, +0.2614, +0.1988, -0.0942, +0.1667, -0.1107, +0.1752, +0.0742, +0.1689, +0.1218, -0.0032, +0.0207, +0.0929, +0.0126, -0.0848, +0.1601, +0.0111, -0.6141, -0.0470, +0.1951, +0.0334, -0.2866, +0.1149],
[ +0.3135, -0.2216, +0.1090, -0.1127, +0.0257, -0.0887, +0.0066, -0.0923, -0.0301, -0.1257, +0.1835, -0.1039, -0.0012, +0.0935, -0.0586, +0.0452, -0.0835, -0.3501, +0.0138, -0.0693, -0.0515, -0.0114, -0.0809, +0.2569, -0.3182, +0.0122, +0.0195, -0.1493, +0.1769, +0.0397, -0.0895, +0.4386],
[ -0.1411, +0.2402, -0.1837, +0.1055, -0.1056, +0.2582, -0.0903, -0.1234, -0.1138, +0.0214, -0.0994, +0.0773, +0.0463, -0.0839, -0.0005, +0.3280, -0.0141, +0.0682, -0.2516, -0.1888, -0.4852, -0.1103, -0.1682, -0.2578, -0.1548, +0.1737, +0.2033, -0.1499, +0.0755, +0.1786, -0.4708, -0.2370],
[ -0.1413, -0.1630, +0.0718, -0.2197, +0.1200, -0.1735, +0.0076, +0.0069, -0.0096, -0.8168, +0.3710, -0.1511, -0.0460, +0.0475, +0.0458, +0.1354, +0.2707, -0.2806, -0.2195, +0.0222, +0.0271, -0.1187, +0.1954, +0.6445, -0.0756, -0.4692, -0.0296, -0.3425, +0.1969, +0.1783, -0.2357, -0.1911],
[ -0.0079, +0.4597, -0.2066, -0.0751, +0.0641, +0.3587, -0.3661, +0.2621, -0.1075, +0.0470, -0.0972, +0.2921, +0.0345, +0.2161, -0.0577, -0.0310, -0.1471, +0.2685, -0.3131, +0.0076, -0.2024, +0.0517, -0.3903, +0.0656, -0.2742, +0.1809, -0.1478, +0.0432, +0.1022, +0.1213, -0.1730, -0.3178],
[ -0.3692, -0.2057, +0.0985, +0.2165, -0.0063, -0.0193, +0.0632, +0.1948, +0.0226, +0.0165, +0.0406, -0.3100, +0.2302, +0.0462, +0.0855, -0.0627, +0.1428, -0.0345, +0.0593, -0.0945, -0.3383, -0.0747, +0.0098, +0.2213, +0.0484, -0.3283, -0.1571, +0.1521, +0.0209, -0.0860, -0.0877, +0.3086]
])

weights_dense2_b = np.array([ +0.1856, +0.1115, +0.1399, -0.0426, -0.0536, +0.1358, +0.2432, -0.0510, -0.1811, +0.0105, +0.0539, +0.0492, +0.0821, +0.0200, -0.0715, +0.0019, +0.0052, +0.0595, +0.2304, +0.0537, -0.0272, +0.0028, -0.0984, -0.0203, +0.1556, +0.0764, -0.0009, +0.2806, -0.0556, +0.2308, +0.1566, -0.2276])

weights_final_w = np.array([
[ -0.3403, +0.0057],
[ -0.3772, -0.1451],
[ -0.0535, +0.2728],
[ -0.1106, -0.1768],
[ +0.1661, -0.2392],
[ -0.1308, -0.3234],
[ +0.0003, +0.5431],
[ +0.1697, -0.4058],
[ +0.2599, -0.0038],
[ -0.3103, -0.5291],
[ +0.0968, +0.1898],
[ -0.2000, -0.3484],
[ +0.1957, -0.4929],
[ -0.1165, -0.2444],
[ -0.0052, -0.3040],
[ -0.2119, +0.2763],
[ +0.3335, -0.1746],
[ -0.0653, -0.4166],
[ -0.5782, +0.3655],
[ +0.1343, +0.1123],
[ -0.0604, -0.3140],
[ +0.4648, -0.1214],
[ +0.3947, +0.0790],
[ +0.2235, +0.0933],
[ -0.3459, +0.0403],
[ -0.1523, -0.4867],
[ +0.0600, -0.3450],
[ -0.3517, +0.3145],
[ -0.1395, +0.0624],
[ -0.0249, +0.3032],
[ +0.0181, +0.5399],
[ +0.2308, +0.2665]
])

weights_final_b = np.array([ -0.1013, +0.0489])

if __name__=="__main__":
    demo_run()