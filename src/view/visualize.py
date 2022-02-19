import matplotlib.pyplot as plt


def visualize(current_frame, candidates, auxilary, traffic_lights, traffic_auxilary, distance):
    fig, (phase1, phase2, phase3) = plt.subplots(1, 3, figsize=(12, 6))

    phase1.set_title('phase 1 results')
    phase1.imshow(current_frame.img)
    for i in range(len(candidates)):
        if auxilary[i] == "red":
            phase1.plot(candidates[i][0], candidates[i][1], 'r+')
        else:
            phase1.plot(candidates[i][0], candidates[i][1], 'g+')

    phase2.set_title('phase 2 results')
    phase2.imshow(current_frame.img)
    for i in range(len(traffic_lights)):
        if traffic_auxilary[i] == "red":
            phase2.plot(traffic_lights[i][0], traffic_lights[i][1], 'r+')
        else:
            phase2.plot(traffic_lights[i][0], traffic_lights[i][1], 'g+')

    phase3.set_title('phase 3 results')
    if distance is not None:
        phase3.imshow(current_frame.img)
        for i in range(len(distance)):
            phase3.text(traffic_lights[i][0], traffic_lights[i][1], r'{0:.1f}'.format(distance[i]), color="orange", fontsize=7)

    plt.show()

